"""
Whale Sightings Data Module (Currently Tailored for Acartia API)
---------------------------------------------------------------

This module provides a flexible framework for retrieving, processing,
standardizing, and exporting whale sighting data. While currently
tailored specifically for the Acartia API, it is designed with
extensibility in mind to support whale sighting data from other sources.

Key Features:
- Query the Acartia API with authentication to fetch recent whale sightings.
- Convert raw API JSON responses into clean, standardized pandas DataFrames.
- Normalize marine mammal type descriptions for consistent classification.
- Preprocess data with enriched time-based features such as year, week, and month.
- Export data efficiently to a partitioned Parquet dataset for downstream analysis.
- Includes a main orchestration function to execute the full data pipeline end-to-end.

Usage Notes:
- Requires an API token set in the environment variable `ACARTIA_API_TOKEN` for Acartia data.
- Supports automatic loading of environment variables from a `.env` file.
- Can be run as a standalone script or imported for customized workflows.

Usage (CLI):
    $ python whale_sightings.py --source ACARTIA

    Or run with debug logging:
    $ python whale_sightings.py --source ACARTIA --debug

Environment Variables Required:
    - ACARTIA_API_URL
    - ACARTIA_API_TOKEN

Dependencies:
- pandas
- requests
- pyarrow
- python-dotenv

Author: Tyler Stevenson
Date: 2025-07-19
Version: 1.0
"""

# ------------------------------------------------- #
#                      MODULES                      #

# Standard Library
import os
import sys
import logging
import argparse
from pathlib import Path
import re
from typing import Optional, List, Dict
from abc import ABC, abstractmethod

# Third-Party Libraries
import pandas as pd
import pyarrow.dataset as ds
from dotenv import load_dotenv
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

#                                                   #
# ------------------------------------------------- #


# ------------------------------------------------- #
#                     FUNCTIONS                     #

######################################
############### GENERAL ##############


# Setup Logging
def setup_logging(debug: bool = False) -> None:
    """
    Configures logging for the whale sightings module.

    Args:
        debug (bool): If True, set logging level to DEBUG. Defaults to False (INFO).
    """
    # Remove existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    current_dir = Path(__file__).resolve().parents[1]
    log_folder = current_dir.parent / "logs"
    log_folder.mkdir(parents=True, exist_ok=True)

    log_file_path = log_folder / "sightings_metadata.log"

    log_level = logging.DEBUG if debug else logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(),
        ],
    )


# Get Environmental Variable
def get_env_var(name: str, default: Optional[str] = None) -> str:
    """
    Fetch an environment variable or raise an error if missing and no default provided.

    Args:
        name (str): Environment variable name.
        default (Optional[str]): Default value if env var not set.

    Returns:
        str: Environment variable value or default.

    Raises:
        EnvironmentError: If environment variable is missing and no default given.
    """
    value = os.getenv(name, default)
    if value is None:
        logging.error(f"Missing required environment variable: {name}")
        raise EnvironmentError(f"⚠️ Missing required environment variable: {name}")

    if default is not None and value == default:
        logging.warning(f"Using default value for environment variable: {name}")
    else:
        logging.debug(f"Environment variable loaded: {name} = {value[:4]}...")

    return value


# Argument Parser
def create_arg_parser():
    """
    Parses command-line arguments for the Whale Sightings CLI tool.

    Command-line arguments:
    --source : str, optional
        The data source to fetch whale sightings from.
        Supported values: 'ACARTIA'.
        Default is 'ACARTIA'.

    --debug : bool, optional
        If set, enables debug-level logging output for more verbose logs.

    Returns:
        argparse.Namespace: Parsed arguments with attributes 'source' and 'debug'.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Whale Sightings Data Pipeline CLI\n\n"
            "This tool fetches, processes, and exports whale sightings data.\n"
            "Currently supports the 'ACARTIA' data source by default.\n\n"
            "Example usage:\n"
            "  python whale_sightings.py --source ACARTIA --output-dir ./data --verbose\n"
            "  python whale_sightings.py --source ACARTIA --dry-run\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-s",
        "--source",
        type=str,
        default="ACARTIA",
        choices=["ACARTIA"],
        help="Data source to use for fetching whale sightings. Default is 'ACARTIA'.",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory path where processed data will be saved. "
            "If not specified, defaults to the project data folder."
        ),
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Path to a local CSV or Parquet file to use instead of querying the API.",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging output."
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the pipeline without exporting data. Useful for testing.",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="Whale Sightings CLI 1.0",
        help="Show the version of the tool and exit.",
    )

    # Parse Args
    args = parser.parse_args()

    return args


######################################
######################################

######################################
######## DATA SOURCE: ACARTIA ########


# Query Acartia API
def query_acartia_api(
    api_url: str, token: str, retries: int = 3, backoff_factor: float = 0.3
) -> Optional[List[Dict]]:
    """
    Queries the Acartia API for whale sightings data.

    Args:
        api_url (str): Endpoint URL.
        token (str): API token.
        retries (int): Number of retry attempts on failure.
        backoff_factor (float): Time factor for exponential backoff between retries.

    Returns:
        Optional[List[Dict]]: JSON response or None on failure.

    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    with requests.Session() as session:
        retry_strategy = Retry(
            total=retries,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            backoff_factor=backoff_factor,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        try:
            response = session.get(api_url, headers=headers, timeout=10)
            response.raise_for_status()
            logging.info(f"Successfully queried Acartia API: {api_url}")
            return response.json()
        except requests.exceptions.HTTPError as e:
            status_code = getattr(e.response, "status_code", "N/A")
            logging.error(f"HTTP error occurred: {e} (Status code: {status_code})")
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed: {e}")
        except ValueError:
            logging.error("Failed to parse JSON response.")

    return None


# Collect Data from Acartia Response
def collect_data_from_acartia_response(data: List[Dict]) -> Optional[pd.DataFrame]:
    """
    Converts a list of JSON-like dictionaries from the Acartia API into a DataFrame.

    Args:
        data (list): Response from the Acartia API. Expected to be a list of dictionaries.

    Returns:
        pd.DataFrame | None: Parsed response as a DataFrame, or None if no data is returned.
    """
    if not data:
        logging.warning("No Acartia sightings returned in the past 7 days.")
        return None

    if not isinstance(data, list):
        logging.error(f"API response is not a list. Got type: {type(data)}")
        return None

    # Check all elements are dicts, or at least the first few
    if not all(
        isinstance(item, dict) for item in data[:5]
    ):  # check first 5 for efficiency
        logging.error("API response list does not contain dictionaries as expected.")
        return None

    # Optionally check required keys exist in the first dict
    required_keys = {"type", "created", "latitude", "longitude"}
    first_keys = set(data[0].keys())
    missing_keys = required_keys - first_keys
    if missing_keys:
        logging.error(
            f"API response dictionaries missing required keys: {missing_keys}"
        )
        return None

    logging.info(f"Collected {len(data)} records from Acartia response.")
    return pd.DataFrame(data)


# Standardize Whale Type Naming
def standardize_acartia_type_description_vectorized(
    type_series: pd.Series,
    translation_dict: dict[str, str] | None = None,
) -> pd.Series:
    """
    Vectorized standardization of whale/marine mammal type descriptions,
    with optional translation support and logging of unmapped types.

    This function:
    - Cleans raw type description strings.
    - Applies translation using a provided dictionary (e.g., for non-English terms).
    - Normalizes terms to a predefined set of marine mammal categories.
    - Logs any values that do not match known categories.
    - Returns standardized categories as a pandas Categorical series.

    Args:
        type_series (pd.Series): Series of raw type description strings.
        translation_dict (dict[str, str] | None): Optional dictionary mapping
            language-specific or variant terms to English labels.

    Returns:
        pd.Series: Standardized type descriptions as a categorical series.
    """
    ts = type_series.astype("object").fillna("").str.lower()
    ts = ts.str.replace(r"sighting|:|\\|\(specify in comments\)|'", "", regex=True)

    # Apply translation dictionary if provided
    if translation_dict:
        pattern = "|".join(map(re.escape, filter(None, translation_dict.keys())))

        def translate_match(match):
            return translation_dict.get(match.group(0), match.group(0))

        ts = ts.str.replace(pattern, translate_match, regex=True)

    # Normalize variants
    ts = ts.str.replace("autre", "unspecified")
    ts = ts.str.replace("commun", "common")
    ts = ts.str.replace("common", "")
    ts = ts.str.replace(r"\s{2,}", " ", regex=True).str.strip()

    # Build matching mask to detect known mappings
    matched_mask = (
        ts.str.contains("orca|killer")
        | ts.str.contains("shark")
        | ts.str.contains("gray|grey")
        | ts.str.contains("minke")
        | ts.str.contains("humpback|jorobada")
        | ts.str.contains("right")
        | ts.str.contains("fin")
        | ts.str.contains("pilot")
        | ts.str.contains("beluga")
        | ts.str.contains("sei")
        | ts.str.contains("blue")
        | ts.str.contains("dolphin")
        | ts.str.contains("porpoise|rorqual")
        | ts.str.contains("sowerbys")
        | ts.str.contains("bairds")
        | ts.str.contains("sealion")
        | ts.str.contains("sperm")
        | ts.str.contains(r"other|unknown|unidentified|non spécifié|^$", regex=True)
    )

    # Log unmapped (unknown) entries
    unmatched = ts[~matched_mask].unique()
    if len(unmatched) > 0:
        logging.warning(f"Unmapped type descriptions found: {unmatched}")

    # Assign categories by matching known patterns
    result = pd.Series("unspecified", index=ts.index)
    result.loc[ts.str.contains("orca|killer")] = "orca"
    result.loc[ts.str.contains("shark")] = "shark"
    result.loc[ts.str.contains("gray|grey")] = "grey whale"
    result.loc[ts.str.contains("minke")] = "minke whale"
    result.loc[ts.str.contains("humpback|jorobada")] = "humpback whale"
    result.loc[ts.str.contains("right")] = "right whale"
    result.loc[ts.str.contains("fin")] = "fin whale"
    result.loc[ts.str.contains("pilot")] = "pilot whale"
    result.loc[ts.str.contains("beluga")] = "beluga whale"
    result.loc[ts.str.contains("sei")] = "sei whale"
    result.loc[ts.str.contains("blue")] = "blue whale"
    result.loc[ts.str.contains("dolphin")] = "dolphin"
    result.loc[ts.str.contains("porpoise|rorqual")] = "porpoise"
    result.loc[ts.str.contains("sowerbys")] = "sowerbys beaked whale"
    result.loc[ts.str.contains("bairds")] = "bairds beaked whale"
    result.loc[ts.str.contains("sealion")] = "sealion"
    result.loc[ts.str.contains("sperm")] = "sperm whale"

    # Define known categories
    known_categories = [
        "unspecified",
        "orca",
        "shark",
        "grey whale",
        "minke whale",
        "humpback whale",
        "right whale",
        "fin whale",
        "pilot whale",
        "beluga whale",
        "sei whale",
        "blue whale",
        "dolphin",
        "porpoise",
        "sowerbys beaked whale",
        "bairds beaked whale",
        "sealion",
        "sperm whale",
    ]

    # Convert to categorical with explicit category list
    result = pd.Categorical(result, categories=known_categories)

    # Add Data Source
    result["DATA_SOURCE"] = "ACARTIA"
    return pd.Series(result, index=ts.index)


# Preprocess Data
def prepare_acartia_data_for_export(
    data: pd.DataFrame,
    translation_dict: dict[str, str] | None = None,
) -> Optional[pd.DataFrame]:
    """
    Prepares Acartia whale sighting data for export by cleaning,
    standardizing, and enriching the input DataFrame.

    Args:
        data (pd.DataFrame): Raw Acartia data, typically parsed from API or file input.
        translation_dict (dict[str, str] | None): Optional dictionary mapping non-English
            or variant type descriptions to English equivalents. This is used during
            type standardization to improve classification.

    Returns:
        Optional[pd.DataFrame]: Cleaned and structured DataFrame ready for export,
        or None if the input is empty or invalid.

    Processing Steps:
    -----------------
    1. Validate the presence of required columns: 'type', 'created', 'trusted',
       'latitude', 'longitude', 'no_sighted'.
    2. Rename and normalize columns (e.g., NO_SIGHTED → NUMBER_SIGHTED).
    3. Parse and standardize timestamp fields.
    4. Standardize marine mammal 'TYPE' descriptions using a vectorized function
       with optional translation support.
    5. Convert numeric fields and clean coordinate data.
    6. Derive temporal features: 'MONTH-YEAR', 'YEAR', 'WEEK'.
    7. Filter out records prior to 2015-01-01 UTC.
    8. Add a placeholder column 'N_OBSERVERS' (default=1) to support downstream
       aggregation or export logic.

    Notes:
        - Logs a warning if the DataFrame is empty or missing required columns.
        - Unrecognized or unmapped type descriptions will be logged and assigned
          as "unspecified".
        - Assumes dates are in ISO or mixed format and will be coerced into UTC.

    Raises:
        ValueError: If required columns are missing from the input DataFrame.
    """
    # Define required columns
    required_columns = {
        "type",
        "created",
        "latitude",
        "longitude",
        "no_sighted",
    }

    # Exit early if empty
    if data.empty:
        logging.warning("Received empty DataFrame for preprocessing.")
        return None

    # Ensure all required columns are present
    missing = required_columns - set(data.columns)
    if missing:
        logging.error(f"Missing required columns: {sorted(missing)}")
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if "trusted" not in data.columns:
        data["trusted"] = 1
        logging.info("Missing 'trusted' column. Defaulting all values to 1.")
    required_columns.add("trusted")

    # Work on a copy with just the required columns
    data = data[list(required_columns)].copy()

    # Rename columns to uppercase, and rename specific ones
    data.columns = data.columns.str.upper()
    data.rename(columns={"NO_SIGHTED": "NUMBER_SIGHTED"}, inplace=True)

    # Drop rows with missing CREATED timestamps
    data = data.dropna(subset=["CREATED"])

    # Parse datetime (safe for mixed/UTC handling)
    data["DATETIME"] = pd.to_datetime(
        data["CREATED"], format="mixed", utc=True, errors="coerce"
    )
    data["DATE"] = data["DATETIME"].dt.floor("D")

    data["TYPE"] = data["TYPE"].astype("category")

    # Vectorized standardization
    data["TYPE"] = standardize_acartia_type_description_vectorized(
        data["TYPE"], translation_dict
    )

    # Clean and convert number fields
    for col in ["NUMBER_SIGHTED", "TRUSTED"]:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0).astype(int)

    # Convert coordinates to float
    data["LATITUDE"] = pd.to_numeric(data["LATITUDE"], errors="coerce")
    data["LONGITUDE"] = pd.to_numeric(data["LONGITUDE"], errors="coerce")

    # Add time features
    data["MONTH-YEAR"] = (
        data["DATE"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp().dt.date
    )
    data["YEAR"] = data["DATE"].dt.year
    data["WEEK"] = data["DATE"].dt.isocalendar().week

    # Filter by date
    data = data[data["DATE"] >= pd.Timestamp("2015-01-01", tz="UTC")]

    # Add Number of Unique Observers
    group_keys = [
        "DATETIME",
        "DATE",
        "LONGITUDE",
        "LATITUDE",
        "TYPE",
        "TRUSTED",
        "NUMBER_SIGHTED",
        "MONTH-YEAR",
        "YEAR",
        "WEEK",
    ]

    # Merge the counts back into the main DataFrame
    data["N_OBSERVERS"] = 1

    # Final column selection
    data = data[
        [
            "DATETIME",
            "DATE",
            "LONGITUDE",
            "LATITUDE",
            "TYPE",
            "TRUSTED",
            "NUMBER_SIGHTED",
            "MONTH-YEAR",
            "YEAR",
            "WEEK",
            "N_OBSERVERS",
        ]
    ]

    logging.info(f"Preprocessed data with {len(data)} records ready for export.")

    return data


# Export Acartia Data to Disk
def export_acartia_data(
    data: pd.DataFrame,
    output_dir: Path | None = None,
    project_root: Path | None = None,
) -> None:
    """
    Export Acartia whale sightings data to a partitioned Parquet dataset on disk.

    This function appends new data to an existing dataset if found, or creates a new one.
    The dataset is partitioned by 'YEAR' and 'WEEK' columns for efficient querying.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the Acartia sightings data to export.
        Must include 'YEAR' and 'WEEK' columns for partitioning.

    output_dir : Path or None, optional
        Custom directory path where the dataset should be saved.
        If None, `project_root` is used to derive the default path:
        `{project_root}/data/SIGHTINGS/ACARTIA`.

    project_root : Path or None, optional
        Root directory of the project.
        Used to construct default output directory if `output_dir` is not provided.
        If None and `output_dir` is None, defaults to current working directory.

    Raises
    ------
    ValueError
        If required columns ('YEAR', 'WEEK') are missing from the input DataFrame.

    Notes
    -----
    - This function requires the 'pyarrow' package for Parquet operations.
    - If an existing dataset is found, the new data is concatenated with existing data.
    - The combined dataset is saved partitioned by 'YEAR' and 'WEEK'.
    """
    if output_dir is None:
        print("output_idr", output_dir)
        if project_root is None:
            print("project root", project_root)
            # Step back two directories from the current script file location
            project_root = Path(__file__).resolve().parents[2]
        data_dir = project_root / "data" / "SIGHTINGS" / "ACARTIA"
    else:
        data_dir = Path(output_dir).expanduser().resolve()

    data_dir.mkdir(parents=True, exist_ok=True)

    required_cols = {"YEAR", "WEEK"}
    missing = required_cols - set(data.columns)
    if missing:
        logging.error(f"Missing required columns for export: {missing}")
        raise ValueError(f"Missing required columns: {missing}")

    try:
        origin_dataset = ds.dataset(data_dir, format="parquet")
        origin_data = origin_dataset.to_table().to_pandas()
        logging.info(f"Found existing dataset with {len(origin_data)} records.")
        data = pd.concat([origin_data, data], ignore_index=True)
    except (FileNotFoundError, OSError):
        logging.info("No existing dataset found, creating new one.")

    data.to_parquet(
        data_dir,
        engine="pyarrow",
        partition_cols=["YEAR", "WEEK"],
        use_dictionary=False,
    )

    logging.info(f"Exported data to {data_dir} partitioned by YEAR and WEEK.")


######################################
######## MAIN - ORCESTRATION #########


class BaseWhaleSightingHandler(ABC):
    """
    Abstract base class for whale sighting data source handlers.

    This class defines the interface and workflow for fetching, processing,
    and exporting whale sighting data from various sources.

    Attributes:
        output_dir (Optional[Path]): Directory to save exported data. If None,
            a default location should be used by the subclass.
        dry_run (bool): If True, skip the export step for testing purposes.
    """

    def __init__(self, output_dir: Optional[Path] = None, dry_run: bool = False):
        """
        Initialize the handler with optional output directory and dry run flag.

        Args:
            output_dir (Optional[Path]): Directory to export data. Defaults to None.
            dry_run (bool): Whether to skip data export. Defaults to False.
        """
        self.output_dir = output_dir
        self.dry_run = dry_run

    def run_pipeline(self) -> None:
        """
        Run the full data pipeline: fetch, process, and export.

        This method orchestrates the entire flow by calling fetch_data(),
        process_data(), and export_data(), respecting the dry_run flag.
        """
        raw_data = self.fetch_data()
        processed_data = self.process_data(raw_data)
        if not self.dry_run:
            self.export_data(processed_data)
        else:
            print("Dry run enabled - skipping export.")

    @abstractmethod
    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch raw data from the data source.

        Returns:
            pd.DataFrame: Raw data retrieved from the source.

        Raises:
            Implementation specific exceptions if fetching fails.
        """
        pass

    @abstractmethod
    def process_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw data into a clean, export-ready DataFrame.

        Args:
            raw_data (pd.DataFrame): Raw data to process.

        Returns:
            pd.DataFrame: Processed data ready for export.
        """
        pass

    @abstractmethod
    def export_data(self, processed_data: pd.DataFrame) -> None:
        """
        Export the processed data to disk or other storage.

        Args:
            processed_data (pd.DataFrame): Data to export.

        Raises:
            Implementation specific exceptions if export fails.
        """
        pass


class AcartiaHandler(BaseWhaleSightingHandler):
    """
    Handler for whale sighting data from the Acartia data source.

    Implements fetching from Acartia API, processing, and exporting data.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        dry_run: bool = False,
        translation_dict: Optional[dict] = None,
    ):
        """
        Initialize the Acartia handler.

        Args:
            output_dir (Optional[Path]): Directory to export data. Defaults to None.
            dry_run (bool): Whether to skip data export. Defaults to False.
            translation_dict (Optional[dict]): Optional whale type translation mapping.
        """
        super().__init__(output_dir, dry_run)
        self.translation_dict = translation_dict or {}

    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch raw whale sighting data from the Acartia API.

        Loads required environment variables for API URL and token,
        queries the API, and converts the response to a DataFrame.

        Returns:
            pd.DataFrame: Raw Acartia whale sighting data.

        Raises:
            ValueError: If environment variables are missing or API fails.
        """
        api_url = get_env_var("ACARTIA_API_URL")
        token = get_env_var("ACARTIA_API_TOKEN")
        raw_response = query_acartia_api(api_url, token)
        df = collect_data_from_acartia_response(raw_response)
        return df

    def process_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and transform raw Acartia whale sighting data for export.

        Applies translation-aware standardization of whale type labels and
        prepares a structured dataset for downstream use.

        Args:
            raw_data (pd.DataFrame): Raw whale sighting records from Acartia sources.

        Returns:
            pd.DataFrame: Cleaned, filtered, and transformed data with:
                - Standardized whale type labels using translation mapping
                - Normalized datetime and coordinate fields
                - Derived time columns: YEAR, WEEK, MONTH-YEAR
                - Observer count estimate (set to 1 per record)

        Raises:
            ValueError: If required fields are missing from the input.
        """
        return prepare_acartia_data_for_export(raw_data, self.translation_dict)

    def export_data(self, processed_data: pd.DataFrame) -> None:
        """
        Export processed Acartia data to disk.

        Saves the data to the specified output directory or defaults to
        'data/export' if none provided.

        Args:
            processed_data (pd.DataFrame): Data to export.

        Raises:
            IOError: If writing the file fails.
        """
        export_acartia_data(processed_data, self.output_dir)


def main_sightings(
    data_source: str,
    output_dir: Optional[str] = None,
    dry_run: bool = False,
    input_file: Optional[str] = None,
) -> None:
    """
    Orchestrate collection, processing, and export of whale sighting data.

    Args:
        data_source (str): The data source identifier.
        output_dir (Optional[str]): Directory to export data.
        dry_run (bool): If True, run pipeline without export.
        input_file (Optional[str]): Local file to load data from.

    Raises:
        ValueError: For unsupported data sources.
    """
    logging.info("Starting the Whale Sightings pipeline...")
    load_dotenv()

    data_source_upper = data_source.upper()
    output_path = Path(output_dir) if output_dir else None

    if data_source_upper not in DATA_SOURCE_HANDLERS:
        raise ValueError(
            f"Unsupported data source: '{data_source}'. Supported sources: {list(DATA_SOURCE_HANDLERS.keys())}"
        )

    handler_class = DATA_SOURCE_HANDLERS[data_source_upper]
    handler = handler_class(output_dir=output_path, dry_run=dry_run)

    if input_file:
        logging.info(f"Loading local data from: {input_file}")
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file does not exist: {input_path}")

        if input_path.suffix == ".csv":
            data = pd.read_csv(input_path)
        elif input_path.suffix == ".parquet":
            data = pd.read_parquet(input_path)
        else:
            raise ValueError("Input file must be a .csv or .parquet")

        # Pass the loaded data directly to handler's process and export methods
        processed_data = handler.process_data(data)

        if dry_run:
            logging.info("Dry run enabled - skipping data export.")
        else:
            handler.export_data(processed_data)
        logging.info("Data pipeline completed successfully.")
    else:
        # Let handler handle fetching, processing, and exporting
        handler.run_pipeline()

    logging.info("Data pipeline finished successfully.")
    logging.info("=" * 80)


######################################
######################################

#                                                   #
# ------------------------------------------------- #

# ------------------------------------------------- #
#                     MAIN CALL                     #

translation_dict = {
    "ballena azul": "blue whale",
    "ballena jorobada": "humpback whale",
    "ballena franca": "right whale",
    "marsouin": "porpoise",
    "delfín": "dolphin",
    "orquin": "orca",
    "tiburón": "shark",
    "baleine bleue": "blue whale",
    "baleine à bosse": "humpback whale",
    "baleine franche": "right whale",
    "baleine pilote": "pilot whale",
    "baleine grise": "grey whale",
    "requin": "shark",
    "dauphin": "dolphin",
    "phoques": "sealion",
    "baleine de sei": "sei whale",
    "meeresschwein": "porpoise",
    "buckelwal": "humpback whale",
    "grauwal": "grey whale",
    "strandrobbe": "sealion",
    "hai": "shark",
    "baleia azul": "blue whale",
    "baleia jubarte": "humpback whale",
    "baleia franca": "right whale",
    "golfinho": "dolphin",
    "orca": "orca",
    "tubarão": "shark",
    "baleia de sei": "sei whale",
    "autre": "unspecified",
    "non spécifié": "unspecified",
    "indéfini": "unspecified",
    "desconhecido": "unspecified",
}

# Registry of handlers
DATA_SOURCE_HANDLERS = {
    "ACARTIA": AcartiaHandler,
    # Add more data sources here as needed:
    # "OTHER_SOURCE": OtherSourceHandler,
}

# Run Main
if __name__ == "__main__":
    # Arguments Parser
    args = create_arg_parser()

    # Setup Logging
    setup_logging(debug=args.verbose)

    # Main Function
    main_sightings(
        data_source=args.source,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        input_file=args.input_file,
    )

    sys.exit(0)

#                                                   #
# ------------------------------------------------- #
