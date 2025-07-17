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
Date: 2025-07-16
"""

# ------------------------------------------------- #
#                      MODULES                      #

# Standard Library
import os
import logging
import argparse
from pathlib import Path
from typing import Optional, List, Dict

# Third-Party Libraries
import pandas as pd
import numpy as np
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

    current_dir = Path(__file__).resolve().parent
    log_folder = current_dir.parent / "logs"
    log_folder.mkdir(parents=True, exist_ok=True)

    log_file_path = log_folder / "whale_sightings.log"

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

    session = requests.Session()

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
        logging.error(f"HTTP error occurred: {e} (Status code: {response.status_code})")
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")
    except ValueError:
        logging.error("Failed to parse JSON response.")

    return None


# Collect Data from Acartia Response
def collect_data_from_acartia_response(data: list) -> Optional[pd.DataFrame]:
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
) -> pd.Series:
    """
    Vectorized standardization of whale/marine mammal type descriptions.

    Args:
        type_series (pd.Series): Series of raw type description strings.

    Returns:
        pd.Series: Series of standardized type labels.
    """
    # Convert to lowercase string, fill NAs with empty string
    ts = type_series.fillna("").str.lower()

    # Remove unwanted substrings using vectorized replace with regex
    # Similar to your substitutions but applied to entire Series
    ts = ts.str.replace(r"sighting|:|\\|\(specify in comments\)|'", "", regex=True)
    ts = ts.str.replace("autre", "unspecified")
    ts = ts.str.replace("commun", "common")
    ts = ts.str.replace("marsouin", "porpoise")
    ts = ts.str.replace("common", "")
    ts = ts.str.replace(r"\s{2,}", " ", regex=True).str.strip()

    # Mark unknowns/unidentified as "unspecified"
    unspecified_mask = ts.str.contains(
        r"other|unknown|unidentified|non spécifié|^$",
        regex=True,
        na=False,
    )
    ts = ts.where(~unspecified_mask, "unspecified")

    # Mapping keywords to labels - vectorized with np.select
    # Conditions for each label
    conditions = [
        ts.str.contains("orca|killer"),
        ts.str.contains("shark"),
        ts.str.contains("gray|grey"),
        ts.str.contains("minke"),
        ts.str.contains("humpback|jorobada"),
        ts.str.contains("right"),
        ts.str.contains("fin"),
        ts.str.contains("pilot"),
        ts.str.contains("beluga"),
        ts.str.contains("sei"),
        ts.str.contains("blue"),
        ts.str.contains("dolphin"),
        ts.str.contains("porpoise|rorqual"),
        ts.str.contains("sowerbys"),
        ts.str.contains("bairds"),
        ts.str.contains("sealion"),
    ]

    choices = [
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
    ]

    ts = pd.Series(np.select(conditions, choices, default=ts), index=ts.index)

    # Finally, replace any remaining empty strings with "unspecified"
    ts = ts.replace("", "unspecified")

    return ts.astype("category")  # Optional: convert to category for memory efficiency


# Preprocess Data
def prepare_acartia_data_for_export(data: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Prepares Acartia data for export.

    Args:
        data (pd.DataFrame): Raw Acartia data.

    Returns:
        pd.DataFrame: Cleaned and preprocessed data, ready for export.

    Steps:
    1. Validate required columns.
    2. Clean and rename columns.
    3. Standardize and parse dates.
    4. Normalize values and datatypes.
    5. Add derived time features (Month, Year, Week).
    """
    # Define required columns
    required_columns = {
        "type",
        "created",
        "trusted",
        "latitude",
        "longitude",
        "no_sighted",
    }

    required_columns = {
        "type",
        "created",
        "trusted",
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
    data["TYPE"] = standardize_acartia_type_description_vectorized(data["TYPE"])

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
        ]
    ]

    logging.info(f"Preprocessed data with {len(data)} records ready for export.")

    return data


# Export Acartia Data to Disk
def export_acartia_data(data: pd.DataFrame, output_dir: Path | None = None) -> None:
    """
    Appends new Acartia data to an existing partitioned Parquet dataset on disk.

    The dataset is saved under the project-root-relative path `data/ACARTIA/`,
    partitioned by the `YEAR` and `WEEK` columns.

    Steps:
    1. Resolves the project root path.
    2. Ensures the data directory exists.
    3. Loads existing Parquet data if present.
    4. Appends new data to the existing dataset.
    5. Writes the combined dataset back to Parquet using partitioning.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing the new Acartia data to append.
        Must include 'YEAR' and 'WEEK' columns for partitioning.
    output_dir : Path or None, optional
        Directory path to save the dataset. If None, defaults to project-root-relative 'data/ACARTIA/'.


    Returns
    -------
    None
        This function writes data to disk and does not return a value.

    Raises
    ------
    ValueError
        If required columns ('YEAR', 'WEEK') are missing from the input DataFrame.
    """
    # Use provided output_dir or default to project data folder
    if output_dir is None:
        project_root = Path(__file__).resolve().parents[1]  # adjust if needed
        data_dir = project_root / "data" / "ACARTIA"
    else:
        data_dir = Path(output_dir).expanduser().resolve()

    data_dir.mkdir(parents=True, exist_ok=True)

    required_cols = {"YEAR", "WEEK"}
    missing = required_cols - set(data.columns)
    if missing:
        logging.error(f"Missing required columns for export: {missing}")
        raise ValueError(f"Missing required columns: {missing}")

    # Check for existing parquet data
    if any(data_dir.glob("*.parquet")) or any(data_dir.glob("YEAR=*/")):
        origin_dataset = ds.dataset(data_dir, format="parquet")
        origin_data = origin_dataset.to_table().to_pandas()
        data = pd.concat([origin_data, data], ignore_index=True)
        logging.info(
            f"Appended data to existing dataset, total records now: {len(data)}"
        )
    else:
        logging.info("No existing dataset found. Creating new dataset.")

    data.to_parquet(
        data_dir,
        engine="pyarrow",
        partition_cols=["YEAR", "WEEK"],
        use_dictionary=False,
    )

    logging.info(f"Exported data to {data_dir} partitioned by YEAR and WEEK.")


# Main Orchestration
def main(
    data_source: str, output_dir: str | None = None, dry_run: bool = False
) -> None:
    """
    Orchestrate collection, processing, and export of whale sighting data.

    Args:
        data_source (str): The data source identifier. Currently supports 'ACARTIA' only.
        output_dir : str or None, optional
            Custom directory to export data. Defaults to project data folder if None.
        dry_run : bool, optional
            If True, run the pipeline without exporting data.

    Raises:
        ValueError: If an unsupported data source is provided.
        EnvironmentError: If required environment variables are missing.

    Note:
        This function is designed to be easily extended to support
        additional whale sighting data sources in the future.
    """
    logging.info("Starting the Whale Sightings pipeline...")
    # Load environment variables from .env file
    load_dotenv()

    if data_source.upper() == "ACARTIA":
        logging.info("Starting Acartia whale sightings data pipeline.")

        # Define API URL (last 7 days of sightings)
        acartia_api_url = get_env_var("ACARTIA_API_URL")

        # Retrieve API token from environment
        token = get_env_var("ACARTIA_API_TOKEN")

        # Step 1: Query Acartia API
        data = query_acartia_api(api_url=acartia_api_url, token=token)

        # Step 2: Convert response to DataFrame
        data = collect_data_from_acartia_response(data)

        # Step 3: Preprocess data for export
        data = prepare_acartia_data_for_export(data)

        # Step 4: Export data if available
        if data is not None:
            if dry_run:
                logging.info("Dry run enabled - skipping data export.")
            else:
                export_acartia_data(
                    data=data, output_dir=Path(output_dir) if output_dir else None
                )
                logging.info("Data pipeline completed successfully.")
        else:
            logging.warning("No data to export.")
    else:
        logging.error(f"Unsupported data source: '{data_source}'")
        raise ValueError(
            f"Unsupported data source: '{data_source}'. Supported sources: 'ACARTIA'."
        )

    logging.info("Data pipeline finished successfully.")
    logging.info("\n" + "=" * 80 + "\n")


######################################
######################################

#                                                   #
# ------------------------------------------------- #

# ------------------------------------------------- #
#                     MAIN CALL                     #

# Run Main
if __name__ == "__main__":
    # Arguments Parser
    args = create_arg_parser()

    # Setup Logging
    setup_logging(debug=args.verbose)

    # Main Function
    main(data_source=args.source, output_dir=args.output_dir, dry_run=args.dry_run)

#                                                   #
# ------------------------------------------------- #
