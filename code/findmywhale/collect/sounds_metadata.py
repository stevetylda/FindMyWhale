"""
sounds_metadata.py

Sound Metadata Pipeline

A pipeline for discovering, processing, and exporting metadata about sound datasets,
starting with ORCASOUND HLS folder structures stored on AWS S3.

Features:
- Connects to an S3 bucket and scans for "HLS" folders.
- Extracts location and timestamp metadata from folder prefixes.
- Exports the metadata as Parquet files for downstream analysis.
- Easily extendable to support additional sound data sources in the future.

Use Cases:
- Automating metadata cataloging of Orcasound HLS data.
- Building lookup tables for sound dataset ingestion pipelines.
- Serving as a template for other sound dataset metadata workflows.

Currently Supports:
- ORCASOUND: Scans an S3 bucket defined by the environment variable ORCASOUND_SOUNDS_BUCKET.

Example CLI Usage:
    python sound_metadata_pipeline.py --data_source ORCASOUND --output_dir ./output --dry_run

Requirements:
- Python 3.10+
- pandas, numpy, quilt3, python-dotenv

Environment Variables:
- ORCASOUND_SOUNDS_BUCKET: S3 bucket name for ORCASOUND data.

Author: Tyler Stevenson
Date: 2025-07-16
Version: 1.0
"""

# ------------------------------------------------- #
#                      MODULES                      #

# Standard Library
import os
import logging
import argparse
from pathlib import Path
from typing import Optional, Union

# Third-Party Libraries
import pandas as pd
import numpy as np
import quilt3 as q3
from dotenv import load_dotenv

#                                                   #
# ------------------------------------------------- #


# ------------------------------------------------- #
#                     FUNCTIONS                     #

######################################
############### GENERAL ##############


# Setup Logging
def setup_logging(debug: bool = False) -> None:
    """
    Configures logging for the sound metadata pipeline.

    Args:
        debug (bool): If True, set logging level to DEBUG. Defaults to INFO.
    """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    current_dir = Path(__file__).resolve().parent
    log_folder = current_dir.parent.parent / "logs"
    log_folder.mkdir(parents=True, exist_ok=True)

    log_file_path = log_folder / "sounds_pipeline.log"

    log_level = logging.DEBUG if debug else logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s [%(module)s]: %(message)s",
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(),
        ],
    )

    logging.getLogger("botocore.credentials").setLevel(logging.WARNING)


# Get Environment Variable
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
def create_arg_parser() -> argparse.Namespace:
    """
    Parses command-line arguments for the sound metadata pipeline CLI.

    Command-line arguments:
    --data_source : str, optional
        The sound data source to process. Default is 'ORCASOUND'.
        Supported values: ['ORCASOUND'].

    --output_dir : str, optional
        Directory path to save exported metadata. Defaults to project-relative data folder.

    --verbose : bool, optional
        Enables debug-level logging output.

    --dry_run : bool, optional
        Runs pipeline without exporting data.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Sound Metadata Pipeline CLI\n\n"
            "Processes sound dataset metadata.\n"
            "Currently supports 'ORCASOUND' data source.\n\n"
            "Example:\n"
            "  python sound_metadata_pipeline.py --data_source ORCASOUND --output_dir ./output --verbose\n"
            "  python sound_metadata_pipeline.py --data_source ORCASOUND --dry_run\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-d",
        "--data_source",
        type=str,
        choices=["ORCASOUND"],
        default="ORCASOUND",
        help="Sound data source to process. Default: ORCASOUND.",
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Output directory for exported metadata. "
            "Defaults to 'data/SOUNDS/ORCASOUND/' under project root if not provided."
        ),
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging output.",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run the pipeline without saving data. Useful for testing.",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="Sound Metadata Pipeline 1.0",
        help="Show tool version and exit.",
    )

    return parser.parse_args()

    ######################################
    ############### ORCASOUND ############

    # def check_for_hls_folder(bucket: q3.Bucket, head_folder: str) -> Optional[str]:
    """
    Identifies if an 'HLS' folder exists in the given head folder.

    Args:
        bucket (q3.Bucket): AWS S3 Bucket object with `.ls()` method.
        head_folder (str): Head folder path (prefix).

    Returns:
        Optional[str]: Prefix of the HLS folder if found, else None.
    """
    if not head_folder.endswith("/"):
        head_folder += "/"

    try:
        contents = bucket.ls(head_folder)
        folders = (
            contents[0] if isinstance(contents, tuple) and len(contents) > 0 else []
        )

        for entry in folders:
            prefix = entry.get("Prefix", "").lower()
            if "/hls/" in prefix or prefix.endswith("hls/"):
                return entry["Prefix"]
    except Exception as e:
        logging.warning(f"Error checking for HLS folder in '{head_folder}': {e}")

    return None


######################################
######################################

######################################
####### DATA SOURCE: ORCASOUND #######


# ORCASOUND - Check for HLS Folder
def check_for_hls_folder(bucket: q3.Bucket, head_folder: str) -> Optional[str]:
    """
    Identifies if an 'HLS' folder exists in the given head folder.

    Args:
        bucket (q3.Bucket): AWS S3 Bucket object with `.ls()` method.
        head_folder (str): Head folder path (prefix).

    Returns:
        Optional[str]: Prefix of the HLS folder if found, else None.
    """
    if not head_folder.endswith("/"):
        head_folder += "/"

    try:
        contents = bucket.ls(head_folder)
        folders = (
            contents[0] if isinstance(contents, tuple) and len(contents) > 0 else []
        )

        for entry in folders:
            prefix = entry.get("Prefix", "").lower()
            if "/hls/" in prefix or prefix.endswith("hls/"):
                return entry["Prefix"]
    except Exception as e:
        logging.warning(f"Error checking for HLS folder in '{head_folder}': {e}")

    return None


# ORCASOUND - Identify HLS Folder
def get_hls_folders(bucket_name: str) -> pd.DataFrame:
    """
    Scans the S3 bucket for HLS folders and builds a DataFrame
    with location and datetime metadata.

    Args:
        bucket_name (str): Name of the S3 bucket.

    Returns:
        pd.DataFrame: DataFrame with columns 'PREFIX', 'LOCATION', 'UNIX_DATETIME', and 'DATETIME'.
    """
    bucket = q3.Bucket(f"s3://{bucket_name}")
    try:
        head_entries = bucket.ls("")[0]  # List top-level prefixes
        head_folders = [entry["Prefix"] for entry in head_entries if "Prefix" in entry]
    except Exception as e:
        raise RuntimeError(
            f"Failed to list top-level folders in bucket '{bucket_name}': {e}"
        )

    items = []
    for head_folder in head_folders:
        hls_folder = check_for_hls_folder(bucket, head_folder)
        if hls_folder:
            try:
                hls_contents = bucket.ls(hls_folder)
                hls_items = hls_contents[0] if hls_contents else []
                items.extend(hls_items)
            except Exception as e:
                logging.warning(f"Failed to list HLS folder '{hls_folder}': {e}")

    if not items:
        return pd.DataFrame(columns=["PREFIX", "LOCATION", "UNIX_DATETIME", "DATETIME"])

    hls_folder_df = pd.DataFrame(items)

    # Extract LOCATION and UNIX_DATETIME using regex for robustness
    pattern = r"^(.*)/hls/(\d+)/?$"
    extracted = hls_folder_df["Prefix"].str.extract(pattern)
    hls_folder_df["LOCATION"] = extracted[0].str.strip("/")
    hls_folder_df["UNIX_DATETIME"] = extracted[1]

    hls_folder_df["PREFIX"] = hls_folder_df["Prefix"].apply(
        lambda x: f"s3://{bucket_name}/{x}"
    )

    # Convert UNIX timestamps safely
    hls_folder_df["DATETIME"] = pd.to_datetime(
        pd.to_numeric(hls_folder_df["UNIX_DATETIME"], errors="coerce"),
        unit="s",
        errors="coerce",
        utc=True,
    )

    return hls_folder_df[["PREFIX", "LOCATION", "UNIX_DATETIME", "DATETIME"]]


# ORCASOUND - Export Orcasound Metadata
def export_orcasound_metadata(
    data: pd.DataFrame,
    output_dir: Optional[Union[Path, str]] = None,
) -> None:
    """
    Exports ORCASOUND HLS folder data to disk as a Parquet file.

    If no output directory is provided, the data is saved under the
    project-root-relative path `data/SOUNDS/ORCASOUND/`.

    Args:
        data (pd.DataFrame): DataFrame containing HLS folder lookup data.
            Expected columns: 'PREFIX', 'LOCATION', 'UNIX_DATETIME', 'DATETIME'.
        output_dir (Optional[Union[Path, str]]): Directory path to save the dataset.
            If None, defaults to 'data/SOUNDS/ORCASOUND/' under project root.

    Raises:
        ValueError: If input DataFrame is empty.
        Exception: If export fails.
    """
    if output_dir is None:
        project_root = Path(__file__).resolve().parents[2]
        data_dir = project_root / "data" / "SOUNDS" / "ORCASOUND"
    else:
        data_dir = Path(output_dir).expanduser().resolve()

    data_dir.mkdir(parents=True, exist_ok=True)

    if data.empty:
        logging.error("Input DataFrame is empty. Aborting export.")
        raise ValueError("Input DataFrame is empty.")

    output_file = data_dir / "orcasound_hls_folders.parquet"

    try:
        data.to_parquet(output_file, index=False)
        logging.info(f"Successfully exported ORCASOUND HLS data to {output_file}")
    except Exception as e:
        logging.error(f"Failed to export ORCASOUND HLS data: {e}")
        raise


######################################
######################################

######################################
######## MAIN - ORCESTRATION #########


# Main Sounds Function
def main_sounds(
    data_source: str,
    output_dir: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """
    Orchestrates the metadata pipeline for sound datasets.

    Args:
        data_source (str): Sound data source identifier. Currently supports 'ORCASOUND'.
        output_dir (Optional[str]): Output directory path. Defaults to project-relative folder if None.
        dry_run (bool): If True, skips writing to disk.

    Raises:
        ValueError: For unsupported data sources.
        EnvironmentError: If required environment variables are missing.
    """
    logging.info("Starting the Whale Sounds Metadata pipeline...")
    load_dotenv()

    try:
        if data_source.upper() == "ORCASOUND":
            logging.info("Running ORCASOUND HLS metadata pipeline.")

            bucket_name = get_env_var("ORCASOUND_SOUNDS_BUCKET")

            hls_df = get_hls_folders(bucket_name)

            if hls_df.empty:
                logging.warning("No HLS folders found in the specified bucket.")
                return

            logging.info(f"Discovered {len(hls_df)} HLS folders.")

            if dry_run:
                logging.info("Dry run enabled — skipping export.")
            else:
                export_orcasound_metadata(
                    data=hls_df, output_dir=Path(output_dir) if output_dir else None
                )
                logging.info("Exported ORCASOUND HLS folder metadata successfully.")
        else:
            logging.error(f"Unsupported data source: '{data_source}'")
            raise ValueError(
                f"Unsupported data source: '{data_source}'. Supported sources: ['ORCASOUND']"
            )

    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        raise

    logging.info("Sound metadata pipeline completed.")


# ------------------------------------------------- #
#                       MAIN                        #

if __name__ == "__main__":
    args = create_arg_parser()
    setup_logging(debug=args.verbose)

    try:
        main_sounds(
            data_source=args.data_source,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )
    except Exception as e:
        logging.error(f"Pipeline terminated with error: {e}")
        raise


#                                                   #
# ------------------------------------------------- #
