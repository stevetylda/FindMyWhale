#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
detection.py

Accesses Data from OrcaSound to Use for Demo + Testing

Part of the FindMyWhale project: https://github.com/stevetylda/FindMyWhale

Author: Tyler Stevenson
Created: 2025-07-16
License: MIT
"""

# ------------------------------------------------- #
#                      MODULES                      #

# General Modules
import os
import itertools
import pandas as pd
import numpy as np

# AWS Modules
import quilt3 as q3

#                                                   #
# ------------------------------------------------- #


# ------------------------------------------------- #
#                     FUNCTIONS                     #

#################################
######## HYDROPHONE DATA ########

##############
#### ORCASOUND


# Identify HLS Folder in Bucket:Head Folder
def check_for_hls_folder(bucket, head_folder: str) -> str | None:
    """Identifies if "HLS" Folder Exists in Head Folder
    Args:
        bucket : AWS S3 Bucket
        head_folder (str): Head Folder Name
    Returns:
        return_val (str): Prefix of HLS Folder or None
    """
    # Default Return Value
    return_val = None

    # Iterate through Items in Bucket to Identify HLS Prefixes
    for item_ in bucket.ls(head_folder):
        if len(item_) > 0 and len(item_) < 2:
            if "Prefix" in item_[0].keys():
                if "hls" in item_[0]["Prefix"]:
                    return_val = item_[0]["Prefix"]
        elif len(item_) > 0 and len(item_) < 3:
            if "hls" in item_[1]["Prefix"]:
                return_val = item_[1]["Prefix"]

    return return_val


# Identify HLS Folders + Build Lookup
def get_hls_folders(bucket_name: str) -> pd.DataFrame:
    """Builds HLS Folder Lookup
    Args:
        bucket_name (str): Name of Bucket to Use
    Returns:
        hls_folder_df (DataFrame): HLS Folder Lookup
    """
    # Connect to Bucket
    bucket = q3.Bucket(f"s3://{bucket_name}")

    # Get Parent Folders
    head_folders = bucket.ls("")[0]
    head_folders = list(
        itertools.chain.from_iterable([list(obj.values()) for obj in head_folders])
    )

    # Get Items with HLS Sub-Folder
    items = []
    for head_folder in head_folders:
        hls_folder = check_for_hls_folder(bucket, head_folder)

        if hls_folder != None:
            # Get Folder within HLS Sub-Folder
            hls_subfolders = bucket.ls(hls_folder)

            for item in hls_subfolders[0]:
                items.append(item)

    # Combine
    hls_folder_df = pd.DataFrame(items)

    # Enrich HLS Lookup
    hls_folder_df["LOCATION"] = (
        hls_folder_df["Prefix"].str.split("/hls/").str[0].str.strip("/")
    )
    hls_folder_df["UNIX_DATETIME"] = (
        hls_folder_df["Prefix"].str.split("/hls/").str[-1].str.strip("/")
    )
    hls_folder_df["PREFIX"] = hls_folder_df["Prefix"].apply(
        lambda x: f"s3://{bucket_name}/{x}"
    )
    hls_folder_df["DATETIME"] = pd.to_datetime(
        hls_folder_df["UNIX_DATETIME"].astype(int), unit="s", utc=True
    )

    hls_folder_df = hls_folder_df[["PREFIX", "LOCATION", "UNIX_DATETIME", "DATETIME"]]

    return hls_folder_df


#################################
#################################

##################################
######## WHALE SIGHT DATA ########

##############
#### ARCARTIA


#################################
#################################


#                                                   #
# ------------------------------------------------- #

# ------------------------------------------------- #
#                     MAIN CALL                     #


#                                                   #
# ------------------------------------------------- #
