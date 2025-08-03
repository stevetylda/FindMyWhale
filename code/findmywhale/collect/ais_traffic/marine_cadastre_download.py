# ------------------------------------------------- #
#                      MODULES                      #

# Standard Library
import requests
import zipfile
import io
import os

#                                                   #
# ------------------------------------------------- #


# ------------------------------------------------- #
#                     FUNCTIONS                     #

######################################
############### GENERAL ##############


# Marine Cadastre AIS Download
def download_marinecadastre_ais(
    year: int,
    month: int,
    dest_dir="data/ais",
    base_url="https://coast.noaa.gov/htdata/CMSP/AISDataHandler",
):
    base_url = f"{base_url}/{year}/"
    month_str = f"{month:02d}"
    filename = f"AIS_{year}_{month_str}_US.csv.zip"
    file_url = base_url + filename

    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, filename)

    print(f"Downloading {filename}...")
    r = requests.get(file_url)
    if r.status_code != 200:
        raise ValueError(f"Failed to download file: {file_url}")

    with open(zip_path, "wb") as f:
        f.write(r.content)

    print(f"Extracting {filename}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dest_dir)

    print("Done!")
    return os.path.join(dest_dir, filename.replace(".zip", ""))


df_path = download_marinecadastre_ais(2023, 1)

import pandas as pd
df = pd.read_csv(df_path, low_memory=False)
df.head()
