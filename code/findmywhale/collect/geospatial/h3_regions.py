"""
GIS Processing for Whale Sightings + Localization Data
---------------------------------------------------------------


Key Features:
- Utilize Uber's H3 Gridding Systems to Produce Equal-Area Polygons for

Usage Notes:
- Requires an Extent Polygon

Usage (CLI):


Environment Variables Required:


Dependencies:
- pandas

Author: Tyler Stevenson
Date: 2025-07-168
Version: 1.0
"""

# ------------------------------------------------- #
#                      MODULES                      #

# Standard Library
import os
from pathlib import Path
from typing import Optional

# Third-Party Libraries
import geopandas as gpd
import h3

from shapely.geometry import shape
from shapely.geometry import Polygon

#                                                   #
# ------------------------------------------------- #

# ------------------------------------------------- #
#                     FUNCTIONS                     #

######################################
############### GENERAL ##############


# Load Dissolved Polygon Extent
def load_dissolved_polygon_geometry(
    polygon_geometry_path: str, output_crs: str = "EPSG:4326"
) -> Optional[gpd.GeoDataFrame]:
    """Loads Polygon Geometry and Dissolves to Form One Region
    Args:
        polygon_geometry_path (str):Path to Polygon Geometry (Shapefile, Geojson)
        output_crs (str): CRS to translate GeoDataFrame
    Returns:
        polygon_gdf (GeoDataFrame): Dissolved Polygon Geometry GeoDataFrame

    TODO:
        - Check Path type to ensure it is a valide file type that geopandas will allow
        - Check that existing geodataframe has CRS
    """
    # Path to SSEA water region shapefile
    polygon_path = Path(polygon_geometry_path)

    # Load shapefile
    polygon_gdf = gpd.read_file(polygon_path)
    polygon_gdf = polygon_gdf[["geometry"]]

    # Dissolve all Geometries Into One Polygon (multi-region merge)
    polygon_gdf = polygon_gdf.dissolve()

    # Update CRS
    if polygon_gdf.crs != output_crs:
        polygon_gdf = polygon_gdf.to_crs(output_crs)

    return polygon_gdf


######################################
############### Uber H3 ##############


# Build H3 Grids Over Polygon
def build_h3_grids_overpolygon(
    polygon_gdf: gpd.GeoDataFrame,
    h3_search_resolution: int = 4,
    h3_target_resolution: int = 6,
) -> gpd.GeoDataFrame:
    """Builds H3 Grids for a Polygon GeoDataFrame
    Args:
        polygon_gdf (gpd.GeoDataFrame):
        h3_search_resolution (int): Search H3 Resolution
        h3_target_resolution (int): Target H3 Resolution of Output
    Returns:
        h3_polygons (GeoDataFrame): H3 Polygons Over Area of Interest
    """
    # Uniformly sample 500 points within the dissolved polygon
    sampled_points = (
        polygon_gdf.sample_points(size=1000, method="uniform").reset_index().explode()
    )

    # Extract latitude and longitude for each point
    sampled_points["latitude"], sampled_points["longitude"] = zip(
        *sampled_points["sampled_points"].apply(lambda point: (point.y, point.x))
    )

    # Assign each point to a unique H3 cell at resolution 3
    sampled_points[f"H3_GRID_{h3_search_resolution}"] = sampled_points.apply(
        lambda row: h3.latlng_to_cell(
            row.latitude, row.longitude, h3_search_resolution
        ),
        axis=1,
    )

    # Identify All H3 Grids
    sampled_points = sampled_points[
        [f"H3_GRID_{h3_search_resolution}"]
    ].drop_duplicates()

    # Find All Children at Higher Resolution
    sampled_points[f"H3_GRID_{h3_target_resolution}"] = sampled_points[
        f"H3_GRID_{h3_search_resolution}"
    ].apply(lambda cell: h3.cell_to_children(cell, h3_target_resolution))

    # Flatten the list of children into rows
    sampled_points = sampled_points.explode(f"H3_GRID_{h3_target_resolution}")
    sampled_points = sampled_points[
        [f"H3_GRID_{h3_target_resolution}"]
    ].drop_duplicates()

    # Generate polygon geometry from each H3 RES Target cell
    sampled_points["geometry"] = sampled_points[
        f"H3_GRID_{h3_target_resolution}"
    ].apply(lambda cell: Polygon(shape(h3.cells_to_geo([cell]))))
    h3_polygons = gpd.GeoDataFrame(sampled_points, geometry="geometry", crs="EPSG:4326")

    h3_polygons["IN_BOUNDS"] = h3_polygons["geometry"].intersects(
        polygon_gdf.geometry, align=True
    )
    h3_polygons = h3_polygons[h3_polygons.IN_BOUNDS == True]
    h3_polygons = h3_polygons[[f"H3_GRID_{h3_target_resolution}", "geometry"]]

    return h3_polygons


# Save H3 Polygons to File
def export_h3_polygons(
    h3_polygons: gpd.GeoDataFrame,
    output_directory: str,
    file_prefix: str,
    h3_target_resolution: str,
) -> None:
    """Saves H3 Polygons to File
    Args:
        h3_polygons (GeoDataFrame): H3 Polygon GeoDataFrame
        output_directory (str): Output Directory
        file_prefix (str): File Name Prefix (e.g., relate to data source)
        h3_target_resolution (str): H3 Target Resolution
    Returns:
        None

    Saves File to Output H3 Polygon Directory
    """
    # Check + Make Path
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Build Output Path
    output_path = f"{output_directory}/{file_prefix}_{h3_target_resolution}.parquet"

    # Export
    h3_polygons.to_parquet(output_path)


######################################
################ MAIN ################


def main(
    polygon_geometry_path,
    h3_search_resolution,
    h3_target_resolution,
    output_directory,
    file_prefix,
):
    """Orchestrates AOI-to-H3 Polygon Creation
    Args:
        polygon_geometry_path
        h3_search_resolution
        h3_target_resolution
        output_directory
        file_prefix
    Returns:
        None

    Save H3 Polygon Creations to Disk

    """
    # Load Dissolved Polygon
    polygon_gdf = load_dissolved_polygon_geometry(polygon_geometry_path)

    # Build H3 Grids Over Polygon at Target Resolution
    h3_polygons = build_h3_grids_overpolygon(
        polygon_gdf,
        h3_search_resolution=h3_search_resolution,
        h3_target_resolution=h3_target_resolution,
    )

    # Export Polygons
    export_h3_polygons(
        h3_polygons=h3_polygons,
        output_directory=output_directory,
        file_prefix=file_prefix,
        h3_target_resolution=h3_target_resolution,
    )


#                                                   #
# ------------------------------------------------- #

# ------------------------------------------------- #
#                        MAIN                       #

if __name__ == "__main__":
    # Data Source: https://soggy2.zoology.ubc.ca/geonetwork/srv/eng/catalog.search#/metadata/e1da07ca-353d-4b8a-a08b-643885e89e3b

    # Set Path to Polygon
    polygon_geometry_path = "/Users/tylerstevenson/Documents/CODE/FindMyWhale/data/GIS/RAW/ssea_regions/ssea_regions.shp"
    h3_search_resolution = 4
    h3_target_resolution = 5

    output_directory = (
        "/Users/tylerstevenson/Documents/CODE/FindMyWhale/data/GIS/POLYGONS"
    )
    file_prefix = "SSEA_REGION"

    main(
        polygon_geometry_path,
        h3_search_resolution,
        h3_target_resolution,
        output_directory,
        file_prefix,
    )

#                                                   #
# ------------------------------------------------- #
