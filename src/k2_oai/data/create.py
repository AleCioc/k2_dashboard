"""
Functions to manipulate the raw data.
"""

import geopandas
import pandas as pd

from k2_oai import dropbox as dbx
from k2_oai.data.load import dbx_load_dataframe, dbx_load_metadata

__all__ = [
    "dbx_create_geo_metadata",
]


def dbx_create_geo_metadata(dropbox_app):
    metadata = (
        dbx_load_metadata(dropbox_app=dropbox_app)
        .dropna(subset=["center_lng", "center_lat"])
        .rename(columns={"center_lng": "lon", "center_lat": "lat"})
    )

    return geopandas.GeoDataFrame(
        metadata,
        geometry=geopandas.points_from_xy(metadata.lon, metadata.lat),
        crs=4326,
    )


def dbx_concat_label_annotations(dropbox_app):
    files = dbx.dropbox_listdir("/k2/metadata/label_annotations", dropbox_app).item_name
    checkpoints = [
        dbx_load_dataframe(file, "/k2/metadata/label_annotations", dropbox_app)
        for file in files
        if "-checkpoint-" in file
    ]

    return (
        pd.concat(checkpoints)
        .sort_values(["roof_id", "annotation_time"])
        .drop_duplicates(subset="roof_id", keep="last")
        .rename(columns={"annotation": "is_trainable"})
        .assign(
            photos_folder=lambda df: df.photos_folder.str.replace("-api_upload", "")
        )
    )
