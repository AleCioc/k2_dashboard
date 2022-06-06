"""
Common utilities for dashboard mode pages.
"""

from __future__ import annotations

import os

import streamlit as st

from k2_oai import dropbox as dbx
from k2_oai.data import data_loader
from k2_oai.dropbox import DROPBOX_RAW_PHOTOS_ROOT
from k2_oai.utils import draw_labels, rotate_and_crop_roof  # , experimental_draw_labels

__all__ = [
    "st_dropbox_connect",
    "st_listdir",
    "st_listdir_no_cache",
    "st_load_dataframe",
    "st_load_metadata",
    "st_load_geo_metadata",
    "st_load_annotations",
    "st_load_photo_list",
    "st_load_photo_list_and_metadata",
    "st_load_photo",
    "st_load_photo_from_roof_id",
    "st_load_photo_and_roof",
    "st_save_annotations",
]


@st.cache(allow_output_mutation=True)
def st_dropbox_connect():
    if "DROPBOX_ACCESS_TOKEN" in os.environ:
        return dbx.dropbox_connect_access_token_only(st.session_state["access_token"])
    return dbx.dropbox_connect(
        st.session_state["access_token"], st.session_state["refresh_token"]
    )


@st.cache
def st_listdir(path):
    dbx_app = st_dropbox_connect()
    return dbx.dropbox_listdir(path, dbx_app)


def st_listdir_no_cache(path):
    """No cache version of st_listdir. It's meant to be used in streamlit components
    where you want to reload the list of files. In this case, the list will be sensitive
    to changes in the dropbox folders which will be reflected immediately. Normally,
    this would only have been possible after refreshing the cache (i.e. by re-running
    the whole application).

    Parameters
    ----------
    path
        Path to the folder to list.

    Returns
    -------
    DataFrame
        DataFrame containing the list of files in the folder and some metadata.
    """
    dbx_app = st_dropbox_connect()
    return dbx.dropbox_listdir(path, dbx_app)


@st.cache
def st_load_dataframe(filename, dropbox_path):
    dbx_app = st_dropbox_connect()
    return data_loader.dbx_load_dataframe(filename, dropbox_path, dbx_app)


@st.cache
def st_load_metadata():
    dbx_app = st_dropbox_connect()
    return data_loader.dbx_load_metadata(dbx_app)


@st.cache
def st_load_geo_metadata():
    dbx_app = st_dropbox_connect()
    return data_loader.dbx_load_geo_metadata(dbx_app)


@st.cache(allow_output_mutation=True)
def st_load_annotations(filename):
    dbx_app = st_dropbox_connect()
    return data_loader.dbx_load_label_annotations(filename, dbx_app)


@st.cache(allow_output_mutation=True)
def st_load_photo_list(photos_folder):

    root_folder = DROPBOX_RAW_PHOTOS_ROOT
    photos_folder_contents = st_listdir(root_folder).item_name.values

    index_file = f"index-{photos_folder}.csv"

    if index_file in photos_folder_contents:
        return st_load_dataframe(index_file, root_folder)
    else:
        photos_path = f"{root_folder}/{photos_folder}"
        return st_listdir(path=photos_path)[["item_name"]]


@st.cache
def st_load_photo_list_and_metadata(
    photos_folder=None,
    geo_metadata: bool = False,
):
    if geo_metadata:
        obstacle_metadata = st_load_geo_metadata()
    else:
        obstacle_metadata = st_load_metadata()

    if photos_folder is None:
        return obstacle_metadata, obstacle_metadata.imageURL.unique()

    photos_list = st_load_photo_list(photos_folder)

    available_metadata = obstacle_metadata.loc[
        lambda df: df.imageURL.isin(photos_list.item_name)
    ]

    return available_metadata, photos_list


@st.cache(allow_output_mutation=True)
def st_load_photo(
    photo_name,
    folder_name,
    greyscale_only: bool = False,
):
    dbx_app = st_dropbox_connect()
    dbx_path = f"{DROPBOX_RAW_PHOTOS_ROOT}/{folder_name}"
    return data_loader.dbx_load_photo(
        photo_name, dbx_path, dbx_app, greyscale_only=greyscale_only
    )


@st.cache(allow_output_mutation=True)
def st_load_photo_from_roof_id(
    roof_id,
    metadata,
    chosen_folder,
    bgr_only=False,
    greyscale_only=False,
):
    dbx_app = st_dropbox_connect()
    dbx_path = f"{DROPBOX_RAW_PHOTOS_ROOT}/{chosen_folder}"

    return data_loader.dbx_load_photos_from_roof_id(
        roof_id, metadata, dbx_path, dbx_app, bgr_only, greyscale_only
    )


def get_coordinates_from_roof_id(roof_id, photos_metadata) -> tuple[str, list[str]]:

    roof_px_coordinates = photos_metadata.loc[
        photos_metadata.roof_id == roof_id, "pixelCoordinates_roof"
    ].iloc[0]

    obstacles_px_coordinates = [
        coord
        for coord in photos_metadata.loc[
            photos_metadata.roof_id == roof_id, "pixelCoordinates_obstacle"
        ].values
    ]

    return roof_px_coordinates, obstacles_px_coordinates


def st_load_photo_and_roof(
    roof_id,
    photos_metadata,
    chosen_folder,
    as_greyscale: bool = False,
):
    roof_px_coord, obstacles_px_coord = get_coordinates_from_roof_id(
        roof_id, photos_metadata
    )

    if as_greyscale:
        photo = st_load_photo_from_roof_id(
            roof_id, photos_metadata, chosen_folder, greyscale_only=True
        )
    else:
        photo = st_load_photo_from_roof_id(
            roof_id, photos_metadata, chosen_folder, bgr_only=True
        )

    labelled_photo = draw_labels(photo, roof_px_coord, obstacles_px_coord)
    # labelled_photo = experimental_draw_labels(
    #     photo,
    #     roof_px_coord,
    #     obstacles_px_coord
    # )

    roof = rotate_and_crop_roof(photo, roof_px_coord)
    labelled_roof = rotate_and_crop_roof(labelled_photo, roof_px_coord)

    return photo, roof, labelled_photo, labelled_roof


def st_save_annotations(data_to_upload, filename, destination_folder):

    dbx_app = st_dropbox_connect()

    file_to_upload = f"/tmp/{filename}"
    destination_path = f"{destination_folder}/{filename}"

    data_to_upload.to_csv(file_to_upload, index=False)

    dbx.dropbox_upload_file_to(dbx_app, file_to_upload, destination_path)
