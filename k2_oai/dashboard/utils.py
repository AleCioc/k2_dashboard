"""
Common utilities for dashboard mode pages.
"""

from __future__ import annotations

import os
from datetime import datetime

import dropbox
import streamlit as st
from dotenv import load_dotenv

from k2_oai.io import data_loader
from k2_oai.io import dropbox as dbx
from k2_oai.io.dropbox_paths import DROPBOX_RAW_PHOTOS_ROOT
from k2_oai.obstacle_detection import (
    binarization_step,
    detect_obstacles,
    detect_obstacles_composite,
    edge_detection,
    hough_transform,
    filtering_step,
    morphological_opening_step,
)
from k2_oai.utils import draw_boundaries, rotate_and_crop_roof
from k2_oai.utils._image_manipulation import draw_obstacles

load_dotenv()

_DROPBOX_NAMESPACE_ID = os.environ.get("DROPBOX_NAMESPACE_ID")
_DROPBOX_USER_EMAIL = os.environ.get("DROPBOX_USER_MAIL")
_DROPBOX_APP_KEY = os.environ.get("APP_KEY")
_DROPBOX_APP_SECRET = os.environ.get("APP_SECRET")
if "DROPBOX_ACCESS_TOKEN" in os.environ:
    _DROPBOX_ACCESS_TOKEN = os.environ.get("DROPBOX_ACCESS_TOKEN")

__all__ = [
    "st_dropbox_oauth2_connect",
    "st_dropbox_connect",
    "st_list_contents_of",
    "st_load_dataframe",
    "st_load_metadata",
    "st_load_geo_metadata",
    "st_load_annotations",
]


def st_dropbox_oauth2_connect(
    dropbox_app_key=_DROPBOX_APP_KEY, dropbox_app_secret=_DROPBOX_APP_SECRET
):
    dropbox_oauth_flow = dropbox.DropboxOAuth2FlowNoRedirect(
        dropbox_app_key,
        dropbox_app_secret,
        token_access_type="offline",
    )

    authorization_url = dropbox_oauth_flow.start()

    placeholder = st.empty()

    with placeholder.container():
        st.title(":key: Dropbox Authentication")
        st.markdown(f"1. Go to [this url]({authorization_url}).")
        st.write('2. Click "Allow" (you might have to log in first).')
        st.write("3. Copy the authorization code.")
        st.write("4. Enter the authorization code here: ")
        authorization_code = st.text_input("")

    try:
        complete_oauth_flow = dropbox_oauth_flow.finish(authorization_code)
        return placeholder, complete_oauth_flow
    except:  # noqa: E722
        if authorization_code is not None and len(authorization_code):
            st.error("Invalid authorization code!")
        return placeholder, None


@st.cache(allow_output_mutation=True)
def st_dropbox_connect():
    if "DROPBOX_ACCESS_TOKEN" in os.environ:
        return dbx.dropbox_connect_access_token_only(st.session_state["access_token"])
    return dbx.dropbox_connect(
        st.session_state["access_token"], st.session_state["refresh_token"]
    )


@st.cache
def st_list_contents_of(folder_name, dropbox_app=None):
    dbx_app = dropbox_app or st_dropbox_connect()
    return dbx.dropbox_list_contents_of(dbx_app, folder_name)


@st.cache
def st_load_dataframe(filename, dropbox_path, dropbox_app=None):
    dbx_app = dropbox_app or st_dropbox_connect()
    return data_loader.dbx_load_dataframe(filename, dropbox_path, dbx_app)


@st.cache
def st_load_metadata(dropbox_app=None):
    dbx_app = dropbox_app or st_dropbox_connect()
    return data_loader.dbx_load_metadata(dbx_app)


@st.cache
def st_load_geo_metadata(dropbox_app=None):
    dbx_app = dropbox_app or st_dropbox_connect()
    return data_loader.dbx_load_geo_metadata(dbx_app)


@st.cache(allow_output_mutation=True)
def st_load_earth(dropbox_app=None):
    dbx_app = dropbox_app or st_dropbox_connect()
    return data_loader.dbx_load_earth(dbx_app)


@st.cache(allow_output_mutation=True)
def st_load_annotations(filename, dropbox_app=None):
    dbx_app = dropbox_app or st_dropbox_connect()
    return data_loader.dbx_load_label_annotations(filename, dbx_app)


@st.cache(allow_output_mutation=True)
def st_load_photo_list(photos_folder_name, photos_folder_path, dropbox_app=None):

    dbx_app = dropbox_app or st_dropbox_connect()

    index_file = f"index-{photos_folder_name}.csv"
    photos_folder_contents = st_list_contents_of(photos_folder_path).item_name.values

    if index_file in photos_folder_contents:
        return st_load_dataframe(
            f"index-{photos_folder_name}.csv", photos_folder_path, dbx_app
        ).item_name
    else:
        return st_list_contents_of(
            folder_name=photos_folder_path, dropbox_app=dbx_app
        ).item_name


@st.cache
def st_load_photo_list_and_metadata(
    photos_folder,
    photos_root_path,
    geo_metadata: bool = False,
    dropbox_app=None,
):
    photos_path = f"{photos_root_path or DROPBOX_RAW_PHOTOS_ROOT}"

    dbx_app = dropbox_app or st_dropbox_connect()

    if geo_metadata:
        photos_metadata = st_load_geo_metadata(dropbox_app=dbx_app)
    else:
        photos_metadata = st_load_metadata(dropbox_app=dbx_app)

    if photos_folder is None:
        return photos_metadata, photos_metadata.imageURL

    photos_list = st_load_photo_list(photos_folder, photos_path, dbx_app)

    available_photos_metadata = photos_metadata.loc[
        photos_metadata.imageURL.isin(photos_list)
    ]

    return available_photos_metadata, photos_list


@st.cache(allow_output_mutation=True)
def st_load_photo(
    photo_name,
    folder_name,
    dropbox_app=None,
    greyscale_only: bool = False,
):
    dbx_app = dropbox_app or st_dropbox_connect()
    dbx_path = f"{DROPBOX_RAW_PHOTOS_ROOT}/{folder_name}"
    return data_loader.dbx_load_photo(photo_name, dbx_path, dbx_app, greyscale_only)


@st.cache(allow_output_mutation=True)
def st_load_photo_from_roof_id(
    roof_id, photos_metadata, chosen_folder, dropbox_app=None, greyscale_only=False
):
    dbx_app = dropbox_app or st_dropbox_connect()
    dbx_path = f"{DROPBOX_RAW_PHOTOS_ROOT}/{chosen_folder}"

    return data_loader.dbx_load_photos_from_roof_id(
        roof_id, photos_metadata, dbx_path, dbx_app, greyscale_only
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


def load_and_crop_roof_from_roof_id(
    roof_id,
    photos_metadata,
    dropbox_path,
    dropbox_app=None,
    greyscale_only: bool = False,
):
    roof_px_coord, obstacles_px_coord = get_coordinates_from_roof_id(
        roof_id, photos_metadata
    )

    if greyscale_only:
        greyscale_image = st_load_photo_from_roof_id(
            roof_id, photos_metadata, dropbox_path, dropbox_app, greyscale_only
        )
        return rotate_and_crop_roof(greyscale_image, roof_px_coord)

    bgr_image, greyscale_image = st_load_photo_from_roof_id(
        roof_id, photos_metadata, dropbox_path, dropbox_app
    )

    k2_labelled_image = draw_boundaries(bgr_image, roof_px_coord, obstacles_px_coord)

    cropped_image = rotate_and_crop_roof(bgr_image, roof_px_coord)
    bgr_roof = draw_obstacles(cropped_image, roof_px_coord, obstacles_px_coord)
    
    greyscale_roof = rotate_and_crop_roof(greyscale_image, roof_px_coord)

    return k2_labelled_image, bgr_roof, greyscale_roof


def obstacle_detection_pipeline(
    greyscale_roof,
    sigma,
    filtering_method,
    binarization_method,
    blocksize,
    tolerance,
    boundary_type,
    return_filtered_roof: bool = False,
):

    filtered_roof = filtering_step(greyscale_roof, sigma, filtering_method.lower())

    if binarization_method == "Simple":
        binarized_roof = binarization_step(filtered_roof, method="s")
    elif binarization_method == "Adaptive":
        binarized_roof = binarization_step(
            filtered_roof, method="a", adaptive_kernel_size=blocksize
        )
    else:
        binarized_roof = binarization_step(
            filtered_roof, method="c", composite_tolerance=tolerance
        )

    blurred_roof = morphological_opening_step(binarized_roof)

    boundary_type = "box" if boundary_type == "Bounding Box" else "polygon"

    blobs, roof_with_bboxes, obstacles_coordinates = detect_obstacles(
        blurred_roof=blurred_roof,
        source_image=greyscale_roof,
        box_or_polygon=boundary_type,
        min_area="auto",
    )

    if tolerance == None:
        tolerance = 25

    blobs_composite = detect_obstacles_composite(
        im_in=filtered_roof,
        source_image=greyscale_roof,
        box_or_polygon=boundary_type,
        min_area="auto",
        tol=tolerance
    )

    edges_im = edge_detection(filtered_roof)
    hough_im = hough_transform(greyscale_roof)

    if return_filtered_roof:
        return blobs, roof_with_bboxes, obstacles_coordinates, filtered_roof, blobs_composite, edges_im, hough_im
    return blobs, roof_with_bboxes, obstacles_coordinates


def save_annotations_to_dropbox(
    data_to_upload, filename, destination_folder, make_checkpoint=False
):

    dbx_app = st_dropbox_connect()

    if make_checkpoint:
        timestamp = datetime.now().replace(microsecond=0).strftime("%Y_%m_%d-%H_%M_%S")
        filename = f"{timestamp}-{filename}"

    file_to_upload = f"/tmp/{filename}"
    destination_path = f"{destination_folder}/{filename}.csv"

    data_to_upload.to_csv(file_to_upload, index=False)

    dbx.dropbox_upload_file_to(dbx_app, file_to_upload, destination_path)
