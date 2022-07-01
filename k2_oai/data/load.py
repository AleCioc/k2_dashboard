"""
Loads data and photos from Dropbox
"""

import os

import cv2 as cv
import geopandas
import pandas as pd

from k2_oai import dropbox as dbx
from k2_oai.dropbox import DROPBOX_LABEL_ANNOTATIONS_PATH, DROPBOX_PHOTOS_METADATA_PATH
from k2_oai.utils.images import draw_roofs_and_obstacles_on_photo, rotate_and_crop_roof

__all__ = [
    "dbx_load_dataframe",
    "dbx_load_metadata",
    "dbx_load_geo_metadata",
    "dbx_load_label_annotations",
    "dbx_load_photo",
    "dbx_load_photos_from_roof_id",
]


def dbx_load_dataframe(filename, dropbox_path, dropbox_app):

    dropbox_file = f"{dropbox_path}/{filename}"

    dropbox_app.files_download_to_file(filename, dropbox_file)

    if filename.endswith(".parquet"):
        data = pd.read_parquet(filename)
    elif filename.endswith(".csv"):
        data = pd.read_csv(filename)
    else:
        raise ValueError("File must be either .parquet or .csv")

    os.remove(filename)

    return data


def dbx_load_geodataframe(filename, dropbox_path, crs, dropbox_app):

    dropbox_file = f"{dropbox_path}/{filename}"

    dropbox_app.files_download_to_file(filename, dropbox_file)

    data = geopandas.read_file(filename, crs=crs)
    os.remove(filename)

    return data


def dbx_load_metadata(dropbox_app):
    return dbx_load_dataframe(
        "join-roofs_images_obstacles.parquet",
        dropbox_path=DROPBOX_PHOTOS_METADATA_PATH,
        dropbox_app=dropbox_app,
    )


def dbx_load_geo_metadata(dropbox_app):
    return dbx_load_dataframe(
        "geometries-roofs_images_obstacles.parquet",
        dropbox_path=DROPBOX_PHOTOS_METADATA_PATH,
        dropbox_app=dropbox_app,
    )


def dbx_create_label_annotations(dropbox_app, num_checkpoints: int = 0):

    metadata_folder_contents = dbx.dropbox_listdir(
        DROPBOX_LABEL_ANNOTATIONS_PATH, dropbox_app
    )

    label_annotation_checkpoints = metadata_folder_contents.loc[
        lambda df: df.item_name.str.contains("-checkpoint-labels_annotations.csv")
    ]

    if num_checkpoints > 0:
        label_annotation_checkpoints = label_annotation_checkpoints.iloc[
            :num_checkpoints
        ]
    dataframes = [
        dbx_load_dataframe(file, DROPBOX_LABEL_ANNOTATIONS_PATH, dropbox_app)
        for file in label_annotation_checkpoints.item_name
    ]

    label_annotations = (
        pd.concat(dataframes, ignore_index=True)
        .dropna(subset=["roof_id"], keep="last")
        .sort_values("roof_id")
        .reset_index(drop=True)
    )

    label_annotations.to_csv("obstacles-labels_annotations.csv", index=False)

    dbx.dropbox_upload_file_to(
        dropbox_app,
        "obstacles-labels_annotations.csv",
        f"{DROPBOX_LABEL_ANNOTATIONS_PATH}/obstacles-labels_annotations.csv",
    )

    os.remove("obstacles-labels_annotations.csv")


def dbx_load_label_annotations(filename, dropbox_app):
    return dbx_load_dataframe(
        filename,
        dropbox_path=DROPBOX_LABEL_ANNOTATIONS_PATH,
        dropbox_app=dropbox_app,
    )


def dbx_load_photo(
    photo_name, dropbox_folder, dropbox_app, bgr_only=False, greyscale_only=False
):
    if greyscale_only and bgr_only:
        raise ValueError("`bgr_only` and `greyscale_only` cannot be both True")

    dropbox_path = f"{dropbox_folder}/{photo_name}"

    dropbox_app.files_download_to_file(photo_name, dropbox_path)

    if bgr_only:
        bgr_image = cv.imread(photo_name, 1)

        os.remove(photo_name)

        return bgr_image

    if greyscale_only:
        greyscale_image = cv.imread(photo_name, 0)

        os.remove(photo_name)

        return greyscale_image

    bgr_image = cv.imread(photo_name, 1)
    greyscale_image = cv.imread(photo_name, 0)

    os.remove(photo_name)

    return bgr_image, greyscale_image


def dbx_load_photos_from_roof_id(
    roof_id,
    metadata,
    dropbox_path,
    dropbox_app,
    bgr_only: bool = False,
    greyscale_only: bool = False,
):
    photo_name = metadata.loc[lambda df: df["roof_id"] == roof_id, "imageURL"].values[0]

    return dbx_load_photo(
        photo_name, dropbox_path, dropbox_app, bgr_only, greyscale_only
    )


def get_coordinates_from_roof_id(roof_id, metadata) -> tuple[str, list[str]]:

    roof_px_coordinates = metadata.loc[
        metadata.roof_id == roof_id, "pixelCoordinates_roof"
    ].iloc[0]

    obstacles_px_coordinates = [
        coord
        for coord in metadata.loc[
            metadata.roof_id == roof_id, "pixelCoordinates_obstacle"
        ].values
    ]

    return roof_px_coordinates, obstacles_px_coordinates


def load_and_crop_roof_from_roof_id(
    roof_id,
    metadata,
    dropbox_path,
    dropbox_app,
    greyscale_only: bool = False,
    bgr_only: bool = False,
    with_labels: bool = False,
):
    roof_px_coord, obstacles_px_coord = get_coordinates_from_roof_id(roof_id, metadata)

    if greyscale_only:
        greyscale_image = dbx_load_photos_from_roof_id(
            roof_id,
            metadata,
            dropbox_path,
            dropbox_app,
            greyscale_only=greyscale_only,
        )
        if with_labels:
            labelled_roof = draw_roofs_and_obstacles_on_photo(
                greyscale_image, roof_px_coord, obstacles_px_coord
            )
            return rotate_and_crop_roof(labelled_roof, roof_px_coord)
        return rotate_and_crop_roof(greyscale_image, roof_px_coord)

    if bgr_only:
        bgr_image = dbx_load_photos_from_roof_id(
            roof_id, metadata, dropbox_path, dropbox_app, bgr_only=bgr_only
        )
        if with_labels:
            labelled_roof = draw_roofs_and_obstacles_on_photo(
                bgr_image, roof_px_coord, obstacles_px_coord
            )
            return rotate_and_crop_roof(labelled_roof, roof_px_coord)
        return rotate_and_crop_roof(bgr_image, roof_px_coord)

    bgr_image, greyscale_image = dbx_load_photos_from_roof_id(
        roof_id, metadata, dropbox_path, dropbox_app
    )

    k2_labelled_image = draw_roofs_and_obstacles_on_photo(
        bgr_image, roof_px_coord, obstacles_px_coord
    )
    labelled_roof = rotate_and_crop_roof(k2_labelled_image, roof_px_coord)
    greyscale_roof = rotate_and_crop_roof(greyscale_image, roof_px_coord)

    return k2_labelled_image, labelled_roof, greyscale_roof
