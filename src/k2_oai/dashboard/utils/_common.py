"""
Common functions (e.g. not related to load data from Dropbox).
"""

from datetime import datetime

import pandas as pd
import streamlit as st
from numpy import ndarray
from pandas import DataFrame

from k2_oai.obstacle_detection import (
    binarization_step,
    detect_obstacles,
    filtering_step,
    morphological_opening_step,
)

__all__ = [
    "obstacle_detection_pipeline",
    "make_filename",
    "annotate_labels",
]


def obstacle_detection_pipeline(
    roof: ndarray,
    sigma: int,
    filtering_method,
    binarization_method: str,
    blocksize,
    tolerance,
    boundary_type,
    return_filtered_roof: bool = False,
):

    filtered_roof = filtering_step(roof, sigma, filtering_method.lower())

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
        source_image=roof,
        box_or_polygon=boundary_type,
        min_area="auto",
    )

    if return_filtered_roof:
        return blobs, roof_with_bboxes, obstacles_coordinates, filtered_roof
    return blobs, roof_with_bboxes, obstacles_coordinates


def make_filename(filename: str, use_checkpoints: bool = False):

    if filename == "New Checkpoint":
        filename = "checkpoint.csv"
    elif not filename.endswith(".csv"):
        filename = f"{filename}.csv"

    if use_checkpoints:
        return f"{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}-{filename}"
    return filename


def annotate_labels(
    marks: dict,
    session_state_key: str,
    roof_id: int,
    photos_folder: str,
    metadata: DataFrame,
    mode: str,
):
    image_url = metadata.loc[metadata["roof_id"] == roof_id, "imageURL"].values[0]

    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    if mode == "labels":
        marks_dtypes = {mark: int for mark in marks.keys()}
    elif mode == "hyperparameters" or mode == "hyperparams":
        marks_dtypes = {
            "sigma": int,
            "filtering_method": str,
            "binarization_method": str,
            "blocksize": float,
            "tolerance": float,
            "boundary_type": str,
        }
    else:
        raise ValueError(f"Invalid mode {mode}. Must be 'labels' or 'hyperparameters'.")

    new_row = pd.DataFrame(
        [
            {
                "roof_id": roof_id,
                "annotation_time": timestamp,
                "imageURL": image_url,
                "photos_folder": photos_folder,
                **marks,
            }
        ]
    ).astype({"roof_id": int, **marks_dtypes})

    st.session_state[session_state_key] = (
        pd.concat([st.session_state[session_state_key], new_row], ignore_index=True)
        .astype({"roof_id": float, **marks_dtypes})
        .drop_duplicates(subset=["roof_id"], keep="last")
        .sort_values("roof_id")
        .reset_index(drop=True)
    )
