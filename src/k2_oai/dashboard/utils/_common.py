"""
Common functions (e.g. not related to load data from Dropbox).
"""

from datetime import datetime

import pandas as pd
import streamlit as st
from pandas import DataFrame

__all__ = [
    "make_filename",
    "annotate_labels",
]


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
            key: (str if key.endswith("_method") else int) for key in marks.keys()
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
