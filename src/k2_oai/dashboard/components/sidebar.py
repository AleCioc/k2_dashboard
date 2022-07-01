"""
Dashboard components, preferably for the sidebar, to select data sources and save data.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st
from numpy import ndarray
from pandas import DataFrame, Series

from k2_oai.dashboard import utils
from k2_oai.dropbox import (
    DROPBOX_HYPERPARAM_ANNOTATIONS_PATH,
    DROPBOX_LABEL_ANNOTATIONS_PATH,
    DROPBOX_RAW_PHOTOS_ROOT,
)

__all__ = [
    "configure_data",
    "choose_folder_to_load_metadata",
    "choose_annotations_checkpoint",
    "write_and_save_annotations",
]


def configure_data(
    mode: str,
    key_photos_folder: str,
    key_drop_duplicates: str,
    key_annotations_cache: str,
    key_annotations_file: str,
    key_annotations_only: str,
    geo_metadata: bool = False,
    only_folders: bool = True,
) -> tuple[DataFrame, DataFrame, Series]:
    """
    1. Load metadata
    2. allow to drop duplicates
    3. set up streamlit cache
    4. load annotations or not (RENAME checkpoint to None)
    5. if NO checkpoint, annotations == cache
    6. if using source, annotations == cache + source
    7. filter only labelled image wrt source file
    """

    obstacles_metadata, _photo_list = choose_folder_to_load_metadata(
        key_photos_folder,
        key_drop_duplicates,
        geo_metadata,
        only_folders,
    )

    # configure cache
    if key_annotations_cache not in st.session_state:
        st.session_state[key_annotations_cache] = pd.DataFrame(
            columns=["roof_id", "annotation_time"]
        )

    cached_annotations = st.session_state[key_annotations_cache]

    loaded_annotations = choose_annotations_checkpoint(key_annotations_file, mode)

    if loaded_annotations is None:
        all_annotations = cached_annotations
    else:
        all_annotations = (
            pd.concat([loaded_annotations, cached_annotations], ignore_index=True)
            .sort_values("roof_id")
            .reset_index(drop=True)
            .drop_duplicates(subset=["roof_id", "annotation_time"], keep="last")
        )

    remaining_roofs = obstacles_metadata.roof_id.loc[
        lambda df: ~df.isin(all_annotations.roof_id.values)
    ].unique()

    # filter only annotated photos
    if st.checkbox("Only show well-labelled roofs", key=key_annotations_only):

        well_labelled_roofs = all_annotations.loc[
            lambda df: df["is_perfectly_labelled"] == 1
        ]

        obstacles_metadata = obstacles_metadata.loc[
            lambda df: df.roof_id.isin(well_labelled_roofs.roof_id.values)
        ]

        if obstacles_metadata.empty:
            st.error(
                "No photos have been annotated in this session, "
                "or no photo in this folder were annotated. "
                "Please uncheck the `Show annotated photos only` checkbox, "
                "or select a different folder."
            )
            st.stop()

        remaining_roofs = obstacles_metadata.roof_id.loc[
            lambda df: df.isin(all_annotations.roof_id.values)
        ]

    return obstacles_metadata, all_annotations, remaining_roofs


def choose_folder_to_load_metadata(
    key_photos_folder: str,
    key_drop_duplicates: str,
    geo_metadata: bool = False,
    only_folders: bool = True,
) -> tuple[DataFrame, DataFrame]:

    st.markdown("## :open_file_folder: Photos Folder")

    # get options for `chosen_folder`
    root_contents = utils.st_listdir_no_cache(DROPBOX_RAW_PHOTOS_ROOT).item_name.values
    photos_folders = sorted(file for file in root_contents if not file.endswith(".csv"))

    if not only_folders:
        photos_folders.insert(0, None)

    chosen_folder = st.selectbox(
        "Select the folder to load the photos from:",
        options=photos_folders,
        index=0,
        key=key_photos_folder,
    )

    obstacles_metadata, photo_list = utils.st_load_photo_list_and_metadata(
        photos_folder=chosen_folder,
        geo_metadata=geo_metadata,
    )

    obstacles_metadata = choose_to_drop_duplicates(
        obstacles_metadata, photo_list, key_drop_duplicates
    )

    return obstacles_metadata, photo_list


def choose_to_drop_duplicates(
    obstacles_metadata: DataFrame,
    photo_list: Series | ndarray,
    key_drop_duplicates: str,
) -> DataFrame:
    st.checkbox("Drop duplicate roofs", key=key_drop_duplicates)

    view_duplicates_count(
        obstacles_metadata,
        photo_list,
    )

    if st.session_state[key_drop_duplicates]:
        obstacles_metadata = obstacles_metadata.drop_duplicates(
            subset=["imageURL", "pixelCoordinates_obstacle"]
        )

    return obstacles_metadata


def view_duplicates_count(
    obstacles_metadata: DataFrame,
    photo_list: Series | ndarray,
):
    total_obst = obstacles_metadata.pixelCoordinates_obstacle.notna().shape[0]
    unique_obst = obstacles_metadata.drop_duplicates(
        subset=["imageURL", "pixelCoordinates_obstacle"]
    ).shape[0]

    roofs_metadata = obstacles_metadata.drop_duplicates(subset="roof_id")
    total_roofs = roofs_metadata.pixelCoordinates_roof.notna().shape[0]
    unique_roofs = roofs_metadata.drop_duplicates(
        subset=["imageURL", "pixelCoordinates_roof"]
    ).shape[0]

    st.info(
        f"""
        Photos with tagged obstacles: {photo_list.shape[0]}

        Total roofs: {total_roofs}

        Total obstacles: {total_obst}
        """
    )

    st.warning(
        f"""
        Duplicate roofs: {total_roofs - unique_roofs}
        ({(total_roofs - unique_roofs) / total_roofs * 100:.2f}%)

        Duplicate obstacles: {total_obst - unique_obst}
        ({(total_obst - unique_obst) / total_obst * 100:.2f}%)
        """
    )


def choose_annotations_checkpoint(
    key_annotations_file: str, mode: str
) -> DataFrame | None:

    if mode == "labels":
        dropbox_path = DROPBOX_LABEL_ANNOTATIONS_PATH
    elif mode == "hyperparameters" or mode == "hyperparams":
        dropbox_path = DROPBOX_HYPERPARAM_ANNOTATIONS_PATH
    else:
        raise ValueError(f"Invalid {mode = }. Must be either 'labels' or 'hyperparams'")

    st.markdown("## :card_index_dividers: Annotations")

    folder_contents = sorted(
        utils.st_listdir_no_cache(dropbox_path).item_name.to_list(),
        reverse=True,
    )

    annotations_file = st.selectbox(
        "Load a file to source existing annotations:",
        options=[None] + folder_contents,
        help="If `None`, no data will be sourced",
        index=1 if len(folder_contents) > 0 else 0,
        key=key_annotations_file,
    )

    if annotations_file is None:
        return None
    return utils.st_load_annotations(annotations_file).convert_dtypes()


def write_and_save_annotations(
    new_annotations: dict[str, Any],
    annotations_data: DataFrame,
    annotations_savefile: str,
    roof_id: int,
    photos_folder: str,
    metadata: DataFrame,
    key_annotations_cache: str,
    mode: str,
):
    if mode == "labels":
        dropbox_path = DROPBOX_LABEL_ANNOTATIONS_PATH
    elif mode == "hyperparameters" or mode == "hyperparams":
        dropbox_path = DROPBOX_HYPERPARAM_ANNOTATIONS_PATH
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'labels' or 'hyperparameters'")

    buf, st_annotate, buf, st_save, buf = st.columns((1.2, 1, 0.2, 1, 1.5))

    st_annotate.button(
        "📝",
        help="Write the annotations to the dataset",
        on_click=utils.annotate_labels,
        args=(
            new_annotations,
            key_annotations_cache,
            roof_id,
            photos_folder,
            metadata,
            mode,
        ),
    )

    savefile_name = st.text_input(
        label="Name of the file to save new annotations to:",
        value=annotations_savefile,
        help="Defaults to the same filename as the file you load "
        "- i.e. it will be overwritten. Specifying '.csv' is not necessary",
        key="savefile_name",
    )

    use_timestamp = st.radio(
        label="Save the annotations with a timestamp",
        options=[True, False],
        index=1,
        help="The file will be saved with a timestamp",
        key="use_timestamp",
    )

    filename = utils.make_filename(savefile_name, use_timestamp)

    if filename == annotations_savefile:
        st.warning(f"This will overwrite {annotations_savefile}!")

    if st_save.button("💾", help=f"Save annotations to {filename}"):
        utils.st_save_annotations(
            annotations_data,
            filename,
            dropbox_path,
        )
