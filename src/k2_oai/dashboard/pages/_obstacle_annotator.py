"""
Dashboard page/mode to accept or reject the obstacle label available from the database.
"""

import streamlit as st

from k2_oai.dashboard import utils
from k2_oai.dashboard.components import buttons, sidebar

__all__ = ["obstacle_annotator_page"]


def obstacle_annotator_page(
    mode: str = "labels",
    geo_metadata: bool = False,
    only_folders: bool = True,
    key_photos_folder: str = "photos_folder",
    key_drop_duplicates: str = "drop_duplicates",
    key_annotations_only: str = "annotations_only",
    key_annotations_cache: str = "labels_annotations",
    key_annotations_file: str = "labels_annotations_file",
):
    st.title(":mag: Obstacle Annotation Tool")

    with st.sidebar:

        # +---------------------+
        # | select data sources |
        # +---------------------+

        obstacles_metadata, all_annotations, remaining_roofs = sidebar.configure_data(
            key_photos_folder=key_photos_folder,
            key_drop_duplicates=key_drop_duplicates,
            key_annotations_cache=key_annotations_cache,
            key_annotations_file=key_annotations_file,
            key_annotations_only=key_annotations_only,
            mode=mode,
            geo_metadata=geo_metadata,
            only_folders=only_folders,
        )

        chosen_roof_id = buttons.choose_roof_id(obstacles_metadata, remaining_roofs)

        # +-------------------+
        # | labelling actions |
        # +-------------------+

        st.markdown("## :pencil: Mark the annotations")

        annotations_cache = st.session_state[key_annotations_cache]

        st.info(
            f"""
            Roofs annotated in this session: {annotations_cache.shape[0]}

            Annotations in `{st.session_state[key_annotations_file]}`:
            {all_annotations.shape[0] - annotations_cache.shape[0]}

            Total annotations: {all_annotations.shape[0]}
            """
        )

        is_perfectly_labelled = st.radio(
            label="Is the photo perfectly labelled?",
            options=[0, 1],
            index=0,
            help="0 means no, 1 means yes",
        )

        is_roof = st.radio(
            label="Does the label depict a roof?",
            options=[0, 1],
            index=0,
            help="e.g. grass was labelled instead of a roof",
        )

        roof_well_cropped = st.radio(
            label="Is the roof well cropped?",
            options=[0, 1],
            index=0,
            help="e.g. the cropped portion is smaller than the roof",
        )

        obstacles_well_cropped = st.radio(
            label="Are the obstacles well cropped?",
            options=[0, 1],
            index=0,
            help="e.g. the label is larger than the real obstacle",
        )

        all_obstacles_found = st.radio(
            label="Are all obstacles labelled?",
            options=[0, 1],
            index=0,
        )

        not_an_obstacle = st.radio(
            label="Was something other than an obstacle labelled?",
            options=[0, 1],
            index=0,
        )

        label_annotations = {
            "is_perfectly_labelled": is_perfectly_labelled,
            "is_roof": is_roof,
            "roof_well_cropped": roof_well_cropped,
            "obstacles_well_cropped": obstacles_well_cropped,
            "all_obstacles_found": all_obstacles_found,
            "not_an_obstacle": not_an_obstacle,
        }

        sidebar.write_and_save_annotations(
            new_annotations=label_annotations,
            annotations_data=all_annotations,
            annotations_savefile=st.session_state[key_annotations_file],
            roof_id=chosen_roof_id,
            photos_folder=st.session_state[key_photos_folder],
            metadata=obstacles_metadata,
            key_annotations_cache=key_annotations_cache,
            mode=mode,
        )

    # +------------------------+
    # | Load and plot the roof |
    # +------------------------+

    roof_id_label = all_annotations.loc[
        lambda df: df.roof_id == chosen_roof_id, "is_perfectly_labelled"
    ].values[0]

    if chosen_roof_id in all_annotations.roof_id.values:
        st.info(
            f"Roof {chosen_roof_id} is already annotated as "
            f"{'`perfectly labelled`' if roof_id_label else '`not perfectly labelled`'}"
        )
    else:
        st.warning(f"Roof {chosen_roof_id} is not annotated")

    photo, roof, labelled_photo, labelled_roof = utils.st_load_photo_and_roof(
        int(chosen_roof_id),
        obstacles_metadata,
        st.session_state[key_photos_folder],
    )

    st_labelled, st_not_labelled = st.columns(2)

    with st_labelled:
        st.image(
            labelled_roof,
            use_column_width=True,
            channels="BGRA",
            caption="Labelled Roof",
        )

        st.image(
            labelled_photo,
            use_column_width=True,
            channels="BGRA",
            caption="Cropped roof",
        )

    with st_not_labelled:
        st.image(
            roof,
            use_column_width=True,
            channels="BGRA",
            caption="Original image, labelled",
        )

        st.image(
            photo,
            use_column_width=True,
            channels="BGRA",
            caption="Original image",
        )

    # +----------------+
    # | Roof metadata  |
    # +----------------+

    with st.expander(f"Roof {chosen_roof_id} metadata:"):
        st.dataframe(
            obstacles_metadata.loc[obstacles_metadata.roof_id == chosen_roof_id]
        )

    with st.expander("View the annotations:", expanded=True):
        st.dataframe(all_annotations)
