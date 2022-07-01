"""
Dashboard mode to explore the OpenCV pipeline for obstacle detection and annotate the
hyperparameters of each photo
"""

import streamlit as st

from k2_oai.dashboard import utils
from k2_oai.dashboard.components import buttons, sidebar

__all__ = ["obstacle_detection_page"]


def obstacle_detection_page(
    mode: str = "hyperparameters",
    geo_metadata: bool = False,
    only_folders: bool = True,
    key_photos_folder: str = "photos_folder",
    key_drop_duplicates: str = "drop_duplicates",
    key_annotations_only: str = "annotations_only",
    key_annotations_cache: str = "hyperparams_annotations",
    key_annotations_file: str = "hyperparams_annotations_file",
):
    st.title(":house_with_garden: Obstacle Detection Dashboard")

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

        chosen_folder = st.session_state[key_photos_folder]

        chosen_roof_id = buttons.choose_roof_id(obstacles_metadata, remaining_roofs)

        # +-------------------+
        # | labelling actions |
        # +-------------------+

        st.markdown("## :control_knobs: Model Hyperparameters")

        annotations_cache = st.session_state[key_annotations_cache]

        st.info(
            f"Roofs annotated so far: {annotations_cache.shape[0]} "
            f"Roofs annotated in {st.session_state[key_annotations_file]}:"
            f"{all_annotations.shape[0]}"
        )

        chosen_sigma = st.slider(
            "Filtering sigma (positive, odd integer):",
            min_value=1,
            step=2,
        )

        chosen_filtering_method = st.radio(
            "Choose filtering method:",
            options=("Bilateral", "Gaussian"),
        )

        chosen_binarisation_method = st.radio(
            "Select the desired binarisation method",
            options=("Simple", "Adaptive", "Composite"),
        )

        if chosen_binarisation_method == "Adaptive":
            chosen_blocksize = st.slider(
                """
                Size of the pixel neighbourhood.
                If -1, it will be deduced from the image's size
                """,
                min_value=-1,
                max_value=255,
                step=2,
            )
            chosen_tolerance = None
        elif chosen_binarisation_method == "Composite":
            chosen_tolerance = st.slider(
                """
                Tolerance for the composite binarisation method.
                If -1, tolerance will be deduced from the histogram's variance
                """,
                min_value=-1,
                max_value=255,
            )
            chosen_blocksize = None
        else:
            chosen_blocksize, chosen_tolerance = None, None

        boundary_type = st.radio(
            "Select the desired drawing technique",
            options=("Bounding Box", "Bounding Polygon"),
        )

        annotations = {
            "sigma": chosen_sigma,
            "filtering_method": chosen_filtering_method,
            "binarization_method": chosen_binarisation_method,
            "blocksize": chosen_blocksize,
            "tolerance": chosen_tolerance,
            "boundary_type": boundary_type,
        }

        sidebar.write_and_save_annotations(
            new_annotations=annotations,
            annotations_data=all_annotations,
            annotations_savefile=st.session_state[key_annotations_file],
            roof_id=chosen_roof_id,
            photos_folder=chosen_folder,
            metadata=obstacles_metadata,
            key_annotations_cache=key_annotations_cache,
            mode=mode,
        )

    # +-------------------------+
    # | Roof & Color Histograms |
    # +-------------------------+

    photo, roof, labelled_photo, _labelled_roof = utils.st_load_photo_and_roof(
        int(chosen_roof_id), obstacles_metadata, chosen_folder
    )

    (
        obstacle_blobs,
        roof_with_bboxes,
        obstacles_coordinates,
        filtered_gs_roof,
    ) = utils.obstacle_detection_pipeline(
        roof=roof,
        sigma=chosen_sigma,
        filtering_method=chosen_filtering_method,
        binarization_method=chosen_binarisation_method,
        blocksize=chosen_blocksize,
        tolerance=chosen_tolerance,
        boundary_type=boundary_type,
        return_filtered_roof=True,
    )

    if chosen_roof_id in all_annotations.roof_id.values:
        st.info(f"Roof {chosen_roof_id} is already annotated")
    else:
        st.warning(f"Roof {chosen_roof_id} is not annotated")

    st_roof, st_labelled_roof = st.columns((1, 1))

    st_roof.image(
        photo,
        use_column_width=True,
        channels="BGRA",
        caption="Satellite photo",
    )

    st_labelled_roof.image(
        labelled_photo,
        use_column_width=True,
        channels="BGRA",
        caption="Satellite photo, labelled",
    )

    # +--------------------+
    # | Plot Model Results |
    # +--------------------+

    st.subheader("Obstacle Detection Steps, Visualized")

    st_results_widgets = st.columns((1, 1))

    st_results_widgets[0].image(
        roof,
        use_column_width=True,
        channels="BGRA",
        caption="Cropped Roof (RGB) with Database Labels",
    )

    st_results_widgets[0].image(
        filtered_gs_roof,
        use_column_width=True,
        caption="Cropped Roof (Greyscale) After Filtering",
    )

    st_results_widgets[1].image(
        (obstacle_blobs * 60) % 256,
        use_column_width=True,
        caption="Auto Obstacle Blobs (Greyscale)",
    )

    st_results_widgets[1].image(
        roof_with_bboxes,
        use_column_width=True,
        caption=f"Auto Labelled {boundary_type}",
    )

    with st.expander("View the annotations:", expanded=True):
        st.dataframe(all_annotations)
