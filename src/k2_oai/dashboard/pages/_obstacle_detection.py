"""
Dashboard mode to explore the OpenCV pipeline for obstacle detection and annotate the
hyperparameters of each photo
"""

import streamlit as st

from k2_oai.dashboard import utils
from k2_oai.dashboard.components import buttons, sidebar
from k2_oai.obstacle_detection import pipelines

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

        chosen_filtering_method = st.radio(
            "Choose filtering method:",
            options=("Bilateral", "Gaussian"),
        ).lower()

        chosen_filtering_sigma = st.slider(
            "Filtering sigma (positive, odd integer):",
            min_value=-1,
            step=2,
        )

        chosen_binarisation_method = st.radio(
            "Select the desired binarisation method:",
            options=("Simple", "Composite"),
        ).lower()

        if chosen_binarisation_method == "composite":
            chosen_binarisation_tolerance = st.slider(
                "Tolerance for the composite binarisation method. "
                "If -1, tolerance will be deduced from the histogram's variance",
                min_value=-1,
                max_value=255,
            )
        else:
            chosen_binarisation_tolerance = None

        chosen_erosion_kernel = st.slider(
            "Kernel shape for morphological opening/erosion. "
            "If -1, it will be inferred",
            min_value=-1,
            max_value=100,
            step=2,
        )

        chosen_min_area = st.slider(
            "Minimum obstacles area, in pixels",
            min_value=0,
            max_value=100,
        )

        st.radio(
            "Select the desired drawing technique",
            options=("Bounding Box", "Bounding Polygon"),
            key="drawing_technique",
        )

        if st.session_state["drawing_technique"] == "Bounding Box":
            chosen_boundary_type = "box"
        else:
            chosen_boundary_type = "polygon"

        roof_annotations = {
            "filtering_sigma": chosen_filtering_sigma,
            "filtering_method": chosen_filtering_method,
            "binarization_method": chosen_binarisation_method,
            "binarization_tolerance": chosen_binarisation_tolerance,
            "erosion_kernel_size": chosen_erosion_kernel,
            "obstacles_min_area": chosen_min_area,
            "boundary_type": chosen_boundary_type,
        }

        sidebar.write_and_save_annotations(
            new_annotations=roof_annotations,
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

    (
        satellite_photo,
        labelled_photo,
        cropped_roof,
        k2_labelled_roof,
    ) = utils.st_load_photo_and_roof(
        int(chosen_roof_id), obstacles_metadata, chosen_folder
    )

    roof_coordinates, k2_obstacles_coordinates = utils.get_coordinates_from_roof_id(
        int(chosen_roof_id), obstacles_metadata
    )

    (
        obstacles_coordinates,
        labelled_roof,
        obstacles_blobs,
    ) = pipelines.manual_obstacle_detection_pipeline(
        satellite_image=satellite_photo,
        roof_px_coordinates=roof_coordinates,
        filtering_method=chosen_filtering_method,
        filtering_sigma=chosen_filtering_sigma,
        binarization_method=chosen_binarisation_method,
        binarization_tolerance=chosen_binarisation_tolerance,
        erosion_kernel_size=chosen_erosion_kernel,
        obstacle_minimum_area=chosen_min_area,
        obstacle_boundary_type=chosen_boundary_type,
        using_dashboard=True,
    )

    if chosen_roof_id in all_annotations.roof_id.values:
        st.info(f"Roof {chosen_roof_id} is already annotated")
    else:
        st.warning(f"Roof {chosen_roof_id} is not annotated")

    st_roof, st_labelled_roof = st.columns((1, 1))

    st_roof.image(
        satellite_photo,
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

    st_results_widgets = st.columns(3)

    st_results_widgets[0].image(
        cropped_roof,
        use_column_width=True,
        channels="BGRA",
        caption="Cropped Roof (RGB) with Database Labels",
    )

    st_results_widgets[1].image(
        (obstacles_blobs * 60) % 256,
        use_column_width=True,
        caption="Obstacle Blobs (Greyscale)",
    )

    st_results_widgets[2].image(
        labelled_roof,
        use_column_width=True,
        caption=f"Auto Labelled {chosen_boundary_type}",
    )

    with st.expander("View the annotations:", expanded=True):
        st.dataframe(all_annotations)
