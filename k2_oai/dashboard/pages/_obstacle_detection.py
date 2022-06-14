"""
Dashboard mode to explore the OpenCV pipeline for obstacle detection and annotate the
hyperparameters of each photo
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from k2_oai.dashboard import components, utils
from k2_oai.io.dropbox_paths import DROPBOX_HYPERPARAM_ANNOTATIONS_PATH

__all__ = ["obstacle_detection_page"]


def annotate_hyperparameters(
    roof_id,
    label_data,
    sigma,
    filtering_method,
    binarization_method,
    bin_adaptive_kernel,
    bin_composite_tolerance,
    drawing_technique,
):

    label_data.loc[lambda df: df.roof_id == roof_id, "sigma"] = sigma
    label_data.loc[
        lambda df: df.roof_id == roof_id, "filtering_method"
    ] = filtering_method
    label_data.loc[
        lambda df: df.roof_id == roof_id, "binarization_method"
    ] = binarization_method
    label_data.loc[
        lambda df: df.roof_id == roof_id, "bin_adaptive_kernel"
    ] = bin_adaptive_kernel
    label_data.loc[
        lambda df: df.roof_id == roof_id, "bin_composite_tolerance"
    ] = bin_composite_tolerance
    label_data.loc[
        lambda df: df.roof_id == roof_id, "drawing_technique"
    ] = drawing_technique


def load_random_photo(roofs_list):
    st.session_state["roof_id"] = np.random.choice(roofs_list)


def obstacle_detection_page():

    # +-------------------------------------+
    # | Adjust title and sidebar, load data |
    # +-------------------------------------+

    st.title(":house_with_garden: Obstacle Detection Dashboard")
    st.write(
        "Explore the results of the obstacle detection algorithm,",
        "and adjust the hyperparameters to improve the results.",
        "You can also save hyperparametres you chose to a Dropbox file.",
    )

    # +---------------------------------------------+
    # | Load photos metadata from a selected folder |
    # +---------------------------------------------+

    with st.sidebar:
        (
            chosen_folder,
            photos_metadata,
            photo_list,
        ) = components.sidebar_chose_photo_folder()

        st.markdown("---")

    # +------------------------------------------+
    # | Cache DataFrame to store hyperparameters |
    # +------------------------------------------+

    if "obstacles_hyperparameters" not in st.session_state:
        st.session_state["obstacles_hyperparameters"] = (
            photos_metadata[["roof_id", "imageURL"]]
            .drop_duplicates("roof_id")
            .sort_values("roof_id")
            .assign(
                sigma=np.NaN,
                filtering_method=np.NaN,
                binarization_method=np.NaN,
                bin_adaptive_kernel=np.NaN,
                bin_composite_tolerance=np.NaN,
                drawing_technique=np.NaN,
            )
        )

    if "roofs_to_label" not in st.session_state:
        st.session_state["roofs_to_label"] = st.session_state[
            "obstacles_hyperparameters"
        ].loc[lambda df: df["sigma"].isna()]

    obstacles_hyperparameters = st.session_state["obstacles_hyperparameters"]
    roofs_to_label = st.session_state["roofs_to_label"]

    # +----------------+
    # | Choose roof id |
    # +----------------+

    with st.sidebar:

        st.subheader("Target Roof")

        # roof ID randomizer
        # ------------------
        st.write("Choose a roof ID randomly...")

        buf, st_rand, buf = st.columns((2, 1, 2))

        st_rand.button("🔀", on_click=load_random_photo, args=(roofs_to_label,))

        # roof ID selector
        # ----------------
        st.write("...or manually:")
        chosen_roof_id = st.selectbox(
            "Roof identifier:",
            options=roofs_to_label,
            help="Identifier of the roof, whose label we want to inspect.",
            key="roof_id",
        )

        st.markdown("---")

    k2_labelled_image, bgr_roof, greyscale_roof = utils.load_and_crop_roof_from_roof_id(
        int(chosen_roof_id), photos_metadata, chosen_folder
    )
    # +-----------------------------+
    # | Obstacle Detection Pipeline |
    # +-----------------------------+

    st.subheader(f"Target roof's identifier: `{chosen_roof_id}`")

    with st.sidebar:

        st_subheader, st_save_params = st.columns((4, 1))

        st_subheader.subheader("Model Hyperparameters")

        chosen_sigma = st.slider(
            "Insert sigma for filtering (positive, odd integer):",
            min_value=1,
            step=2,
            key="sigma",
        )

        chosen_filtering_method = st.radio(
            "Choose filtering method:",
            options=("Bilateral", "Gaussian"),
            key="filtering_method",
        )

        chosen_binarisation_method = st.radio(
            "Select the desired binarisation method",
            options=("Simple", "Adaptive", "Composite"),
            key="binarization_method",
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
                key="bin_adaptive_kernel",
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
                key="bin_composite_tolerance",
            )
            chosen_blocksize = None
        else:
            chosen_blocksize, chosen_tolerance = None, None

        chosen_drawing_technique = st.radio(
            "Select the desired drawing technique",
            options=("Bounding Box", "Bounding Polygon"),
            key="drawing_technique",
        )

        st_save_params.button(
            "💾",
            help="Record Hyperparameters",
            on_click=annotate_hyperparameters,
            args=(
                chosen_roof_id,
                obstacles_hyperparameters,
                chosen_sigma,
                chosen_filtering_method,
                chosen_binarisation_method,
                chosen_blocksize,
                chosen_tolerance,
                chosen_drawing_technique,
            ),
        )

    (
        obstacle_blobs,
        roof_with_bboxes,
        obstacles_coordinates,
        filtered_gs_roof,
        obstacle_blobs_composite,
        edges_image
    ) = utils.obstacle_detection_pipeline(
        greyscale_roof=greyscale_roof,
        sigma=chosen_sigma,
        filtering_method=chosen_filtering_method,
        binarization_method=chosen_binarisation_method,
        blocksize=chosen_blocksize,
        tolerance=chosen_tolerance,
        boundary_type=chosen_drawing_technique,
        return_filtered_roof=True,
    )

    # +-------------------------+
    # | Roof & Color Histograms |
    # +-------------------------+

    st_roof, st_histograms = st.columns((1, 1))

    # original roof
    # -------------
    st_roof.image(
        k2_labelled_image,
        use_column_width=True,
        channels="BGRA",
        caption="Original image with database labels",
    )

    # RGB color histogram
    # -------------------
    fig, ax = plt.subplots(figsize=(3, 1))

    n, bins, patches = ax.hist(
        bgr_roof[:, :, 0].flatten(), bins=50, edgecolor="blue", alpha=0.5
    )
    n, bins, patches = ax.hist(
        bgr_roof[:, :, 1].flatten(), bins=50, edgecolor="green", alpha=0.5
    )
    n, bins, patches = ax.hist(
        bgr_roof[:, :, 2].flatten(), bins=50, edgecolor="red", alpha=0.5
    )

    ax.set_title("Cropped Roof RGB Histogram")
    ax.set_xlim(0, 255)

    st_histograms.pyplot(fig, use_column_width=True)

    # greyscale histogram
    # -------------------
    fig, ax = plt.subplots(figsize=(3, 1))

    n, bins, patches = ax.hist(
        filtered_gs_roof.flatten(), bins=range(256), edgecolor="black", alpha=0.9
    )

    ax.set_title("Roof Greyscale Histogram After Filtering")
    ax.set_xlim(0, 255)

    st_histograms.pyplot(fig, use_column_width=True)

    # +--------------------+
    # | Plot Model Results |
    # +--------------------+

    st.subheader("Obstacle Detection Steps, Visualized")

    st_results_widgets = st.columns((1, 1))

    st_results_widgets[0].image(
        bgr_roof,
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
        caption=f"Auto Labelled {chosen_drawing_technique}",
    )

    st_results_widgets[0].image(
        (obstacle_blobs_composite * 60) % 256,
        use_column_width=True,
        caption="Composite blobs",
    )

    st_results_widgets[1].image(
        edges_image,
        use_column_width=True,
        caption="Edge detection",
    )

    st_data, st_save = st.columns((7, 1))

    with st_data:
        with st.expander("View stored hyperparameters"):
            st.dataframe(obstacles_hyperparameters.dropna(subset="sigma"))

    if st_save.button("💾", help="Save hyperparameters to Dropbox"):
        utils.save_annotations_to_dropbox(
            obstacles_hyperparameters.dropna(subset="sigma"),
            "obstacles-annotated_hyperparameters",
            DROPBOX_HYPERPARAM_ANNOTATIONS_PATH,
        )
