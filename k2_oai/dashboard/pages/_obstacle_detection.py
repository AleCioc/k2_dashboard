from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from k2_oai.dashboard import utils
from k2_oai.io import dropbox as dbx

_DBX_PHOTOS_PATH = "/k2/raw_photos"
_DBX_LABELS_PATH = "/k2/metadata/transformed_data"


def _load_random_photo(roofs_list):
    st.session_state["roof_id_selector"] = np.random.choice(roofs_list.roof_id)


def _mark_photo(mark: str, roof_id=None, label_data=None):
    if roof_id is None:
        roof_id = st.session_state["roof_id_selector"]

    if label_data is None:
        label_data = st.session_state["obstacles_hyperparameters"]

    label_data.loc[label_data["roof_id"] == roof_id, "label_quality"] = mark


def _record_hyperparams(
    sigma,
    filtering_method,
    binarization_method,
    bin_adaptive_kernel,
    bin_composite_tolerance,
    drawing_technique,
    roof_id=None,
    label_data=None,
):

    if roof_id is None:
        roof_id = st.session_state["roof_id_selector"]

    if label_data is None:
        label_data = st.session_state["obstacles_hyperparameters"]

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


def _save_data_to_dropbox(
    hyperparams_data=None,
    filename="obstacle_detection_hyperparameters.csv",
    upload_to="/k2/hyperparameters",
):

    if hyperparams_data is None:
        hyperparams_data = st.session_state["obstacles_hyperparameters"]

    dbx_app = utils.dbx_get_connection()

    timestamp = datetime.now().replace(microsecond=0).strftime("%Y_%m_%d-%H_%M_%S")

    target_file_path = f"/tmp/{timestamp}-{filename}.csv"
    upload_path = f"{upload_to}/{timestamp}-{filename}.csv"

    (
        hyperparams_data.dropna(subset=["sigma", "label_quality"], how="all").to_csv(
            target_file_path, index=False
        )
    )

    dbx.upload_file_to_dropbox(dbx_app, target_file_path, upload_path)


def obstacle_detection_page():

    # +-------------------------------------+
    # | Adjust title and sidebar, load data |
    # +-------------------------------------+

    st.title(":house_with_garden: Obstacle Detection Dashboard")

    with st.sidebar:

        st.subheader("Data Source")

        # get options for `chosen_folder`
        photos_folders = utils.dbx_list_dir_contents(_DBX_PHOTOS_PATH).item_name

        chosen_folder = st.selectbox(
            "Select the folder to load the photos from: ",
            options=photos_folders,
            index=3,
        )

        photos_metadata, dbx_photo_list = utils.dbx_get_photos_and_metadata(
            photos_folder=chosen_folder,
            photos_root_path=_DBX_PHOTOS_PATH,
        )

        st.info(f"Available photos: {dbx_photo_list.shape[0]}")

        st.markdown("---")

    with st.expander("Click to see the list of photos"):
        st.dataframe(dbx_photo_list)

    with st.expander("Click to see the photos' metadata"):
        st.dataframe(photos_metadata)

    # +------------------------------------------+
    # | Cache DataFrame to store hyperparameters |
    # +------------------------------------------+

    if "obstacles_hyperparameters" not in st.session_state:
        st.session_state["obstacles_hyperparameters"] = (
            photos_metadata[["roof_id", "imageURL"]]
            .drop_duplicates("roof_id")
            .sort_values("roof_id")
            .assign(
                label_quality=np.NaN,
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
        ].loc[lambda df: df.sigma.isna()]

    obstacles_hyperparameters = st.session_state["obstacles_hyperparameters"]
    roofs_to_label = st.session_state["roofs_to_label"]

    # +----------------+
    # | Choose roof id |
    # ++---------------+

    with st.sidebar:

        st.subheader("Target Roof")

        # roof ID randomizer
        # ------------------
        st.write("You can choose a roof ID randomly...")

        buf, st_rand, buf = st.columns((2, 1, 2))

        st_rand.button("🔀", on_click=_load_random_photo, args=(roofs_to_label,))

        # roof ID selector
        # ----------------
        st.write("...or choose it manually:")
        chosen_roof_id = st.selectbox(
            "Roof identifier:",
            options=roofs_to_label,
            help="Identifier of the roof, whose label we want to inspect.",
            key="roof_id_selector",
        )

        st.markdown("---")

    k2_labelled_image, bgr_roof, greyscale_roof = utils.crop_roofs_from_roof_id(
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
            max_value=151,
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
            chosen_blocksize = st.text_input(
                """
                Size of the pixel neighbourhood (positive, odd integer):
                If 'auto', tolerance will be deduced from the histogram's variance
                """,
                key="bin_adaptive_kernel",
            )
            chosen_tolerance = None
        elif chosen_binarisation_method == "Composite":
            chosen_tolerance = st.text_input(
                "Insert the desired tolerance for composite binarisation",
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
            on_click=_record_hyperparams,
            args=(
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

    # +-------------------+
    # | Labelling Actions |
    # +-------------------+

    buf, st_keep, st_drop, st_maybe, buf = st.columns((2, 1, 1, 1, 2))

    st_keep.button(
        "Keep label",
        help="Mark the label as good",
        on_click=_mark_photo,
        args=("Y",),
    )

    st_drop.button(
        "Drop label",
        help="Mark the label as bad",
        on_click=_mark_photo,
        args=("N",),
    )

    st_maybe.button(
        "Maybe",
        help="Do not mark the label and move on",
        on_click=_mark_photo,
        args=("M",),
    )

    # +----------------------+
    # | Hyperparameters Data |
    # +----------------------+

    st_data, st_save = st.columns((6, 1))

    with st_data:
        with st.expander("View stored hyperparameters"):
            st.dataframe(
                obstacles_hyperparameters.dropna(
                    subset=["label_quality", "sigma"], how="all"
                )
            )

    if st_save.button("💾", help="Save the labels vetted so far to Dropbox"):
        _save_data_to_dropbox(obstacles_hyperparameters)
