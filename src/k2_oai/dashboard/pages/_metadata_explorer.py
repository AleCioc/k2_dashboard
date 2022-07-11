"""
Dashboard mode to view and explore roof distribution in the world, as well as the
satellite photos' zoom levels.
"""

import altair as alt
import numpy as np
import streamlit as st

from k2_oai.dashboard.components import sidebar

__all__ = ["metadata_explorer_page"]


def metadata_explorer_page(
    mode: str = "labels",
    geo_metadata: bool = True,
    only_folders: bool = False,
    key_photos_folder: str = "key_photos_folder",
    key_drop_duplicates: str = "drop_duplicates",
    key_annotations_only: str = "metadata_annotations_only",
    key_annotations_cache: str = "metadata_annotations",
    key_annotations_file: str = "metadata_annotations_file",
):
    st.title(":bar_chart: Metadata Explorer")

    # +------------------------------+
    # | Update Sidebar and Load Data |
    # +------------------------------+

    with st.sidebar:

        obstacles_metadata, _, _ = sidebar.configure_data(
            key_photos_folder=key_photos_folder,
            key_drop_duplicates=key_drop_duplicates,
            key_annotations_cache=key_annotations_cache,
            key_annotations_file=key_annotations_file,
            key_annotations_only=key_annotations_only,
            mode=mode,
            geo_metadata=geo_metadata,
            only_folders=only_folders,
        )

        roofs_metadata = obstacles_metadata.drop_duplicates(subset="roof_id")

        chosen_folder = st.session_state[key_photos_folder]

    # +---------------+
    # | Zoom Levels   |
    # +---------------+

    st.subheader(
        f"Distribution by Zoom Level of photos in {chosen_folder or 'all folders'}"
    )
    with st.expander("What's the zoom level?"):
        st.info(
            """
        Each increase in zoom level is twice as large in both the x and y directions.
        Therefore, each higher zoom level results in a resolution four times higher than
        the preceding level. For example, at zoom level 0 the map consists of one single
        256x256 pixel tile. At zoom level 1, the map consists of four 256x256 pixel
        tiles, resulting in a pixel space from 512x512.

        Furthermore, the zoom level range depends on the world area you are looking at.
        In other words, in some parts of the world, zoom is available up to only a
        certain level.

        The graph below shows the distribution of the zoom levels in the world. The most
        represented class is European roofs with zoom level 18, which makes up 43% of
        all available roofs. Roofs from Europe make up around 95% of all roof data.
        """
        )

    zoom_levels_by_continent = (
        roofs_metadata.groupby(["continent", "zoom"])
        .size()
        .reset_index()
        .rename(
            columns={
                0: "Number of Roofs",
                "zoom": "Zoom Level",
                "continent": "Continent",
            }
        )
        .astype({"Zoom Level": "int8"})
        .assign(
            Percentage=lambda df: np.round(
                df["Number of Roofs"] / df["Number of Roofs"].sum() * 100
            )
        )
    )

    continents = zoom_levels_by_continent.Continent.unique()

    fig = (
        alt.Chart(zoom_levels_by_continent)
        .mark_bar()
        .encode(
            x="Zoom Level:N",
            y="Number of Roofs:Q",
            color="Continent:N",
            tooltip=["Number of Roofs", "Percentage"],
        )
        .interactive()
    )

    st.altair_chart(fig, use_container_width=True)

    # +-------------------------+
    # | Zoom level by continent |
    # +-------------------------+

    with st.expander("Inspect zoom levels by continent"):
        st_plot, st_selector = st.columns((3, 1))

        st_selector.selectbox(
            "Select a continent",
            options=continents,
            key="continent_detail",
            index=2 if len(continents) > 2 else 0,
        )

        fig = (
            alt.Chart(
                zoom_levels_by_continent.loc[
                    lambda df: df.Continent == st.session_state["continent_detail"]
                ]
            )
            .mark_bar()
            .encode(
                x="Zoom Level:N", y="Number of Roofs:Q", tooltip=["Number of Roofs"]
            )
        )

        st_plot.altair_chart(fig, use_container_width=True)

    # +----------------+
    # | World Map      |
    # +----------------+

    st.subheader("Zoom Level Distribution by Country")

    zoom_levels_by_country = (
        roofs_metadata.groupby(["continent", "name", "zoom"])
        .size()
        .reset_index()
        .rename(
            columns={
                0: "Number of Roofs",
                "name": "Country",
                "zoom": "Zoom Level",
                "continent": "Continent",
            }
        )
        .astype({"Zoom Level": "int8"})
        .assign(
            Percentage=lambda df: np.round(
                df["Number of Roofs"] / df["Number of Roofs"].sum() * 100
            )
        )
    )

    st_plot, st_selector = st.columns((3, 1))

    st_selector.selectbox(
        "Select a continent",
        options=continents,
        key="continent_selector",
        index=2 if len(continents) > 2 else 0,
    )

    fig = (
        alt.Chart(
            zoom_levels_by_country.loc[
                lambda df: df.Continent == st.session_state["continent_selector"]
            ]
        )
        .mark_bar()
        .encode(
            x="Zoom Level:N",
            y="Number of Roofs:Q",
            color="Country:N",
            tooltip=["Country", "Number of Roofs", "Percentage"],
        )
    )

    st_plot.altair_chart(fig, use_container_width=True)
