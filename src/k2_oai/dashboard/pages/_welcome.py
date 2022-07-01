"""
First page of the dashboard.
"""

import streamlit as st

__all__ = ["welcome_page"]


def welcome_page():
    st.sidebar.success("Choose a mode from the sidebar to get started.")

    st.markdown(
        """
    # :house: Welcome!

    This is OAI's dashboard to explore the image segmentation models designed to
    detect obstacles on satellite images. The dashboard has the following modes:

    * `Instructions` is this page.
    * `Data Explorer` is an interface to perform exploratory data analysis.
    * `Obstacle Annotation Tool` us a tool to annotate images and create new labels.
    * `Obstacle Detection` is the interface to explore the image segmentation model.

    """
    )
