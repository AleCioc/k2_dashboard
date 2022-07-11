"""
Launches the streamlit dashboard.
"""

import os

import streamlit as st

from k2_oai.dashboard import pages
from k2_oai.dashboard.components import login


def run():
    st.set_page_config(
        page_title="K2 <-> OAI",
        page_icon=":house_buildings:",
        layout="wide",
        initial_sidebar_state="auto",
    )

    st_oauth_text_boxes = st.empty()
    if "access_token" not in st.session_state:
        if "DROPBOX_ACCESS_TOKEN" in os.environ:
            st.session_state["access_token"] = os.environ.get("DROPBOX_ACCESS_TOKEN")
        else:
            st_oauth_text_boxes, oauth_result = login.dropbox_oauth2_connect()
            if oauth_result is not None:
                st.session_state["access_token"] = oauth_result.access_token
                st.session_state["refresh_token"] = oauth_result.refresh_token

    if "access_token" in st.session_state:
        st_oauth_text_boxes.empty()

        pages_options = {
            "Welcome": pages.welcome_page,
            "Metadata Explorer": pages.metadata_explorer_page,
            "Obstacle Annotation Tool": pages.obstacle_annotator_page,
            "Obstacle Detection": pages.obstacle_detection_page,
        }

        st.sidebar.title(":gear: Settings")
        app_mode = st.sidebar.selectbox(
            "Which interface would you like to use?",
            options=pages_options.keys(),
        )

        pages_options[app_mode]()
