"""
Login UIs, e.g. log in to Dropbox account
"""

import os

import dropbox
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

_DROPBOX_NAMESPACE_ID = os.environ.get("DROPBOX_NAMESPACE_ID")
_DROPBOX_USER_EMAIL = os.environ.get("DROPBOX_USER_MAIL")
_DROPBOX_APP_KEY = os.environ.get("APP_KEY")
_DROPBOX_APP_SECRET = os.environ.get("APP_SECRET")
if "DROPBOX_ACCESS_TOKEN" in os.environ:
    _DROPBOX_ACCESS_TOKEN = os.environ.get("DROPBOX_ACCESS_TOKEN")


def dropbox_oauth2_connect(
    dropbox_app_key=_DROPBOX_APP_KEY, dropbox_app_secret=_DROPBOX_APP_SECRET
):
    dropbox_oauth_flow = dropbox.DropboxOAuth2FlowNoRedirect(
        dropbox_app_key,
        dropbox_app_secret,
        token_access_type="offline",
    )

    authorization_url = dropbox_oauth_flow.start()

    placeholder = st.empty()

    with placeholder.container():
        st.title(":key: Dropbox Authentication")
        st.markdown(f"1. Go to [this url]({authorization_url}).")
        st.write('2. Click "Allow" (you might have to log in first).')
        st.write("3. Copy the authorization code.")
        st.write("4. Enter the authorization code here: ")
        authorization_code = st.text_input("")

    try:
        complete_oauth_flow = dropbox_oauth_flow.finish(authorization_code)
        return placeholder, complete_oauth_flow
    except:  # noqa: E722
        if authorization_code is not None and len(authorization_code):
            st.error("Invalid authorization code!")
        return placeholder, None
