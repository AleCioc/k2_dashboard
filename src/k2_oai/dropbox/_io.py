"""
Create primitives to create dropbox app instances to connect to dropbox and upload or
download files.
"""

import os

import dropbox
import pandas as pd
from dotenv import load_dotenv
from dropbox.exceptions import ApiError, AuthError
from dropbox.files import WriteMode

load_dotenv()

_DROBOX_NAMESPACE_ID = os.environ.get("DROPBOX_NAMESPACE_ID")
_DROPBOX_USER_EMAIL = os.environ.get("DROPBOX_USER_MAIL")
_DROPBOX_APP_KEY = os.environ.get("APP_KEY")
_DROPBOX_APP_SECRET = os.environ.get("APP_SECRET")
if "DROPBOX_ACCESS_TOKEN" in os.environ:
    _DROPBOX_ACCESS_TOKEN = os.environ.get("DROPBOX_ACCESS_TOKEN")

__all__ = [
    "dropbox_oauth2_connect",
    "dropbox_connect",
    "dropbox_connect_access_token_only",
    "dropbox_listdir",
    "dropbox_upload_file_to",
    "dropbox_download_from",
]


def dropbox_oauth2_connect(
    dropbox_app_key=_DROPBOX_APP_KEY, dropbox_app_secret=_DROPBOX_APP_SECRET
):

    dropbox_oauth2_flow = dropbox.DropboxOAuth2FlowNoRedirect(
        dropbox_app_key, dropbox_app_secret
    )

    authorization_url = dropbox_oauth2_flow.start()

    print(
        f" 1. Go to: {authorization_url}\n",
        '2. Click "Allow" (you might have to log in first)\n',
        "3. Copy the authorization code\n",
        "4. Enter the authorization code here:",
    )
    authorization_code = input()

    try:
        return dropbox_oauth2_flow.finish(authorization_code)
    except:  # noqa: E722
        print("Invalid authorization code!")
        return None


def _get_team_member_id(dropbox_team_app, dropbox_user_email):
    return (
        dropbox_team_app.team_members_get_info(
            [dropbox.team.UserSelectorArg("email", dropbox_user_email)]
        )[0]
        .get_member_info()
        .profile.team_member_id
    )


def dropbox_connect_access_token_only(
    dropbox_access_token,
    dropbox_namespace_id=_DROBOX_NAMESPACE_ID,
    dropbox_user_email=_DROPBOX_USER_EMAIL,
):
    try:
        dbx_team = dropbox.DropboxTeam(dropbox_access_token)

        dbx_team_member_id = _get_team_member_id(dbx_team, dropbox_user_email)

        return dbx_team.with_path_root(
            dropbox.common.PathRoot.namespace_id(dropbox_namespace_id)
        ).as_user(team_member_id=dbx_team_member_id)

    except AuthError as e:
        print("Error connecting to Dropbox with access token: " + str(e))
        return None


def dropbox_connect(
    dropbox_access_token,
    dropbox_refresh_token,
    dropbox_app_key=_DROPBOX_APP_KEY,
    dropbox_app_secret=_DROPBOX_APP_SECRET,
    dropbox_user_email=_DROPBOX_USER_EMAIL,
    dropbox_namespace_id=_DROBOX_NAMESPACE_ID,
):
    """Create a connection to Dropbox."""

    try:

        dbx_team = dropbox.DropboxTeam(
            oauth2_access_token=dropbox_access_token,
            oauth2_refresh_token=dropbox_refresh_token,
            app_key=dropbox_app_key,
            app_secret=dropbox_app_secret,
        )

        team_member_id = _get_team_member_id(dbx_team, dropbox_user_email)

        return dbx_team.with_path_root(
            dropbox.common.PathRoot.namespace_id(dropbox_namespace_id)
        ).as_user(team_member_id=team_member_id)

    except AuthError as e:
        print("Error connecting to Dropbox with access token: " + str(e))
        return None


def _parse_dropbox_folder_content(folder_content):
    files_list = []

    files = folder_content.entries

    for file in files:
        if isinstance(file, dropbox.files.FolderMetadata):
            metadata = {
                "item_type": "folder",
                "item_dropbox_id": file.id,
                "item_name": file.name,
                "item_abs_path": file.path_display,
                # 'client_modified': file.client_modified,
                # 'server_modified': file.server_modified
            }
            files_list.append(metadata)
        elif isinstance(file, dropbox.files.FileMetadata):
            metadata = {
                "item_type": "file",
                "item_dropbox_id": file.id,
                "item_name": file.name,
                "item_abs_path": file.path_display,
                # 'client_modified': file.client_modified,
                # 'server_modified': file.server_modified
            }
            files_list.append(metadata)

    return files_list


def dropbox_listdir(dropbox_path, dropbox_app):
    """Return a Pandas dataframe of files in a given Dropbox folder path in the Apps
    directory.
    """

    files_list = []

    dbx_folder_contents = dropbox_app.files_list_folder(dropbox_path)
    files = dbx_folder_contents.entries

    while dbx_folder_contents.has_more or len(files):
        files = dbx_folder_contents.entries
        files_list += _parse_dropbox_folder_content(dbx_folder_contents)
        dbx_folder_contents = dropbox_app.files_list_folder_continue(
            dbx_folder_contents.cursor
        )

    return pd.DataFrame.from_records(files_list)


def dropbox_upload_file_to(
    dropbox_app, upload_from, save_to, remove_original: bool = False
):
    with open(upload_from, "rb") as f:
        try:
            dropbox_app.files_upload(f.read(), save_to, mode=WriteMode.add)
        except ApiError:
            dropbox_app.files_upload(f.read(), save_to, mode=WriteMode.overwrite)

    if remove_original:
        os.remove(upload_from)


def dropbox_download_from(dropbox_app, save_to, download_from):
    dropbox_app.files_download_to_file(save_to, download_from)
