"""
Function(s) to load data from the hard disk to the Dropbox "database".
"""

import os

from k2_oai.dropbox import dropbox_upload_file_to

_HARD_DISK_PATH = "/Volumes/Elements/Big_Size_Images_2"


def upload_hard_disk_data(dropbox_app, hard_disk_path=None):
    hd_path = hard_disk_path or _HARD_DISK_PATH
    photo_list = os.listdir(hd_path)
    for photo in photo_list[5000:10000]:
        if "Thumbs" not in photo:
            print(photo)
            dropbox_upload_file_to(
                dropbox_app,
                os.path.join(hd_path, photo),
                f"/k2/raw_photos/large_photos-5K_10K/{photo}",
            )
