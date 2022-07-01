"""
String paths to the core dropbox data folders. Paths are collected here to ensure
that they are consistent across all the functions in the package.
"""

__all__ = [
    "DROPBOX_DESIGN_MATRIX_PATH",
    "DROPBOX_EXTERNAL_DATA_PATH",
    "DROPBOX_HYPERPARAM_ANNOTATIONS_PATH",
    "DROPBOX_LABEL_ANNOTATIONS_PATH",
    "DROPBOX_PHOTOS_METADATA_PATH",
    "DROPBOX_RAW_PHOTOS_ROOT",
]

DROPBOX_DESIGN_MATRIX_PATH = "/k2/design_matrix"
DROPBOX_EXTERNAL_DATA_PATH = "/k2/external_data"
DROPBOX_HYPERPARAM_ANNOTATIONS_PATH = "/k2/metadata/hyperparameters"
DROPBOX_LABEL_ANNOTATIONS_PATH = "/k2/metadata/label_annotations"
DROPBOX_PHOTOS_METADATA_PATH = "/k2/metadata/transformed_data"
DROPBOX_RAW_PHOTOS_ROOT = "/k2/raw_photos"
