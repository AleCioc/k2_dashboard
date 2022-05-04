"""
This module contains two auxiliary functions for the obstacle detection module.
For example, draws the boundaries of roofs and obstacles on a given image,
rotates and crops the roofs, or applies padding to an image.
"""

from __future__ import annotations

import cv2 as cv
import numpy as np
from numpy.core.multiarray import ndarray

from k2_oai.utils._parsers import parse_str_as_coordinates


def read_image_from_bytestring(
    bytestring_image: bytes,
    as_greyscale: bool = True,
) -> ndarray:
    """Reads the bytestring and returns it as a numpy array.
    This passage is necessary because the API sends a file that is transferred
    to the server as a bytestring.

    Parameters
    ----------
    bytestring_image : bytes
        The bytestring of the image.
    as_greyscale : bool
        If True, the image is converted to greyscale. The default is True.

    Returns
    -------
    ndarray
        The image as a numpy array.
    """
    image_array: ndarray = np.fromstring(bytestring_image, np.uint8)

    if as_greyscale:
        return cv.imdecode(image_array, cv.IMREAD_GRAYSCALE)
    return cv.imdecode(image_array, cv.IMREAD_COLOR)


def pad_image(
    image: ndarray,
    padding_percentage: int | None = None,
) -> tuple[ndarray, tuple[int, int]]:
    """Applies padding to an image (e.g. to remove borders).

    Parameters
    ----------
    image : ndarray
        The image to be padded.
    padding_percentage : int or None (default: None)
        The size of the padding, as integer between 1 and 100.
        If None, no padding is applied.

    Returns
    -------
    ndarray, tuple[int, int]
        The padded image and the margins for the padding.
    """
    if padding_percentage not in range(0, 101):
        raise ValueError("Parameter `padding` must range between 1 and 100.")
    elif padding_percentage is None or padding_percentage == 0:
        margin_h, margin_w = 0, 0
    else:
        margin_h, margin_w = (
            int(image.shape[n] / padding_percentage) for n in range(2)
        )

    padded_image: ndarray = image[
        margin_h : image.shape[0] - margin_h,
        margin_w : image.shape[1] - margin_w,
    ]

    return padded_image, (margin_h, margin_w)


def draw_boundaries(
    input_image: ndarray,
    roof_coordinates: str | ndarray,
    obstacle_coordinates: str | ndarray | list[str] | list[ndarray] | None = None,
) -> ndarray:
    """Draws roof and obstacle labels on the input image from their coordinates.

    Parameters
    ----------
    input_image : ndarray
        Input image.
    roof_coordinates : str or ndarray
        Roof coordinates, either as string or list of lists of integers.
    obstacle_coordinates : str or ndarray or None (default: None)
        Obstacle coordinates. Can be None if there are no obstacles. Defaults to None.

    Returns
    -------
    ndarray
        Image with labels drawn.
    """

    points: np.array = parse_str_as_coordinates(roof_coordinates).reshape((-1, 1, 2))
    result: np.ndarray = cv.polylines(input_image, [points], True, (0, 0, 255), 2)

    for obst in obstacle_coordinates:
        points: np.array = parse_str_as_coordinates(obst).reshape((-1, 1, 2))
        result: np.array = cv.polylines(result, [points], True, (255, 0, 0), 2)

    return result


def _compute_rotation_matrix(coordinates):
    diff = np.subtract(coordinates[1], coordinates[0])
    theta = np.mod(np.arctan2(diff[0], diff[1]), np.pi / 2)
    center = coordinates[0]

    return cv.getRotationMatrix2D(
        (int(center[0]), int(center[1])), -theta * 180 / np.pi, 1
    )


def rotate_and_crop_roof(input_image: ndarray, roof_coordinates: str) -> ndarray:
    """Rotates the input image to make the roof sides parallel to the image,
    then crops it.

    Parameters
    ----------
    input_image : ndarray
        The input image.
    roof_coordinates : str
        Roof coordinates: if string, it is parsed as a string of coordinates
        (i.e. a list of list of integers: [[x1, y1], [x2, y2], ...]).

    Returns
    -------
    ndarray
        The rotated and cropped roof.
    """
    coord = parse_str_as_coordinates(
        roof_coordinates, dtype="int32", sort_coordinates=True
    )

    if len(input_image.shape) < 3:
        im_alpha = cv.cvtColor(input_image, cv.COLOR_GRAY2BGRA)
    else:
        im_alpha = cv.cvtColor(input_image, cv.COLOR_BGR2BGRA)

    # rectangular roofs
    if len(coord) == 4:
        rotation_matrix = _compute_rotation_matrix(coord)

        im_affine = cv.warpAffine(
            im_alpha,
            rotation_matrix,
            im_alpha.shape[0:2],
            cv.INTER_LINEAR,
            cv.BORDER_CONSTANT,
        )

        diff = np.subtract(coord[1], coord[0])

        if diff[1] > 0:
            dist_y = np.linalg.norm(coord[1] - coord[0]).astype(int)
            dist_x = np.linalg.norm(coord[2] - coord[0]).astype(int)
        else:
            dist_y = np.linalg.norm(coord[2] - coord[0]).astype(int)
            dist_x = np.linalg.norm(coord[1] - coord[0]).astype(int)

        im_result = im_affine[
            coord[0][1] : coord[0][1] + dist_y, coord[0][0] : coord[0][0] + dist_x, :
        ]

    # polygonal roofs
    else:
        coord = parse_str_as_coordinates(
            roof_coordinates, dtype="int32", sort_coordinates=False
        )
        mask = np.zeros(input_image.shape[0:2], dtype="uint8")

        pts = np.array(coord, np.int32).reshape((-1, 1, 2))

        cv.fillConvexPoly(mask, pts, (255, 255, 255))

        im_alpha[:, :, 3] = mask

        bot_right = np.max(pts, axis=0)
        top_left = np.min(pts, axis=0)

        im_result = im_alpha[
            top_left[0][1] : bot_right[0][1], top_left[0][0] : bot_right[0][0], :
        ]

    return im_result