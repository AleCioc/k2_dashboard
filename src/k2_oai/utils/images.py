"""
This module contains two auxiliary functions for the obstacle detection module.
For example, draws the boundaries of roofs and obstacles on a given image,
rotates and crops the roofs, or applies padding to an image.
"""

from __future__ import annotations

import cv2 as cv
import numpy as np
from numpy import ndarray

from k2_oai.utils._parsers import parse_str_as_coordinates

__all__ = [
    "read_image_from_bytestring",
    "pad_image",
    "downsample_image",
    "draw_obstacles_on_cropped_roof",
    "draw_obstacles_on_black_fill",
    "draw_roofs_and_obstacles_on_photo",
    "rotate_and_crop_roof",
]


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
    padding_percentage: int = 0,
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
        raise ValueError("Parameter `padding` must range between 0 and 100.")
    elif padding_percentage == 0:
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


def downsample_image(image, downsampling_factor):
    new_shape = np.divide(image.shape, downsampling_factor).astype(int)
    output_image = np.zeros(new_shape, dtype="uint8")

    for i in range(output_image.shape[0]):
        for j in range(output_image.shape[1]):
            output_image[i, j] = 255 * np.any(
                image[
                    i * downsampling_factor : i * downsampling_factor
                    + downsampling_factor,
                    j * downsampling_factor : j * downsampling_factor
                    + downsampling_factor,
                ]
            )

    return output_image


def draw_roofs_and_obstacles_on_photo(
    photo: ndarray,
    roof_coordinates: str,
    obstacle_coordinates: str | None,
):
    """Draws roof and obstacle labels on the input image from their coordinates.

    Parameters
    ----------
    photo : ndarray
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
    photo_copy = photo.copy()

    points: ndarray = parse_str_as_coordinates(roof_coordinates).reshape((-1, 1, 2))
    labelled_image: ndarray = cv.polylines(photo_copy, [points], True, (0, 0, 255), 2)

    if obstacle_coordinates is None:
        return labelled_image

    for obst in obstacle_coordinates:
        points: ndarray = parse_str_as_coordinates(obst).reshape((-1, 1, 2))
        labelled_image: ndarray = cv.polylines(
            photo_copy, [points], True, (255, 0, 0), 2
        )

    return labelled_image


def draw_obstacles_on_cropped_roof(
    cropped_roof: ndarray,
    roof_coordinates: str,
    obstacle_coordinates: str,
) -> ndarray:
    """Draws roof and obstacle labels on the input image from their coordinates.

    Parameters
    ----------
    cropped_roof : ndarray
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
    labelled_roof = cropped_roof.copy()

    if obstacle_coordinates is None:
        return labelled_roof

    roof_coords = parse_str_as_coordinates(
        roof_coordinates, dtype="int32", sort_coordinates=True
    )

    # rectangular roof
    if len(roof_coords) == 4:
        rotation_matrix = _compute_rotation_matrix(roof_coords)
        center = roof_coords[0]

        for obst in obstacle_coordinates:
            points_obs: np.array = parse_str_as_coordinates(obst)

            obst_vertex = len(points_obs)

            pts1 = np.hstack((points_obs, np.ones((obst_vertex, 1))))

            pts_rotated = np.matmul(rotation_matrix, np.transpose(pts1))

            offset = np.transpose(np.tile(center, (obst_vertex, 1)))

            pts_new = np.subtract(pts_rotated, offset).astype(int)
            pts_new = np.transpose(pts_new)

            cv.polylines(
                labelled_roof, [pts_new], True, (255, 0, 0, 255), 1, lineType=cv.LINE_4
            )

    # polygonal roof
    else:
        for obst in obstacle_coordinates:
            points: np.array = parse_str_as_coordinates(obst).reshape((-1, 1, 2))

            top_left = np.min(roof_coords, axis=0)

            points_list = []
            for pts in points:
                points_list.append(np.subtract(pts, top_left))

            points_offset = np.array(points_list).reshape((-1, 1, 2))

            cv.polylines(
                labelled_roof,
                [points_offset],
                True,
                (255, 0, 0, 255),
                1,
                lineType=cv.LINE_4,
            )

    return labelled_roof


def draw_obstacles_on_black_fill(
    cropped_roof: ndarray,
    roof_coordinates: str,
    obstacle_coordinates: str,
) -> ndarray:
    """Draws roof and obstacle labels on the input image from their coordinates.

    Parameters
    ----------
    cropped_roof : ndarray
        Input image.
    roof_coordinates : str or ndarray
        Roof coordinates, either as string or list of lists of integers.
    obstacle_coordinates : str or ndarray or None (default: None)
        Obstacle coordinates. Can be None if there are no obstacles. Defaults to None.

    Returns
    -------
    ndarray
        A black mask with the labels drawn
    """

    black_background = np.zeros(cropped_roof.shape, np.uint8)

    roof_coords = parse_str_as_coordinates(
        roof_coordinates, dtype="int32", sort_coordinates=True
    )

    # rectangular roof
    if len(roof_coords) == 4:
        rotation_matrix = _compute_rotation_matrix(roof_coords)
        center = roof_coords[0]

        for obst in obstacle_coordinates:
            points_obs: np.array = parse_str_as_coordinates(obst)

            obst_vertex = len(points_obs)

            pts1 = np.hstack((points_obs, np.ones((obst_vertex, 1))))

            pts_rotated = np.matmul(rotation_matrix, np.transpose(pts1))

            offset = np.transpose(np.tile(center, (obst_vertex, 1)))

            pts_new = np.subtract(pts_rotated, offset).astype(int)
            pts_new = np.transpose(pts_new)

            cv.fillConvexPoly(black_background, pts_new, (255, 255, 255, 255), 1)

    # polygonal roof
    else:
        for obst in obstacle_coordinates:
            points: np.array = parse_str_as_coordinates(obst).reshape((-1, 1, 2))

            top_left = np.min(roof_coords, axis=0)

            points_list = []
            for pts in points:
                points_list.append(np.subtract(pts, top_left))

            points_offset = np.array(points_list).reshape((-1, 1, 2))

            cv.fillConvexPoly(black_background, points_offset, (255, 255, 255, 255), 1)

    return black_background


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
    # dont'change dtype
    roof_coords = parse_str_as_coordinates(
        roof_coordinates, dtype="int32", sort_coordinates=True
    )

    if len(input_image.shape) < 3:
        image_bgra = cv.cvtColor(input_image, cv.COLOR_GRAY2BGRA)
    else:
        image_bgra = cv.cvtColor(input_image, cv.COLOR_BGR2BGRA)

    # rectangular roofs
    if len(roof_coords) == 4:
        rotation_matrix = _compute_rotation_matrix(roof_coords)

        rotated_image = cv.warpAffine(
            image_bgra,
            rotation_matrix,
            (image_bgra.shape[0] * 2, image_bgra.shape[1] * 2),
            cv.INTER_LINEAR,
            cv.BORDER_CONSTANT,
        )

        diff = np.subtract(roof_coords[1], roof_coords[0])

        if diff[1] > 0:
            dist_y = np.linalg.norm(roof_coords[1] - roof_coords[0]).astype(int)
            dist_x = np.linalg.norm(roof_coords[2] - roof_coords[0]).astype(int)
        else:
            dist_y = np.linalg.norm(roof_coords[2] - roof_coords[0]).astype(int)
            dist_x = np.linalg.norm(roof_coords[1] - roof_coords[0]).astype(int)

        return rotated_image[
            roof_coords[0][1] : roof_coords[0][1] + dist_y,
            roof_coords[0][0] : roof_coords[0][0] + dist_x,
            :,
        ]

    # polygonal roofs
    else:
        roof_coords = parse_str_as_coordinates(
            roof_coordinates, dtype="int32", sort_coordinates=False
        )
        mask = np.zeros(input_image.shape[0:2], dtype="uint8")

        pts = np.array(roof_coords, np.int32).reshape((-1, 1, 2))

        cv.fillConvexPoly(mask, pts, (255, 255, 255))

        image_bgra[:, :, 3] = mask

        bot_right = np.max(pts, axis=0)
        top_left = np.min(pts, axis=0)

        return image_bgra[
            top_left[0][1] : bot_right[0][1], top_left[0][0] : bot_right[0][0], :
        ]
