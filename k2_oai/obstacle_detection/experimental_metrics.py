"""
Collection of error metrics to evaluate image segmentation models.
"""
from __future__ import annotations

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray

from k2_oai.utils import _compute_rotation_matrix, experimental_parse_str_as_array


def surface_absolute_error(
    input_image: ndarray,
    roof_coordinates: str | ndarray,
    obstacle_coordinates: str | ndarray | list[str] | list[ndarray],
    label_coord: ndarray,
):
    # TODO: add docstring
    """ """

    if isinstance(roof_coordinates, (str, ndarray)):
        roof_coordinates: ndarray = experimental_parse_str_as_array(
            roof_coordinates, sort_coordinates=True
        )
    else:
        raise TypeError(
            f"Expected a string or a numpy array, but got {type(roof_coordinates)}"
        )

    _, rotation_matrix = _compute_rotation_matrix(roof_coordinates)

    black_background = np.zeros(input_image.shape, np.uint8)

    if isinstance(obstacle_coordinates, list):
        obstacle_list: list[ndarray] = [
            experimental_parse_str_as_array(coordinates, sort_coordinates=True)
            for coordinates in obstacle_coordinates
        ]
    elif isinstance(obstacle_coordinates, (str, ndarray)):
        obstacle_list: list[ndarray] = [
            experimental_parse_str_as_array(obstacle_coordinates, sort_coordinates=True)
        ]
    else:
        raise TypeError(
            f"Expected a string or a numpy array, but got {type(obstacle_coordinates)}"
        )

    for obstacle in obstacle_list:
        pts = []
        for obstacle_coordinate in obstacle:
            obstacle_coordinate.append(1)
            coord_ext = np.array(obstacle_coordinate)
            pts_rot = np.matmul(rotation_matrix, coord_ext)

            pts.append(pts_rot.tolist())

        pts_ar = np.subtract(np.array(pts), np.array(roof_coordinates[0])).astype(int)
        im_draw = cv.fillConvexPoly(black_background, pts_ar, (255, 255, 255), 1)

    im_result = np.zeros(input_image.shape, np.uint8)
    for label in label_coord:
        vertex_tl = label[0]
        vertex_br = label[1]
        pts = np.array(
            [
                label[0],
                (vertex_tl[0], vertex_br[1]),
                label[1],
                (vertex_br[0], vertex_tl[1]),
            ],
            np.int32,
        )
        im_result = cv.fillConvexPoly(im_result, pts, (255, 255, 255), 1)

    im_error = cv.bitwise_xor(im_result, im_draw)

    fig, ax = plt.subplots(1, 3, figsize=(12, 12))

    ax[0].imshow(im_draw, cmap="gray")
    ax[0].set_title("Image Labelled by K2")

    ax[1].imshow(im_result, cmap="gray")
    ax[1].set_title("Algorithm Labelled Image")

    ax[2].imshow(im_error, cmap="gray")
    ax[2].set_title("Error Measurement")

    return np.sum(im_error == 255) / im_error.size * 100
