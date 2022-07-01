"""
Collection of error metrics to evaluate image segmentation models.
"""

from __future__ import annotations

import cv2 as cv
import numpy as np
from numpy import ndarray

from k2_oai.utils.images import downsample_image, draw_obstacles_on_black_fill


def surface_absolute_error(
    evaluation_mask: ndarray,
    roof_coordinates: str,
    obstacle_coordinates: str,
    downsampling_factor: int,
    return_error_mask: bool = False,
):
    # TODO: add docstring
    """ """

    labelled_roof = draw_obstacles_on_black_fill(
        evaluation_mask,
        roof_coordinates=roof_coordinates,
        obstacle_coordinates=obstacle_coordinates,
    )

    downsampled_labelled_roof = downsample_image(labelled_roof, downsampling_factor)
    downsampled_eval_mask = downsample_image(evaluation_mask, downsampling_factor)

    error_mask = cv.bitwise_xor(downsampled_labelled_roof, downsampled_eval_mask)
    error = np.sum(error_mask == 255) / error_mask.size * 100

    if return_error_mask:
        return error, error_mask
    return error
