"""
Collection of error metrics to evaluate image segmentation models.
"""
from __future__ import annotations
from pickletools import uint8

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from numpy.core.multiarray import ndarray

#from k2_oai.utils import experimental_parse_str_as_array
from k2_oai.utils._image_manipulation import _compute_rotation_matrix, draw_obstacles
from k2_oai.utils._parsers import parse_str_as_coordinates


def surface_absolute_error(
    roof_coordinates: str | ndarray,
    obstacle_coordinates: str | ndarray | list[str] | list[ndarray],
    image_evaluation: ndarray,
    downsampling: int
):
    # TODO: add docstring
    """ """

    k2_labels = draw_obstacles(image_evaluation, 
                               roof_coordinates=roof_coordinates, 
                               obstacle_coordinates=obstacle_coordinates,
                               fill=True)

    #downsample
    k2_downsample = downsample_image(k2_labels, downsampling)
    evaluation_downsample = downsample_image(image_evaluation, downsampling)

    #calcolo errore
    im_error = cv.bitwise_xor(k2_downsample, evaluation_downsample)
    error = np.sum(im_error == 255) / im_error.size * 100

    return error, im_error


def downsample_image(im_in, downsample):
    
    new_shape = np.divide(im_in.shape, downsample).astype(int)
    output_image = np.zeros(new_shape, dtype="uint8")

    for i in range(output_image.shape[0]):
        for j in range(output_image.shape[1]):
            output_image[i, j] = 255 * np.any(im_in[i*downsample : i*downsample + downsample,
                                                    j*downsample : j*downsample + downsample])

    return output_image