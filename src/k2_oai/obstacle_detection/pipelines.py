"""
Functions that input an image and run the full set of operations defined in other
modules, e.g. crop the roof out of the satellite image and detect the obstacles on it.
"""

from __future__ import annotations

import cv2 as cv
from numpy.core.multiarray import ndarray

from k2_oai.utils import is_valid_method
from k2_oai.utils.images import draw_obstacles_on_cropped_roof, rotate_and_crop_roof

from .steps import (
    detect_obstacles,
    image_composite_binarization,
    image_erosion,
    image_filtering,
    image_simple_binarization,
)
from .utils import light_and_dark_thresholding

__all__ = [
    "manual_obstacle_detection_pipeline",
    "automatic_obstacle_detection_pipeline",
]


def manual_obstacle_detection_pipeline(
    satellite_image: ndarray,
    roof_px_coordinates: str | ndarray,
    filtering_method: str,
    filtering_sigma: int,
    binarization_method: str,
    binarization_histogram_bins: int = 64,
    binarization_tolerance: int = -1,
    erosion_kernel_size: int = -1,
    obstacle_minimum_area: int = 0,
    obstacle_boundary_type: str = "box",
    using_dashboard: bool = False,
):
    """Takes in a satellite image and returns the coordinates of the obstacles on it,
    and optionally the image where obstacles have been drawn.

    Parameters
    ----------
    satellite_image : ndarray
        The satellite image.
    roof_px_coordinates : str or ndarray
        The coordinates of the roof in the satellite image. If it's string, then is
        parsed as ndarray.
    filtering_sigma : int
        The sigma value of the filter. It must be a positive, odd integer. If -1, the
        value is inferred.
    filtering_method : { "b", "bilateral", "g", "gaussian" }
        The type of filter to apply as first step of the pipeline. Can either be
        "bilateral" (or "b") or "gaussian" (or "g").
    binarization_method : { "o", "otsu", "c", "composite"}
        Algorithm to binarize the image. Composite binarization combines two images:
        one binarized using a higher (i.e. light) threshold and another with a lower
        (i.e. dark) threshold.
    binarization_histogram_bins : int, default: 64
        Number of bins used to create the picture's greyscale histogram, used to compute
        the binarization threshold value.
    binarization_tolerance : int, default: -1.
        Required for composite binarization only, this is the value to be added and
        subtracted from the binarization threshold to obtain the light and dark
        thresholded images.
    erosion_kernel_size : int, default: -1.
        Size of the kernel used for the morphological opening. Must be a positive, odd
        number. If -1, defaults to 3 if image size is greater than 10_000, otherwise 1
    obstacle_boundary_type: { "box", "polygon" }, default: "box".
        The type of boundary for the detected obstacle. Can either be "box" or
        "polygon".
    obstacle_minimum_area : int, default: 0.
        The minimum area to consider a blob a valid obstacle. Defaults to 0.
        If -1, it will default to the largest component of the image
        (height or width), divided by 10 and then rounded up.
    using_dashboard : bool, default: False
        Returns multiple outputs. Used only inside the dashboard.

    Returns
    -------
    # TODO: write docs for return values
    """

    is_valid_method(binarization_method, ["s", "simple", "o", "otsu", "c", "composite"])

    cropped_roof: ndarray = rotate_and_crop_roof(
        satellite_image=satellite_image, roof_coordinates=roof_px_coordinates
    )

    filtered_roof: ndarray = image_filtering(
        roof_image=cropped_roof,
        filtering_method=filtering_method,
        filtering_sigma=filtering_sigma,
    )

    if binarization_method in ["o", "otsu", "s", "simple"]:
        binarized_roof: ndarray = image_simple_binarization(roof_image=filtered_roof)
    else:
        binarized_roof: ndarray = image_composite_binarization(
            roof_image=filtered_roof,
            histogram_bins=binarization_histogram_bins,
            threshold_tolerance=binarization_tolerance,
        )

    blurred_roof: ndarray = image_erosion(
        roof_image=binarized_roof, kernel_size=erosion_kernel_size
    )

    if not using_dashboard:
        obstacles_coordinates, _ = detect_obstacles(
            processed_roof=blurred_roof,
            box_or_polygon=obstacle_boundary_type,
            min_area=obstacle_minimum_area,
            source_image=satellite_image,
            draw_obstacles=False,
        )
        return obstacles_coordinates

    elif using_dashboard:
        obstacles_coordinates, labelled_roof, obstacles_blobs = detect_obstacles(
            processed_roof=blurred_roof,
            box_or_polygon=obstacle_boundary_type,
            min_area=obstacle_minimum_area,
            source_image=satellite_image,
            draw_obstacles=True,
        )
        return obstacles_coordinates, obstacles_blobs, labelled_roof


def automatic_obstacle_detection_pipeline(
    satellite_image: ndarray,
    roof_px_coordinates: str | ndarray,
    filtering_method: str,
    filtering_sigma: int,
    binarization_histogram_bins: int = 64,
    binarization_tolerance: int = -1,
    erosion_kernel_size: int = -1,
    obstacle_minimum_area: int = 0,
    obstacle_boundary_type: str = "box",
    using_dashboard: bool = False,
):
    """Takes in a satellite image and returns the coordinates of the obstacles on it,
    and optionally the image where obstacles have been drawn.

    Parameters
    ----------
    satellite_image : ndarray
        The satellite image.
    roof_px_coordinates : str or ndarray
        The coordinates of the roof in the satellite image. If it's string, then is
        parsed as ndarray.
    filtering_sigma : int
        The sigma value of the filter. It must be a positive, odd integer. If -1, the
        value is inferred.
    filtering_method : { "b", "bilateral", "g", "gaussian" }
        The type of filter to apply as first step of the pipeline. Can either be
        "bilateral" (or "b") or "gaussian" (or "g").
    binarization_histogram_bins : int, default: 64
        Number of bins used to create the picture's greyscale histogram, used to compute
        the binarization threshold value.
    binarization_tolerance : int, default: -1.
        Required for composite binarization only, this is the value to be added and
        subtracted from the binarization threhsold to obtain the light and dark
        threshodled images.
    erosion_kernel_size : int, default: -1.
        Size of the kernel used for the morphological opening. Must be a positive, odd
        number. If -1, defaults to 3 if image size is greater than 10_000, otherwise 1
    obstacle_boundary_type: { "box", "polygon" }, default: "box".
        The type of boundary for the detected obstacle. Can either be "box" or
        "polygon".
    obstacle_minimum_area : int or None, default: 0.
        The minimum area to consider a blob a valid obstacle. Defaults to 0.
        If -1, it will default to the largest component of the image
        (height or width), divided by 10 and then rounded up.
    using_dashboard : bool, default: False
        Returns multiple outputs. Used only inside the dashboard.

    Returns
    -------
    # TODO: write docs for return values
    """
    cropped_roof: ndarray = rotate_and_crop_roof(satellite_image, roof_px_coordinates)

    filtered_roof: ndarray = image_filtering(
        roof_image=cropped_roof,
        filtering_method=filtering_method,
        filtering_sigma=filtering_sigma,
    )

    light_thresholded_roof, dark_thresholded_roof = light_and_dark_thresholding(
        filtered_roof, binarization_histogram_bins, binarization_tolerance
    )

    light_blurred_roof = image_erosion(dark_thresholded_roof, erosion_kernel_size)
    dark_blurred_roof = image_erosion(light_thresholded_roof, erosion_kernel_size)

    if not using_dashboard:
        light_obstacles_coords, _ = detect_obstacles(
            processed_roof=light_blurred_roof,
            box_or_polygon=obstacle_boundary_type,
            min_area=obstacle_minimum_area,
            source_image=satellite_image,
            draw_obstacles=False,
        )
        dark_obstacles_coords, _ = detect_obstacles(
            processed_roof=dark_blurred_roof,
            box_or_polygon=obstacle_boundary_type,
            min_area=obstacle_minimum_area,
            source_image=satellite_image,
            draw_obstacles=False,
        )

        return list(set(light_obstacles_coords + dark_obstacles_coords))

    elif using_dashboard:
        light_obstacles_coords, light_blobs = detect_obstacles(
            processed_roof=light_blurred_roof,
            box_or_polygon=obstacle_boundary_type,
            min_area=obstacle_minimum_area,
            source_image=satellite_image,
            draw_obstacles=False,
        )
        dark_obstacles_coords, dark_blobs = detect_obstacles(
            processed_roof=dark_blurred_roof,
            box_or_polygon=obstacle_boundary_type,
            min_area=obstacle_minimum_area,
            source_image=satellite_image,
            draw_obstacles=False,
        )

        obstacles_coordinates = list(
            set(light_obstacles_coords + dark_obstacles_coords)
        )

        obstacles_blobs = cv.bitwise_or(light_blobs, dark_blobs)

        labelled_roof = draw_obstacles_on_cropped_roof(
            cropped_roof=cropped_roof,
            roof_coordinates=roof_px_coordinates,
            obstacles_coordinates=obstacles_coordinates,
        )

        return obstacles_coordinates, obstacles_blobs, labelled_roof
