"""
Functions to preprocess roof images and perform obstacle detection.
"""

from __future__ import annotations

import cv2 as cv
import numpy as np
from numpy import ndarray

from k2_oai.obstacle_detection.utils import (
    compute_otsu_threshold,
    get_bounding_boxes,
    get_bounding_polygon,
    light_and_dark_thresholding,
)
from k2_oai.utils import is_positive_odd_integer, is_valid_method

__all__: list[str] = [
    "image_filtering",
    "image_simple_binarization",
    "image_composite_binarization",
    "image_erosion",
    "detect_obstacles",
]


def image_filtering(
    roof_image: ndarray, filtering_method: str, filtering_sigma: int
) -> ndarray:
    """Applies a filter on the input image, which can be either greyscale, BGR or BGRA.

    Parameters
    ----------
    roof_image : ndarray
        The image that the filter will be applied to. Is a greyscale image.
    filtering_method : { "g", "gaussian", "b", "bilateral" }
        The method used to apply the filter. It must be either 'bilateral' (or 'b')
        or 'gaussian' (or 'g').
    filtering_sigma : int
        The sigma value of the filter. It must be a positive, odd integer. If -1, the
        value is inferred.

    Returns
    -------
    ndarray
        The filtered image, with 4 channels (BGRA).
    """

    def bilateral_filter(_image, _sigma):
        return cv.bilateralFilter(src=_image, d=9, sigmaColor=_sigma, sigmaSpace=_sigma)

    is_valid_method(filtering_method, ["b", "g", "bilateral", "gaussian"])

    if filtering_sigma < -1 or filtering_sigma == 0 or filtering_sigma % 2 == 0:
        raise ValueError("`filtering_sigma` must be -1 or a positive odd integer")
    if filtering_sigma == -1:
        sigma_x, sigma_y = np.floor(np.divide(roof_image.shape, 30)).astype(int)
        if sigma_x % 2 == 0:
            sigma_x += 1

        if sigma_y % 2 == 0:
            sigma_y += 1
    else:
        sigma_x, sigma_y = filtering_sigma, filtering_sigma

    if filtering_method == "b" or filtering_method == "bilateral":

        if len(roof_image.shape) > 2:
            if roof_image.shape[2] > 3:  # bgra image
                bgr_roof = cv.cvtColor(roof_image, cv.COLOR_BGRA2BGR)
                roof_image[:, :, 0:3] = bilateral_filter(bgr_roof, sigma_x)
                return roof_image
            elif roof_image.shape[2] <= 3:  # bgr image
                filtered_roof = bilateral_filter(roof_image, sigma_x)
                return cv.cvtColor(filtered_roof, cv.COLOR_BGR2BGRA)

        # grayscale image
        filtered_roof = bilateral_filter(roof_image, sigma_x)
        return cv.cvtColor(filtered_roof, cv.COLOR_GRAY2BGRA)

    else:
        return cv.GaussianBlur(roof_image, (sigma_x, sigma_y), 0)


def image_simple_binarization(roof_image: ndarray) -> ndarray:
    """Binarizes the image using the Otsu method.

    Parameters
    ----------
    roof_image : ndarray
        A BGRA image.

    Return
    ------
    ndarray
        The thresholded image.
    """

    zeros_in_mask = int(np.sum(roof_image[:, :, 3] == 0))

    otsu_threshold, _ = compute_otsu_threshold(roof_image, zeros_in_mask)

    _threshold, binarized_roof = cv.threshold(
        roof_image, otsu_threshold, 255, cv.THRESH_BINARY
    )

    if np.sum(binarized_roof == 255) > np.sum(binarized_roof == 0) - zeros_in_mask:
        binarized_roof = cv.bitwise_not(binarized_roof)
        binarized_roof = cv.bitwise_and(binarized_roof, roof_image[:, :, 3])

    return binarized_roof


def image_composite_binarization(
    roof_image: ndarray, histogram_bins: int = 64, threshold_tolerance: int = -1
) -> ndarray:
    """Binarizes an image using the "composite" method. The procedure takes the image
    and applies simple binarization twice: the first time with threshold -= tolerance to
    highlight "light" obstacles, and the second one with threshold += tolerance to
    highlight the "dark" ones.

    Parameters
    ----------
    roof_image : ndarray
        A BGRA image.
    histogram_bins : int
        Number of bins used to compute the image's greyscale histogram. This histogram
        is used to compute the binarization threshold.
    threshold_tolerance : int
        The value to add and subtract from the binarization threshold to obtain the
        light and dark thresholded images. If -1, is inferred.

    Returns
    -------
    ndarray
        The thresholded image

    """
    zeros_in_mask = int(np.sum(roof_image[:, :, 3] == 0))

    light_thresholded_roof, dark_thresholded_roof = light_and_dark_thresholding(
        source_image=roof_image,
        binarization_histogram_bins=histogram_bins,
        binarization_tolerance=threshold_tolerance,
        zeros_in_mask=zeros_in_mask,
    )

    binarized_roof = cv.bitwise_or(light_thresholded_roof, dark_thresholded_roof)
    binarized_roof = cv.bitwise_and(binarized_roof[:, :, 0], roof_image[:, :, 3])

    if np.sum(binarized_roof == 255) > np.sum(binarized_roof == 0) - zeros_in_mask:
        binarized_roof = cv.bitwise_not(binarized_roof)
        binarized_roof = cv.bitwise_and(binarized_roof, roof_image[:, :, 3])

    return binarized_roof


def image_erosion(roof_image: ndarray, kernel_size: int = -1) -> ndarray:
    """Applies an opening[1] (i.e., erosion followed by dilation) on the input image,
    to remove noise.

    Parameters
    ----------
    roof_image : ndarray
        The input image to which the opening will be applied.
    kernel_size : int, default: -1
        Size of the kernel used for the morphological opening.
        Must be a positive, odd number. If -1, defaults to 3 if image size is greater
        than 10_000, otherwise to 1.

    Returns
    -------
    ndarray
        The image.

    References
    ----------
    .. [1]
        https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    """
    if kernel_size == -1:
        kernel_size: int = 1 if roof_image.size < 10_000 else 3
    else:
        is_positive_odd_integer(kernel_size)

    kernel: ndarray = np.ones((kernel_size, kernel_size), np.uint8)
    roof_open_morph = cv.morphologyEx(roof_image, cv.MORPH_OPEN, kernel)
    return cv.morphologyEx(roof_open_morph, cv.MORPH_CLOSE, kernel)


def detect_obstacles(
    processed_roof: ndarray,
    box_or_polygon: str = "box",
    min_area: int = -1,
    source_image: ndarray | None = None,
    draw_obstacles: bool = False,
):
    """Finds the connected components in a binary image and assigns a label to them.
    First, crops the border of the image (depending on the cut_border parameter), then
    applies a morphological opening. After the connected components' analysis, the
    algorithm rejects the components having an area less than the area_min parameter.

    Parameters
    ----------
    processed_roof : ndarray
        Input image.
    box_or_polygon : {"box", "polygon"}, default: 'box'
        String indicating whether to using bounding boxes or bounding polygon.
    min_area : int, default: 0
        Minimum area of the connected components to be kept. Defaults to zero.
        If set to "auto", it will default to the largest component of the image
        (height or width), divided by 10 and then rounded up.
    source_image : ndarray or None, default: None
        The image from where the roof was cropped. Used only if the param draw_obstacles
        is True.
    draw_obstacles : bool, default: False
        Whether to return a copy of the source image with obstacles bounding boxes/
        polygons drawn on it.

    Returns
    -------
    # TODO: write docs for return values
    """

    if min_area == -1:
        min_area: int = int(np.max(processed_roof.shape) / 20)
    elif min_area < -1:
        raise ValueError("`min_area` must be greater than -1.")

    is_valid_method(box_or_polygon, ["box", "polygon"])

    _, obstacles_blobs, blobs_stats, blobs_centroids = cv.connectedComponentsWithStats(
        processed_roof, connectivity=8
    )

    if draw_obstacles and source_image is None:
        raise ValueError("`source_image` cannot be None when `draw_obstacles` is True")
    elif draw_obstacles:
        if box_or_polygon == "box":
            obstacles_coordinates, labelled_roof = get_bounding_boxes(
                blobs_stats, min_area, source_image, draw_obstacles=True
            )
        else:
            obstacles_coordinates, labelled_roof = get_bounding_polygon(
                blobs_stats,
                obstacles_blobs,
                min_area,
                source_image,
                draw_obstacles=True,
            )

        return obstacles_coordinates, labelled_roof, obstacles_blobs
    elif not draw_obstacles:
        if box_or_polygon == "box":
            obstacles_coordinates = get_bounding_boxes(
                blobs_stats, min_area, draw_obstacles=False
            )
        else:
            obstacles_coordinates = get_bounding_polygon(
                blobs_stats,
                obstacles_blobs,
                min_area,
                draw_obstacles=False,
            )

        return obstacles_coordinates, obstacles_blobs


def experimental_canny_edge_detection(source_image: ndarray):
    edges = cv.Canny(source_image, threshold1=70, threshold2=100)

    return edges


def experimental_hough_transform(source_image: ndarray):
    hough_transformer = cv.createGeneralizedHoughGuil()

    # image_mask = cv.bitwise_and(source_image[:, :, 0], source_image[:, :, 3])
    #
    # templ_shape = np.divide(source_image.shape, 3).astype(int)
    # im_masked, background = (source_image, 64)

    template = np.full((50, 80), 0, dtype=np.uint8)
    template[0, :] = 255
    template[-1, :] = 255
    template[:, 0] = 255
    template[:, -1] = 255

    hough_transformer.setTemplate(template)
    hough_transformer.setPosThresh(80)
    hough_transformer.setScaleThresh(7000)
    hough_transformer.setAngleThresh(300000)
    hough_transformer.setMaxScale(2.0)

    edges = cv.Canny(source_image, threshold1=70, threshold2=100)
    positions, votes = hough_transformer.detect(edges)
    print(positions)

    output_image = source_image.copy()
    if positions is not None:
        positions_list = positions[0]
        i = 0
        for x, y, scale, orientation in positions_list:
            half_height = template.shape[0] / 2.0 * scale
            half_width = template.shape[1] / 2.0 * scale
            p1 = (int(x - half_width), int(y - half_height))
            p2 = (int(x + half_width), int(y + half_height))
            print(
                "x = {}, y = {}, scale = {}, orientation = {}, p1 = {}, p2 = {}".format(
                    x, y, scale, orientation, p1, p2
                )
            )
            print(
                "pos_v = {}, scale_v = {}, angle_v = {}".format(
                    votes[0, i, 0], votes[0, i, 1], votes[0, i, 2]
                )
            )
            cv.rectangle(output_image, p1, p2, (0, 255, 0, 255))
            i = i + 1

    return output_image
