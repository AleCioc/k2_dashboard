"""
Applies a pure computer vision approach to detect obstacles in the roof images.

The methods in this module are implemented sequentially in the `pipeline` module.

The function takes in a greyscale image; then applies two steps:
1. Applying a filter - either a Gaussian blur or a bilateral filter.
2. Applying a threshold to the image.
3. Apply a morphological opening to the image, to further reduce noise.

Finally, a connected component analysis is used to perform a blob analysis.

As a reference, see the official Python tutorial from OpenCV:
https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html
"""

from __future__ import annotations

import cv2 as cv
import numpy as np
from numpy import ndarray

from k2_oai.utils import is_positive_odd_integer, is_valid_method, pad_image

__all__: list[str] = [
    # "BoundingBox",
    "filtering_step",
    "binarization_step",
    "morphological_opening_step",
    "detect_obstacles",
]

BoundingBox = tuple[tuple[int, int], tuple[int, int]]


def filtering_step(input_image: ndarray, sigma: int, method: str = "b") -> ndarray:
    """Applies a filter on the input image, which is greyscale.

    Parameters
    ----------
    input_image : ndarray
        The image that the filter will be applied to. Is a greyscale image.
    sigma : int
        The sigma value of the filter. It must be a positive, odd integer.
    method : str (default: "b")
        The method used to apply the filter. It must be either 'bilateral' (or 'b')
        or 'gaussian' (or 'g').

    Returns
    -------
    ndarray
        The filtered image, with 4 channels (BGRA).
    """

    def _bilateral_filter(image, _sigma):
        return cv.bilateralFilter(src=image, d=9, sigmaColor=_sigma, sigmaSpace=_sigma)

    is_positive_odd_integer(sigma)
    is_valid_method(method, ["b", "g", "bilateral", "gaussian"])

    if method == "b" or method == "bilateral":

        if len(input_image.shape) > 2:
            if input_image.shape[2] > 3:  # bgra image
                image_to_bgr = cv.cvtColor(input_image, cv.COLOR_BGRA2BGR)
                input_image[:, :, 0:3] = _bilateral_filter(image_to_bgr, sigma)
                return input_image
            elif input_image.shape[2] <= 3:  # bgr image
                filtered_image = _bilateral_filter(input_image, sigma)
                return cv.cvtColor(filtered_image, cv.COLOR_BGR2BGRA)

        # grayscale image
        filtered_image = _bilateral_filter(input_image, sigma)
        return cv.cvtColor(filtered_image, cv.COLOR_GRAY2BGRA)

    else:
        return cv.GaussianBlur(input_image, (0, 0), sigma)


def _compute_otsu_thresholding(input_image, zeros):
    hist = cv.calcHist([input_image], [0], None, [256], [0, 256])
    hist[0] = hist[0] - zeros  # correction for images with masks
    hist_norm = hist.ravel() / hist.sum()
    Q = hist_norm.cumsum()
    bins = np.arange(256)
    fn_min = np.inf
    thresh = -1
    for i in range(1, 256):
        p1, p2 = np.hsplit(hist_norm, [i])  # probabilities
        q1, q2 = Q[i], Q[255] - Q[i]  # cum sum of classes
        if q1 < 1.0e-6 or q2 < 1.0e-6:
            continue
        b1, b2 = np.hsplit(bins, [i])  # weights
        # finding means and variances
        m1, m2 = np.sum(p1 * b1) / q1, np.sum(p2 * b2) / q2
        v1, v2 = np.sum(((b1 - m1) ** 2) * p1) / q1, np.sum(((b2 - m2) ** 2) * p2) / q2
        # calculates the minimization function
        fn = v1 * q1 + v2 * q2
        if fn < fn_min:
            fn_min = fn
            thresh = i

    return thresh, fn_min


def binarization_step(
    input_image: ndarray,
    method: str = "s",
    adaptive_kernel_size: int | None = None,
    adaptive_constant: int = 0,
    composite_tolerance: int | None = None,
) -> ndarray:
    """Applies a threshold on the grayscale input image. Depending on the method,
    applies simple thresholding (using the Otsu method to compute the threshold) or
    adaptive thresholding - which is more robust with respect to the different lighting
    conditions of the image.

    Parameters
    ----------
    input_image : ndarray
        The input image to which thresholding will be applied.
    method : str (default: "s")
        The thresholding method to use:
        - 's' or 'simple' stands for simple thresholding;
        - 'a' or 'adaptive' stands for adaptive thresholding;
        - 'c' or 'composite' stands for composite thresholding;
    adaptive_kernel_size : int (default: None)
        Only used in adaptive thresholding. The size of the kernel for binarization.
        Must be a positive, odd number. If None (or -1), then defaults to 0,001 times
        the size of the image.
    adaptive_constant : int (default: 0)
        (Only for adaptive thresholding) The constant subtracted from the mean,
        or weighted mean. Normally is positive, but can also be negative.
    composite_tolerance : int (default)
        (Only for composite thresholding) A threshold in the range [0, 255]. If None (or
        -1), it is computed from the variance of the color histogram.

    Return
    ------
    ndarray
        The thresholded image.
    """

    is_valid_method(method, ["s", "simple", "a", "adaptive", "c", "composite"])

    masked_image = cv.bitwise_and(input_image[:, :, 0], input_image[:, :, 3])
    n_zeros_mask = np.sum(input_image[:, :, 3] == 0)

    if method == "s" or method == "simple":
        otsu_threshold, _ = _compute_otsu_thresholding(masked_image, n_zeros_mask)
        _, binarized_image = cv.threshold(
            masked_image, otsu_threshold, 255, cv.THRESH_BINARY
        )
    elif method == "a" or method == "adaptive":
        if adaptive_kernel_size is None or adaptive_kernel_size == -1:
            threshold_kernel: int = int(input_image.size / 1000)
            if threshold_kernel % 2 == 0:
                threshold_kernel += 1
        else:
            is_positive_odd_integer(adaptive_kernel_size)
            threshold_kernel: int = adaptive_kernel_size

        binarized_image = cv.adaptiveThreshold(
            masked_image,
            255,
            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY,
            threshold_kernel,
            adaptive_constant,
        )
    else:  # method == 'c', i.e. composite
        # compute the greyscale histogram just for the first channel
        greyscale_histogram = cv.calcHist(
            [input_image], [0], None, [256], [0, 256], accumulate=False
        )
        greyscale_histogram[0] = greyscale_histogram[0] - n_zeros_mask

        if composite_tolerance is None or composite_tolerance == -1:
            composite_tolerance = int(np.var(greyscale_histogram) / 15000)
            if composite_tolerance > 255:
                composite_tolerance = 255
        elif composite_tolerance not in range(0, 256):
            raise ValueError("Composite tolerance must be in the range [0, 255].")

        max_frequency = np.argmax(np.array(greyscale_histogram))

        _, im_tresh_light = cv.threshold(
            input_image,
            max_frequency + composite_tolerance,
            255,
            cv.THRESH_BINARY,
        )
        _, im_tresh_dark = cv.threshold(
            input_image,
            max_frequency - composite_tolerance,
            255,
            cv.THRESH_BINARY_INV,
        )
        binarized_image = cv.bitwise_or(im_tresh_light, im_tresh_dark)
        binarized_image = cv.bitwise_and(binarized_image[:, :, 0], input_image[:, :, 3])

    if np.sum(binarized_image == 255) > np.sum(binarized_image == 0) - n_zeros_mask:
        binarized_image = cv.bitwise_not(binarized_image)
        binarized_image = cv.bitwise_and(binarized_image, input_image[:, :, 3])

    return binarized_image


def morphological_opening_step(
    image: ndarray, kernel_size: int | None = None
) -> ndarray:
    """Applies an opening[1] (i.e., erosion followed by dilation) on the input image,
    to remove noise.

    Parameters
    ----------
    image : ndarray
        The input image to which the opening will be applied.
    kernel_size : int or None (default: None)
        Size of the kernel used for the morphological opening.
        Must be a positive, odd number. If None, defaults to 3 if image size is greater
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
    if kernel_size is None:
        kernel_size: int = 1 if image.size < 10_000 else 3
    else:
        is_positive_odd_integer(kernel_size)

    kernel: ndarray = np.ones((kernel_size, kernel_size), np.uint8)
    image_open_morphology = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
    return cv.morphologyEx(image_open_morphology, cv.MORPH_CLOSE, kernel)


def _get_bounding_boxes(input_image, stats, margins, min_area):
    bounding_box_coordinates = []

    for i in range(0, stats.shape[0]):

        # if stats[i, cv.CC_STAT_AREA] > min_area:
        top_left_px = (
            stats[i, cv.CC_STAT_LEFT] + margins[1],  # height
            stats[i, cv.CC_STAT_TOP] + margins[0],  # width
        )
        height = stats[i, cv.CC_STAT_HEIGHT]
        width = stats[i, cv.CC_STAT_WIDTH]
        bottom_right_px = (top_left_px[0] + width, top_left_px[1] + height)

        # if height < draw_image.shape[0]*0.8 and width < draw_image.shape[1]*0.8:

        bounding_box_coordinates.append((top_left_px, bottom_right_px))
        input_image = cv.rectangle(
            input_image, top_left_px, bottom_right_px, (255, 0, 0), 1
        )

    return bounding_box_coordinates, input_image


def _get_bounding_polygon(blobs, background, stats, min_area):
    polygon_coordinates = []

    for i in range(0, stats.shape[0]):

        # if stats[i, cv.CC_STAT_AREA] > min_area:
        obst_im = (blobs == i) * 255
        contours, hierarchy = cv.findContours(
            obst_im.astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
        )
        approximated_boundary = cv.approxPolyDP(contours[0], 5.0, True)
        cv.polylines(background, [approximated_boundary], True, (255, 0, 0), 2)
        polygon_coordinates.append(approximated_boundary)

    return polygon_coordinates, background


def detect_obstacles(
    blurred_roof: ndarray,
    source_image: ndarray,
    box_or_polygon: str = "box",
    min_area: int | str = 0,
    padding_percentage: int = 0,
):
    """Finds the connected components in a binary image and assigns a label to them.
    First, crops the border of the image (depending on the cut_border parameter), then
    applies a morphological opening. After the connected components' analysis, the
    algorithm rejects the components having an area less than the area_min parameter.

    Parameters
    ----------
    blurred_roof : ndarray
        Input image.
    source_image : ndarray
        The image where obstacles have been labelled.
    box_or_polygon : str (default='bbox')
        String indicating whether to using bounding boxes or bounding polygon.
    min_area : int (default: 0)
        Minimum area of the connected components to be kept. Defaults to zero.
        If set to "auto", it will default to the largest component of the image
        (height or width), divided by 10 and then rounded up.
    padding_percentage : int (default=0)
        Percentage of the image shape that has to be cut from the borders.

    Returns
    -------
    tuple[ndarray, ndarray, list[tuple[str]]]
        Returns a tuple of three objects:
        - The cropped image with labels
        - The image with the bounding box of the labels,
        - The list of coordinates of the top-left and bottom-right points of the
          bounding boxes of the obstacles that have been found.
    """

    if min_area == "auto":
        min_area: int = int(np.max(blurred_roof.shape) / 10)
    elif min_area < 0:
        raise ValueError("`min_area` must be a positive integer.")

    # padding
    padded_image, padding_margins = pad_image(blurred_roof, padding_percentage)

    (
        total_labels,
        obstacles_blobs,
        stats,
        blobs_centroids,
    ) = cv.connectedComponentsWithStats(padded_image, connectivity=8)

    background_image = cv.cvtColor(source_image, cv.COLOR_BGRA2BGR)

    is_valid_method(box_or_polygon, ["box", "polygon"])
    if box_or_polygon == "box":
        bounding_boxes_coordinates, background_with_bboxes = _get_bounding_boxes(
            background_image, stats, padding_margins, min_area
        )
    else:
        bounding_boxes_coordinates, background_with_bboxes = _get_bounding_polygon(
            obstacles_blobs,
            background_image,
            stats,
            min_area,
        )

    return obstacles_blobs, background_with_bboxes, bounding_boxes_coordinates
