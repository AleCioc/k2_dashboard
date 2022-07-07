"""
Utilities necessary for the .steps module of the obstacle_detection package.
"""

from __future__ import annotations

import cv2 as cv
import numpy as np
from numpy import ndarray

__all__ = [
    "compute_otsu_threshold",
    "compute_composite_threshold",
    "compute_composite_tolerance",
    "light_and_dark_thresholding",
    "get_bounding_boxes",
    "get_bounding_polygon",
]


def compute_otsu_threshold(
    source_image: ndarray, zeros_in_mask: int | None = None, histogram_bins: int = 256
) -> tuple[int, float]:
    """Computes the threshold for filtering using the OTSU method

    Parameters
    ----------
    source_image : ndarray
        Should be applied to a cropped roof.
    zeros_in_mask : int or None (default: None)
        The number of elements equal to zero in the image_mask. If not provided, is
        computed automatically.
    histogram_bins : int (default: 256)

    Returns
    -------
    int, float
    """

    if zeros_in_mask is None:
        zeros_in_mask = np.sum(source_image[:, :, 3] == 0)

    greyscale_histogram = cv.calcHist(
        [source_image], [0], None, [histogram_bins], [0, 256]
    )

    greyscale_histogram[0] = greyscale_histogram[0] - zeros_in_mask

    normalised_histogram = greyscale_histogram.ravel() / greyscale_histogram.sum()
    his_cumsum = normalised_histogram.cumsum()

    bins = np.arange(256)
    fn_min = np.inf
    threshold = -1

    for i in range(1, 256):
        p1, p2 = np.hsplit(normalised_histogram, [i])  # probabilities
        q1, q2 = his_cumsum[i], his_cumsum[255] - his_cumsum[i]  # cum sum of classes
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
            threshold = i

    return threshold, fn_min


def compute_composite_threshold(
    source_image: ndarray,
    image_mask: ndarray | None = None,
    zeros_in_mask: int | None = None,
    histogram_bins: int = 64,
    infer_tolerance: bool = True,
) -> int | tuple[int, int]:
    """Infers the threshold for composite binarization.

    Parameters
    ----------
    source_image : ndarray
        Should be applied to a cropped roof.
    image_mask : ndarray or None (default: None)
        The number of elements equal to zero in the image_mask. If not provided, is
        computed automatically.
    zeros_in_mask : int or None (default: None)
        The number of elements equal to zero in the image_mask. If not provided, is
        computed automatically.
    histogram_bins : int (default: 64)
    infer_tolerance : bool (default: True)

    Returns
    -------
    int or tuple[int, int]

    """

    if image_mask is None:
        image_mask = cv.bitwise_and(source_image[:, :, 0], source_image[:, :, 3])

    if zeros_in_mask is None:
        zeros_in_mask = np.sum(source_image[:, :, 3] == 0)

    greyscale_histogram = cv.calcHist(
        [source_image], [0], None, [histogram_bins], [0, 256], accumulate=False
    )

    greyscale_histogram[0] = greyscale_histogram[0] - zeros_in_mask

    composite_threshold = (
        np.argmax(np.array(greyscale_histogram)) * 256 / histogram_bins
    )

    if infer_tolerance:
        tolerance = compute_composite_tolerance(
            image_mask, greyscale_histogram, histogram_bins
        )
        return composite_threshold, tolerance

    return composite_threshold


def compute_composite_tolerance(
    image_mask: ndarray, greyscale_histogram: ndarray, histogram_bins: int
) -> int:
    """Automatically infers tolerance for composite thresholding.

    Parameters
    ----------
    image_mask : ndarray
    greyscale_histogram : ndarray
    histogram_bins : int

    Returns
    -------
    int
        The tolerance value to add and subtract from the composite threshold

    """

    normalized_histogram = np.divide(greyscale_histogram, image_mask.size)

    mean = 0
    for i in range(histogram_bins):
        mean = mean + i * normalized_histogram[i]

    var = 0
    for i in range(histogram_bins):
        var = var + (i - mean) ** 2 * normalized_histogram[i]

    tolerance = int(2 * var)

    return tolerance


def light_and_dark_thresholding(
    source_image: ndarray,
    binarization_histogram_bins: int,
    binarization_tolerance: int,
    image_mask: ndarray | None = None,
    zeros_in_mask: int | None = None,
):
    """Given one image, returns a light and dark thresholded version of it.

    Parameters
    ----------
    source_image : ndarray
        Should be applied to a cropped roof.
    binarization_histogram_bins : int
        Number of bins to use to compute the histogram.
    binarization_tolerance : int
        Cutoff to add and subtract from the threshold to obtain the light and dark
        thresholded images.
    image_mask : ndarray or None (default: None)
        The number of elements equal to zero in the image_mask. If not provided, is
        computed automatically.
    zeros_in_mask : int or None (default: None)
        The number of elements equal to zero in the image_mask. If not provided, is
        computed automatically.

    Returns
    -------

    """

    if image_mask is None:
        image_mask = cv.bitwise_and(source_image[:, :, 0], source_image[:, :, 3])

    if zeros_in_mask is None:
        zeros_in_mask = np.sum(source_image[:, :, 3] == 0)

    if binarization_tolerance not in range(-1, 256):
        raise ValueError("Tolerance must be in the range [-1, 255].")
    elif binarization_tolerance == -1:
        binarization_threshold, binarization_tolerance = compute_composite_threshold(
            source_image=source_image,
            image_mask=image_mask,
            zeros_in_mask=zeros_in_mask,
            histogram_bins=binarization_histogram_bins,
            infer_tolerance=True,
        )
    else:
        binarization_threshold = compute_composite_threshold(
            source_image=source_image,
            image_mask=image_mask,
            zeros_in_mask=zeros_in_mask,
            histogram_bins=binarization_histogram_bins,
            infer_tolerance=False,
        )

    _threshold, light_thresholded_roof = cv.threshold(
        image_mask,
        binarization_threshold + binarization_tolerance,
        255,
        cv.THRESH_BINARY,
    )

    _threshold, dark_thresholded_roof = cv.threshold(
        image_mask,
        binarization_threshold - binarization_tolerance,
        255,
        cv.THRESH_BINARY_INV,
    )

    return light_thresholded_roof, dark_thresholded_roof


def get_bounding_boxes(
    blob_stats: ndarray,
    min_area: int = 0,
    source_image: ndarray | None = None,
    draw_obstacles: bool = False,
):
    """Obtain bounding boxes from blobs resulting from the connected components'
    analysis. Then, draw them on the input image.

    Parameters
    ----------
    blob_stats : ndarray
    min_area : int, default: 0
    source_image : ndarray or None, default: None
    draw_obstacles : bool, default: False
    """
    bounding_box_coordinates = []

    if draw_obstacles and source_image is None:
        raise ValueError("`source_image` cannot be None when `draw_obstacles` is True")
    elif draw_obstacles:
        labelled_image = source_image.copy()
        labelled_image = cv.cvtColor(labelled_image, cv.COLOR_BGRA2BGR)

    for i in range(1, blob_stats.shape[0]):

        # TODO: rewrite so it returns the full set of coordinates
        if blob_stats[i, cv.CC_STAT_AREA] > min_area:
            top_left_px = (
                blob_stats[i, cv.CC_STAT_LEFT],
                blob_stats[i, cv.CC_STAT_TOP],
            )
            height = blob_stats[i, cv.CC_STAT_HEIGHT]
            width = blob_stats[i, cv.CC_STAT_WIDTH]
            bottom_right_px = (top_left_px[0] + width, top_left_px[1] + height)

            bounding_box_coordinates.append((top_left_px, bottom_right_px))

            if draw_obstacles:
                labelled_image = cv.rectangle(
                    labelled_image, top_left_px, bottom_right_px, (255, 0, 0), 1
                )

    if draw_obstacles:
        return bounding_box_coordinates, labelled_image

    return bounding_box_coordinates


def get_bounding_polygon(
    blobs_stats: ndarray,
    blobs: ndarray,
    min_area: int = 0,
    source_image: ndarray | None = None,
    draw_obstacles: bool = False,
):
    """Obtain polygons delimiting obstacles, from blobs resulting from the connected
    components' analysis. Then, draw them on the input image.

    Parameters
    ----------
    blobs_stats : ndarray
    blobs : ndarray
    min_area : int, default: 0
    source_image : ndarray or None, default: None
    draw_obstacles : bool, default: False

    Returns
    -------

    """

    polygon_coordinates = []

    if draw_obstacles and source_image is None:
        raise ValueError("`source_image` cannot be None when `draw_obstacles` is True")
    elif draw_obstacles:
        labelled_image = source_image.copy()
        labelled_image = cv.cvtColor(labelled_image, cv.COLOR_BGRA2BGR)

    for i in range(1, blobs_stats.shape[0]):
        if blobs_stats[i, cv.CC_STAT_AREA] > min_area:
            obst_im = (blobs == i) * 255
            contours, hierarchy = cv.findContours(
                obst_im.astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
            )
            approximated_boundary = cv.approxPolyDP(contours[0], 5.0, True)
            polygon_coordinates.append(approximated_boundary)

            if draw_obstacles:
                cv.polylines(
                    labelled_image, [approximated_boundary], True, (255, 0, 0), 2
                )

    if draw_obstacles:
        return polygon_coordinates, labelled_image
    return polygon_coordinates
