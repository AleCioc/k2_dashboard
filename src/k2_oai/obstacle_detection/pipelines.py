"""
Functions that input an image and run the full set of operations defined in other
modules, e.g. crop the roof out of the satellite image and detect the obstacles on it.
"""

from __future__ import annotations

from numpy.core.multiarray import ndarray

from k2_oai.obstacle_detection import (
    binarization_step,
    detect_obstacles,
    filtering_step,
    morphological_opening_step,
)
from k2_oai.utils import rotate_and_crop_roof

__all__ = ["obstacle_detection_pipeline"]


def obstacle_detection_pipeline(
    satellite_image: ndarray,
    roof_px_coordinates: str | ndarray,
    filtering_sigma: int,
    filter_method: str = "b",
    binarization_method: str = "s",
    binarization_kernel: int | None = None,
    binarization_tolerance: int | None = None,
    binarization_constant: int = 0,
    morphology_kernel: int | None = None,
    obstacle_minimum_area: int | None = 0,
    obstacle_boundary_type: str = "box",
    trim_edges: bool = False,
):
    """Takes in a greyscale image of a roof and returns the same image, coloured (BGR),
    where obstacles have been tagged.

    Parameters
    ----------
    satellite_image : ndarray
        The satellite image.
    roof_px_coordinates : str or ndarray
        The coordinates of the roof in the satellite image. If it's string, then is
        parsed as ndarray.
    filtering_sigma : int
        The sigma value of the filter. It must be a positive, odd integer.
    filter_method : { "b", "bilateral", "g", "gaussian" }, default: "b".
        The type of filter to apply as first step of the pipeline. Can either be
        "bilateral" (equivalent to "b", default) or "gaussian" (equivalent to "g").
    binarization_method : { "s", "simple", "a", "adaptive", "c", "composite" },
        (default: "s").
        The method to use for binarization. Can be either:
        - "s" or "simple"
        - "a" or "adaptive"
        - "c" or "composite"
    binarization_kernel : int or None, default: None.
        Only used in adaptive thresholding. The size of the kernel for binarization.
        Must be a positive, odd number. If None, then defaults to 0,001 times the size
        of the image.
    binarization_constant : int, default: 0.
        Only used in adaptive thresholding. A constant that is subtracted from the
        (weighted) mean used in the algorithm.
    binarization_tolerance : int or None, default: None.
        Only used in composite thresholding. The tolerance used to alter the threshold,
        i.e. used to "soften" the two threshold values.
    morphology_kernel : int or None, default: None.
        Size of the kernel used for the morphological opening.
        Must be a positive, odd number. If None, defaults to 3 if image size is greater
        than 10_000, otherwise to 1
    obstacle_boundary_type: { "box", "polygon" }, default: "box".
        The type of boundary for the detected obstacle. Can either be "box" or
        "polygon".
    obstacle_minimum_area : int or None, default: 0.
        The minimum area to consider a blob a valid obstacle. Defaults to 0.
        If set to None, it will default to the largest component of the image
        (height or width), divided by 10 and then rounded up.
    trim_edges : bool, default: False
        Whether to pad the image to remove edges. Defaults to False.

    Returns
    -------
        - The array of blobs, i.e. the obstacles detected via the pipeline.
        - The source RGB image, where bounding boxes have been drawn.
        - The list of coordinates of the top-left and bottom-right points of the
          bounding boxes of the obstacles that have been found.
    """

    # crop the roof from the image using the coordinates
    cropped_roof: ndarray = rotate_and_crop_roof(satellite_image, roof_px_coordinates)

    # filtering steps
    filtered_roof: ndarray = filtering_step(
        input_image=cropped_roof, sigma=filtering_sigma, method=filter_method
    )
    binarized_roof: ndarray = binarization_step(
        filtered_roof,
        method=binarization_method,
        adaptive_kernel_size=binarization_kernel,
        adaptive_constant=binarization_constant,
        composite_tolerance=binarization_tolerance,
    )
    blurred_roof: ndarray = morphological_opening_step(
        binarized_roof,
        kernel_size=morphology_kernel,
    )

    # edges for image segmentation
    if trim_edges:
        padding: int = 12 if cropped_roof.size < 10_000 else 15
    else:
        padding: int = 0

    return detect_obstacles(
        blurred_roof=blurred_roof,
        source_image=satellite_image,
        box_or_polygon=obstacle_boundary_type,
        min_area=obstacle_minimum_area,
        padding_percentage=padding,
    )
