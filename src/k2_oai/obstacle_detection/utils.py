from __future__ import annotations

import cv2 as cv
import numpy as np


def get_bounding_boxes(
    source_photo, blob_stats, margins, min_area: int = 0, draw_image=True
):
    """Obtain bounding boxes from blobs resulting from the connected components'
    analysis. Then, draw them on the input image.

    Parameters
    ----------
    draw_image
    """
    bounding_box_coordinates = []

    labelled_image = source_photo.copy()

    for i in range(0, blob_stats.shape[0]):

        # TODO: evaluate to return the full set of coordinates
        if blob_stats[i, cv.CC_STAT_AREA] > min_area:
            top_left_px = (
                blob_stats[i, cv.CC_STAT_LEFT] + margins[1],  # height
                blob_stats[i, cv.CC_STAT_TOP] + margins[0],  # width
            )
            height = blob_stats[i, cv.CC_STAT_HEIGHT]
            width = blob_stats[i, cv.CC_STAT_WIDTH]
            bottom_right_px = (top_left_px[0] + width, top_left_px[1] + height)

            bounding_box_coordinates.append((top_left_px, bottom_right_px))

            if draw_image:
                labelled_image = cv.rectangle(
                    labelled_image, top_left_px, bottom_right_px, (255, 0, 0), 1
                )

    if draw_image:
        return bounding_box_coordinates, labelled_image
    return bounding_box_coordinates


def get_bounding_polygon(
    source_image, blobs_stats, blobs, min_area: int = 0, draw_image: bool = True
):
    """Obtain polygons delimiting obstacles, from blobs resulting from the connected
    components' analysis. Then, draw them on the input image.

    Parameters
    ----------
    draw_image
    """
    labelled_image = source_image.copy()

    polygon_coordinates = []

    for i in range(0, blobs_stats.shape[0]):
        if blobs_stats[i, cv.CC_STAT_AREA] > min_area:
            obst_im = (blobs == i) * 255
            contours, hierarchy = cv.findContours(
                obst_im.astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
            )
            approximated_boundary = cv.approxPolyDP(contours[0], 5.0, True)
            polygon_coordinates.append(approximated_boundary)

            if draw_image:
                cv.polylines(
                    labelled_image, [approximated_boundary], True, (255, 0, 0), 2
                )

    if draw_image:
        return polygon_coordinates, labelled_image
    return polygon_coordinates
