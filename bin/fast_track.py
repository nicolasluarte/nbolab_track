import cv2
import numpy as np
import skfmm
from scipy.spatial.distance import cdist


def postprocess_image(image, kx, ky):
    """
         Close morphological operation is faster and retains the tail
         It help in removing noise and fillling the gaps within the rat
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kx, ky))
    open_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return open_image


def contour_extraction(image, width, height, threshold=0.1):
    """
        Selects the contour with the largest area
        This prevents to perform calculation in other objects

        1. Input is binary image
        2. Gets the contours
        3. Creates a black to draw the contour
        4. Extracts the contour with the larges area and puts it into the canvas
    """
    # variable preset
    extraction = np.zeros((image.shape[0], image.shape[1]))
    centroidX = None
    centroidY = None
    area = None
    err = None
    # first a canvas is created, here the contour will be drawn, so no noise is present
    canvas_body = np.zeros((image.shape[0], image.shape[1]))
    # contours of the main image are extracted
    contours, _ = cv2.findContours(
        image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # if at least 1 contour is detected
    if len(contours) != 0:
        # get the largest contour by area
        cnt = max(contours, key=cv2.contourArea)
        # the same but for the tail
        # the area of the largest contour is computed
        # if the area is largest than a percent threshold
        # its considered as a detection error
        # 10% (0.1) is the default
        area = cv2.contourArea(cnt) / (width * height)
        # if threshold area is lower that threshold perform further computation
        if area < threshold:
            # both contours are drawn to two separate canvas
            extraction = cv2.drawContours(
                canvas_body, [cnt], -1, 255, thickness=-1)

            # The centroid or center of mass is calculated from image moments
            # This is donde for the center of mass of tail-less rat and
            # for the isolated tail
            M = cv2.moments(extraction)
            centroidX = int(M['m10'] / M['m00'])
            centroidY = int(M['m01'] / M['m00'])
            err = False
        else:
            print('Area too large, points not computed')
            # otherwise false
            err = True
            extraction = canvas_body
            centroidX = "None"
            centroidY = "None"
            area = "None"

    else:
        print('No contours found, points not computed')
        # if no contour is detected an empty canvas is returned
        # all other values are returned as none
        extraction = canvas_body
        centroidX = "None"
        centroidY = "None"
        area = "None"

    """
        extraction: the drawn contour of the main rat image
        extraction_tail: the drawn contour of the isolated tail
        centroidX: x coordinate of the rat main image
        centroidY: y coordinate of the rat main image
        centroidXT: x coordinate of the isolated tail
        centroidYT: y coordinate of the isolated tail 
        centroidXH: x coordinate of the estimated head position 
        centroidYH: y coordinate of the estimated head position 
        area: the contour area relative to the whole image
        err: True is some error happend, False if area and contour numbers are good
    """
    return centroidX, centroidY, area, err


def bgfg_diff(background, foreground):
    """
        Simple function to substract the rat from the background
        NOTE: it assumes both images are in grayscale
    """
    diff = cv2.absdiff(background, foreground)
    _, thresholded = cv2.threshold(
        diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded
