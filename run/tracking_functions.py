import cv2
import numpy as np
import skfmm
from scipy.spatial.distance import cdist

def preprocess_image(image, d, sigma1, sigma2):
    """
        Bilateral filter smooth the images while retaining edges
        NOTE: it assumes image are already in gray scale
        Grayscale is done in the run file
    """
    return cv2.bilateralFilter(image, d, sigma1, sigma2)

def postprocess_image(image, kx, ky):
   """
        Close morphological operation is faster and retains the tail 
        It help in removing noise and fillling the gaps within the rat
   """
   kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kx,ky))
   open_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
   tophat_image = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
   return open_image, tophat_image

def contour_extraction(image, tail_image, width, height, threshold=0.1):
    """
        Selects the contour with the largest area
        This prevents to perform calculation in other objects

        1. Input is binary image
        2. Gets the contours
        3. Creates a black to draw the contour
        4. Extracts the contour with the larges area and puts it into the canvas
    """

    # first a canvas is created, here the contour will be drawn, so no noise is present
    canvas_body = np.zeros((image.shape[0], image.shape[1]))
    canvas_tail = np.zeros((image.shape[0], image.shape[1]))
    # contours of the main image are extracted
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours of the isolated tail are extracted
    tail_contour, _ = cv2.findContours(tail_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # if at least 1 contour is detected
    if len(contours) != 0:

        # get the largest contour by area
        cnt = max(contours, key = cv2.contourArea)
        # the same but for the tail
        cnt_tail = max(tail_contour, key = cv2.contourArea)
        # the area of the largest contour is computed
        # if the area is largest than a percent threshold 
        # its considered as a detection error
        # 10% (0.1) is the default
        area = cv2.contourArea(cnt) / (width * height)
        # if threshold area is lower that threshold perform further computation
        if area < threshold:

            # both contours are drawn to two separate canvas
            extraction = cv2.drawContours(canvas_body, [cnt], -1, 255, thickness=-1)
            extraction_tail = cv2.drawContours(canvas_tail, [cnt_tail], -1, 255, thickness=-1)

            # The centroid or center of mass is calculated from image moments
            # This is donde for the center of mass of tail-less rat and
            # for the isolated tail
            M = cv2.moments(extraction)
            MT = cv2.moments(extraction_tail)
            centroidX = int(M['m10'] / M['m00'])
            centroidY = int(M['m01'] / M['m00'])
            centroidXT = int(MT['m10'] / MT['m00'])
            centroidYT = int(MT['m01'] / MT['m00'])
            
            # For purposes of drawing we put both points as lists
            # This makes it easier to plot
            tail_centroid = [centroidXT, centroidYT]
            body_centroid = [centroidX, centroidY]
            
            # We do gift wrapping upon the contour (hull)
            hull = cv2.convexHull(cnt)
            # we get all the extreme points in the hull
            extLeft = tuple(hull[hull[:, :, 0].argmin()][0])
            extRight = tuple(hull[hull[:, :, 0].argmax()][0])
            extTop = tuple(hull[hull[:, :, 1].argmin()][0])
            extBot = tuple(hull[hull[:, :, 1].argmax()][0])
            # make a list of them
            points = [extLeft, extRight, extTop, extBot]
            # calculate the distance from the tail centroid to every point
            # in the hull
            distant_points = cdist([(centroidXT, centroidYT)], points, 'euclidean')
            # The furthest apart point is the head 
            idx = np.argmax(distant_points)
            # we index the point in the hull corresponding with the head
            head = points[idx]
            centroidXH = head[0]
            centroidYH = head[1]
            # if computations are succesful err is false
            err = False
        else:
            # otherwise false
            err = True
            extraction = canvas_body
            extraction_tail = canvas_tail
            centroidX = "None"
            centroidY = "None"
            centroidXT = "None"
            centroidYT = "None"
            centroidXH = "None"
            centroidYH = "None"
            area = "None"
            head = "None"

    else:
        # if no contour is detected an empty canvas is returned
        # all other values are returned as none
        extraction = canvas_body
        extraction_tail = canvas_tail
        centroidX = "None"
        centroidY = "None"
        centroidXT = "None"
        centroidXT = "None"
        centroidXH = "None"
        centroidYH = "None"
        area = "None"
        head = "None"

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

    return extraction, extraction_tail, centroidX, centroidY, centroidXT, centroidYT, centroidXH, centroidYH, area, err


def bgfg_diff(background, foreground):
    """
        Simple function to substract the rat from the background
        NOTE: it assumes both images are in grayscale
    """
    diff = cv2.absdiff(background, foreground)
    _, thresholded = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded

""" this function does all the pipeline """
def image_full_process(background, foreground, d, sigma1, sigma2, kx, ky):
    """
        This function performs all image processing pipeline
        
        1. Substract the background from rat
        2. Makes all the relevant post-processing, to isolate the rat and remove noise
        3. Gets the contours and selects the one with the rat
    """
    diff = bgfg_diff(foreground, background, d, sigma1, sigma2)
    diff_postproc = postprocess_image(diff, kx, ky)
    final_image = contour_extraction(diff_postproc)
    return final_image

def body_tracking(image):
    """
        This function get the head, body and tail by performing geodesic in images with boundaries
        Is too slow for the pi ~1-3 FPS
    """
    # first we get the geodesic distance
    # because mice and rats are fluffy the torso is the furthest away distance
    distance = skfmm.distance(image)
    # get the highest point
    bx, by = np.unravel_index(distance.argmax(), distance.shape)
    # the furthest away point from this is the tail
    # create a boolen mask
    mask = ~image.astype(bool)
    # we mark with 0 the body
    tail = np.ones_like(image)
    tail[bx][by] = 0
    t = np.ma.MaskedArray(tail, mask)
    # get the distance from 0
    # the highest one is the tip of the tail
    tail_distance = skfmm.distance(t)
    # same to get the head, but calculate it from the tip of the tail
    head = np.ones_like(image)
    tx, ty = np.unravel_index(tail_distance.argmax(), tail_distance.shape)
    head[tx][ty] = 0
    h = np.ma.MaskedArray(head, mask)
    head_distance = skfmm.distance(h)
    # get the exact point
    hx, hy = np.unravel_index(head_distance.argmax(), head_distance.shape)
    body_points = (by, bx)
    tail_points = (ty, tx)
    head_points = (hy, hx)
    points = [body_points, tail_points, head_points]
    return distance, tail_distance, head_distance, points


def take_background(path, capture, d, sigma1, sigma2, height=480, width=640):
    """
        Simple function to take the background image
        Uses a bilateral filter on the image
        Need to specify resolution
    """
    cam = cv2.VideoCapture(capture)
    cam.set(3, width)
    cam.set(4, height)
    ret, frame = cam.read()
    gray_background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_filtered_image = cv2.bilateralFilter(gray_background, d, sigma1, sigma2)
    if ret:    # frame captured without any errors
        cv2.imwrite(path, gray_filtered_image) #save image
        cam.release()
    else:
        print("Something went wrong")
