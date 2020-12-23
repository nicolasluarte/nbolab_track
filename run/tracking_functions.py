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

def contour_extraction(image, tail_image, width, height):
    """
        Selects the contour with the largest area
        This prevents to perform calculation in other objects

        1. Input is binary image
        2. Gets the contours
        3. Creates a black to draw the contour
        4. Extracts the contour with the larges area and puts it into the canvas
    """
    canvas = np.zeros((image.shape[0], image.shape[1]))
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    tail_contour, _ = cv2.findContours(tail_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = cv2.contourArea(cnt) / (width * height)
    if len(contours) != 0 and area < 0.1:
        area = cv2.contourArea(cnt)
        cnt = max(contours, key = cv2.contourArea)
        cnt_tail = max(tail_contour, key = cv2.contourArea)
        extraction_tail = cv2.drawContours(canvas, [cnt_tail], -1, 255, thickness=-1)
        """ get the tail centroid """
        MT = cv2.moments(extraction_tail)
        centroidXT = int(MT['m10'] / MT['m00'])
        centroidYT = int(MT['m01'] / MT['m00'])

        intersection = [centroidXT, centroidYT]
        extraction = cv2.drawContours(canvas, [cnt], -1, 255, thickness=-1)
        M = cv2.moments(extraction)
        centroidX = int(M['m10'] / M['m00'])
        centroidY = int(M['m01'] / M['m00'])
        hull = cv2.convexHull(cnt)
        extLeft = tuple(hull[hull[:, :, 0].argmin()][0])
        extRight = tuple(hull[hull[:, :, 0].argmax()][0])
        extTop = tuple(hull[hull[:, :, 1].argmin()][0])
        extBot = tuple(hull[hull[:, :, 1].argmax()][0])
        points = [extLeft, extRight, extTop, extBot]
        distant_points = cdist([(centroidXT, centroidYT)], points, 'euclidean')
        idx = np.argmax(distant_points)
        head = points[idx]
    else:
        extraction = canvas

    return extraction, centroidX, centroidY, area, head, intersection

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

def fast_track(image, contour):
    """
        This functions uses image moments to get the desired points
        Input image must be the processed one
    """
    # gets the centroid point
    cnt = contour
    M = cv2.moments(image)
    centroidX = int(M['m10'] / M['m00'])
    centroidY = int(M['m01'] / M['m00'])
    hull = cv2.convexHull(cnt)
    extLeft = tuple(hull[hull[:, :, 0].argmin()][0])
    extRight = tuple(hull[hull[:, :, 0].argmax()][0])
    extTop = tuple(hull[hull[:, :, 1].argmin()][0])
    extBot = tuple(hull[hull[:, :, 1].argmax()][0])
    points = [extLeft, extRight, extTop, extBot]
    distant_points = cdist((centroidX, centroidY), points, 'euclidean')
    np.fill_diagonal(distant_points, -np.inf)
    idx = np.unravel_index(np.argmin(distant_points, axis=None), distant_points.shape)
    
    return print(idx)


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
