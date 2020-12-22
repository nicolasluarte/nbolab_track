from tracking_functions import *
import os
import glob
import csv
import socket
import argparse
from configparser import ConfigParser
from time import time as timer
from pathlib import Path
import datetime
import time

bg = cv2.imread('/home/nicoluarte/Downloads/background.jpeg', cv2.IMREAD_GRAYSCALE)
#bg = cv2.resize(bg, (120, 120), interpolation = cv2.INTER_AREA)

"""
    Program main loop

    Writes a csv with the points extracted from the body tracking functions
    
    All parameters are defined in the config file, and initialized here

"""

### PARAMETERS ###
d=3
sigma1=125
sigma2=125
kx=5
ky=5
### PARAMETERS END ###

print("OFFLINE MODE")
img = glob.glob('/home/nicoluarte/Downloads/mice_test/Annotated/redlight/rat_01_seq_03_redlight/Frames_2017_10_27_12_53_25_correctedFrames/*.png')
print(img)
for i in range(1000):
    START = time.time()
    # read a single frame
    frame = cv2.imread(img[i], cv2.IMREAD_GRAYSCALE)
    color = cv2.imread(img[i])
    #frame = cv2.resize(frame, (120, 120), interpolation = cv2.INTER_LINEAR)

    ### IMAGE PROCESSING ###
    start_time1 = time.time()
    frame_filter = preprocess_image(frame, d, sigma1, sigma2)
    start_time2 = time.time()
    frame_diff = bgfg_diff(bg, frame_filter)
    start_time3 = time.time()
    contours, nc = contour_extraction(frame_diff)
    start_time4 = time.time()
    frame_post = postprocess_image(contours, kx, ky)
    ### IMAGE PROCESSING END ###

    ### POINTS EXTRACTION ###
    print("NC: " + str(nc))
    M = cv2.moments(frame_post)
    centroidX = int(M['m10'] / M['m00'])
    centroidY = int(M['m01'] / M['m00'])
    tailX = 0
    tailY = 0
    headX = 0
    headY = 0
    img_jpg = cv2.circle(color, (centroidX, centroidY), radius=5, color=(0, 255, 0), thickness=-1)
    cv2.imshow('frame', color)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_post = []
    ### POINTS EXTRACTION END ###
    print("--- %s seconds ---" % (time.time() - START))

