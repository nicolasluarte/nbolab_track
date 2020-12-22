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

x = 160
y = 120
bg = cv2.imread('/home/pi/background.jpeg', cv2.IMREAD_GRAYSCALE)
bg = cv2.resize(bg, (x, y), interpolation = cv2.INTER_AREA)

"""
    Program main loop

    Writes a csv with the points extracted from the body tracking functions
    
    All parameters are defined in the config file, and initialized here

"""

### PARAMETERS ###
d=3
sigma1=75
sigma2=75
kx=3
ky=3
### PARAMETERS END ###

print("OFFLINE MODE")
files = glob.glob('/home/pi/rat/*.png')
files_slice = files[:300]
print(files_slice)
img = [cv2.imread(file) for file in files_slice]
for i in range(len(img)):
    START = time.time()
    # read a single frame
    #frame = cv2.cvtColor(img[i], cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(cv2.cvtColor(img[i], cv2.COLOR_BGR2GRAY), (x, y), interpolation = cv2.INTER_LINEAR)
    color = cv2.resize(img[i], (x, y), interpolation = cv2.INTER_LINEAR)

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
    percent = nc / (x * y)
    if percent < 0.1:
        M = cv2.moments(frame_post)
        centroidX = int(M['m10'] / M['m00'])
        centroidY = int(M['m01'] / M['m00'])
        tailX = 0
        tailY = 0
        headX = 0
        headY = 0
        img_jpg = cv2.circle(color, (centroidX, centroidY), radius=2, color=(0, 255, 0), thickness=-1)
        cv2.imshow('frame', color)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print('DROP FRAME')
    ### POINTS EXTRACTION END ###
    print("--- %s seconds ---" % (time.time() - START))
    time.sleep(1)

