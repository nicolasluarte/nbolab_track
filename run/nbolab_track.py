from tracking_functions import *
import os
import glob
import csv
import socket
import argparse
from configparser import ConfigParser
from time import time as timer
from pathlib import Path
import picamera
import datetime
import time
from vidgear.gears import PiGear

"""
    Paths definition

    Home
    Repository
    Config
    Background
    Background file: bg_HOSTNAME.png
    csv files
"""
home = str(Path.home())
repo = home + '/nbolab_track'
config = repo + '/config/config.conf'
backgrounds = repo + '/background'
hostname = os.popen('hostname').read().rstrip('\n')
csv_files = repo + '/csv_bak'

"""
    Read the parameters in the configuration file
    NOTE: file must by built with python
"""
parser = ConfigParser()
parser.read(config)


"""
    Read the arguments passed for program execution
"""
parserArg = argparse.ArgumentParser(description='write tracked frame to csv')
parserArg.add_argument('--file_name', type=str, help='name of the file')
parserArg.add_argument('--background', type=str, help='specify the path of back')
parserArg.add_argument('--capture', type=int, help='set the camera stream vide0 as default')
parserArg.add_argument('--fps', type=int, help='set frames per second for processing')
args = parserArg.parse_args()

"""
    Set the arguments passed or set to defaults
"""
# defines the csv filename, uses hostname + date as default
if args.file_name is not None:
    label = '/' + str(args.file_name)
else:
    label = '/' + str(socket.gethostname()) + "_" + \
        datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# set the background, uses the background folder as default
if args.background is not None:
    bg = cv2.imread(args.background, cv2.IMREAD_GRAYSCALE)
    print("Background size:" + str(bg.shape))
else:
    bg = cv2.imread(backgrounds + '/bg_' + str(hostname) + '.png', cv2.IMREAD_GRAYSCALE)
    print("Background size:" + str(bg.shape) + " Loaded from: " + backgrounds + '/bg_' + str(hostname) + '.png')

# set fps control
if args.fps is not None:
    fps = args.fps
    fps /= 1000
    print("user defined FPS: " + str(args.fps))
else:
    fps = 30

# set the capture device
stream = PiGear(resolution=(320, 240), framerate=60, colorspace='COLOR_BGR2GRAY').start()
test_frame = stream.read()
print("Foreground size: " + str(test_frame.shape))

"""
    Program main loop

    Writes a csv with the points extracted from the body tracking functions
    
    All parameters are defined in the config file, and initialized here

"""

### PARAMETERS ###
d=parser.getint('preprocess', 'filter_size')
sigma1=parser.getint('preprocess', 'sigma_color')
sigma2=parser.getint('preprocess', 'sigma_space')
kx=parser.getint('postprocess', 'kernelx')
ky=parser.getint('postprocess', 'kernely')
print("Filter size: " + str(d))
print("Sigma1 size: " + str(sigma1))
print("Sigma2 size: " + str(sigma2))
print("Kernel size: " + str((kx, ky)))
### PARAMETERS END ###

with open(csv_files + label + '.csv', 'w') as f:

    ### CSV HEADERS ###
    writer = csv.writer(f)
    writer.writerow(["YEAR", "MONTH", "DAY", "HOUR", "MINUTE", "SECOND", "MICROSECOND", "centroid_x", "centroid_y", "tail_x", "tail_y", "head_x", "head_y"])
    ### CSV HEADERS END ###

    while(True):

        ### FPS CONTROL ### 
        # start = timer()
        ### FPS CONTROL END ###

        ### EXEC TIME CALC ### 
        start_time = time.time()
        ###                ###

        # read a single frame
        frame = stream.read()

        ### IMAGE PROCESSING ###
        start_time1 = time.time()
        frame_filter = preprocess_image(frame, d, sigma1, sigma2)
        print("--- %s filter seconds ---" % (time.time() - start_time1))
        start_time2 = time.time()
        frame_diff = bgfg_diff(bg, frame_filter, d, sigma1, sigma2)
        print("--- %s diff seconds ---" % (time.time() - start_time2))
        start_time3 = time.time()
        contours, nc = contour_extraction(frame_diff)
        print("--- %s contours seconds ---" % (time.time() - start_time3))
        print("Contours: " + str(nc))
        start_time4 = time.time()
        frame_post = postprocess_image(contours, kx, ky)
        print("--- %s post seconds ---" % (time.time() - start_time4))
        time_stamp = datetime.datetime.now().strftime("%Y %m %d %H %M %S %f")
        ### IMAGE PROCESSING END ###

        ### POINTS EXTRACTION ###
        if nc != 0:
            M = cv2.moments(frame_post)
            centroidX = int(M['m10'] / M['m00'])
            centroidY = int(M['m01'] / M['m00'])
            tailX = 0
            tailY = 0
            headX = 0
            headY = 0
        ### POINTS EXTRACTION END ###

        ### EXEC TIME CALC ###
        print("--- %s seconds ---" % (time.time() - start_time))
        ###           ###

        ### PARSING DATA ###
        log = list(map(int, time_stamp.split())) + [centroidX,
                centroidY,
                tailX,
                tailY,
                headX,
                headY]
        writer.writerow(log)
        ### PARSING DATA END ###

        ### FPS CONTROL ###
        # diff = timer() - start
        # while diff < fps:
        #     diff = timer() - start
        ### FPS CONTROL END ###

### STOP ###
stream.stop()
cap.release()
cv2.destroyAllWindows()
