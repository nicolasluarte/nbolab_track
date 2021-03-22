# libs
import time
from vidgear.gears import PiGear
import csv
import datetime
import socket
from fast_track import *
import multiprocessing as mp

# file name
fileName = 'cam_recording_' + \
    datetime.datetime.now().strftime("%H:%M:%S_%d%m%Y") + '.csv'
# get host name
hostName = socket.gethostname() + '/'
# get background path
backgroundPath = '/home/pi/nbolab_EXPERIMENTS/' + \
    hostName + 'background/' + 'bg.jpg'
# get folder to store csv file
csvPath = '/home/pi/nbolab_EXPERIMENTS/' + hostName + 'data_cam/' + fileName
# get folder to store preview images
previewPath = '/home/pi/nbolab_EXPERIMENTS/' + hostName + 'preview_cam/'
# counter to name image
counter = 0
# width and height
w = 320
h = 240

# algorithm parameters
d = 30  # filter size
sigma1 = 125  # sigma color
sigma2 = 125  # sigma space
kx = 5  # kernel x
ky = 5  # kernel y

# set the background file
bg = cv2.imread(backgroundPath, cv2.IMREAD_GRAYSCALE)
# resize the background
bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_AREA)

# set the capture device (camera)
stream = PiGear(resolution=(w, h), framerate=60,
                colorspace='COLOR_BGR2GRAY').start()


def videoP():
    # start reading from cam stream
    frame = stream.read()
    # image processing
    frameDiff = bgfg_diff(bg, frame)  # background - foreground
    framePost = postprocess_image(
        frameDiff, kx, ky)  # further processing
    # contour extraction
    centroidX, centroidY, area, err = contour_extraction(
        framePost, w, h)
    print(centroidX, centroidY, area, err)


# open the csv and write headers
with open(csvPath, 'w') as f:
    # write headers
    writer = csv.writer(f)
    writer.writerow([
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "microsecond",
        "bodyCentroidX",
        "bodyCentroidY",
        "bodyContourArea",
        "error",
        "executionTime",
        "frameWidth",
        "frameHeight"
    ])

    while True:
        # timing related stuff

        mp.Pool(4).map(videoP)

        # data log
        # write preview images

# stop stream
stream.stop()
