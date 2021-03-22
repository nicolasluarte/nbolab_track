# libs
import time
from vidgear.gears import PiGear
import csv
import datetime
import socket
from track_functions import *

# file name
fileName = 'cam_recording_' + \
    datetime.datetime.now().strftime("%H:%M:%S_%d%m%Y") + '.csv'
# get host name
hostName = socket.gethostname() + '/'
# get background path
backgroundPath = '/home/pi/nbolab_EXPERIMENTS/' + \
    hostName + 'background/' + 'bg.png'
# get folder to store csv file
csvPath = '/home/pi/nbolab_EXPERIMENTS/' + hostName + 'data_cam/' + fileName
# get folder to store preview images
previewPath = '/home/pi/nbolab_EXPERIMENTS/' + hostName + 'preview_cam/'
# counter to name image
counter = 0

# algorithm parameters
d = 30  # filter size
sigma1 = 125  # sigma color
sigma2 = 125  # sigma space
kx = 5  # kernel x
ky = 5  # kernel y

# set the background file
bg = cv2.imread(backgroundPath, cv2.IMREAD_GRAYSCALE)
# resize the background
bg = cv2.resize(bg, (640, 480), interpolation=cv2.INTER_AREA)

# set the capture device (camera)
stream = PiGear(resolution=(640, 480), framerate=60,
                colorspace='COLOR_BGR2GRAY').start()

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
        "tailCentroidX",
        "tailCentroidY",
        "headCentroidX",
        "headCentroidY",
        "bodyContourArea",
        "error",
        "executionTime",
        "frameWidth",
        "frameHeight"
    ])

    startTime = time.time()
    while True:
        # start reading from cam stream
        frame = stream.read()
        # image processing
        frameDiff = bgfg_diff(bg, frame)  # background - foreground
        framePost, tailImage = postprocess_image(
            frameDiff, kx, ky)  # further processing
        # contour extraction
        extractionBody, extractionTail, centroidX, centroidY, centroidXT, centroidYT, centroidXH, centroidYH, area, err = contour_extraction(
            framePost, tailImage, 640, 480)
        # timing related stuff
        timeStamp = datetime.datetime.now().strftime("%Y %m %d %H %M %S %f")
        execTime = (time.time() - startTime)

        # data log
        log = list(map(int, timeStamp.split())) + \
            [
            centroidX,
            centroidY,
            centroidXT,
            centroidYT,
            centroidXH,
            centroidYH,
            area,
            err,
            execTime,
            640,
            480
        ]
        writer.writerow(log)
        # write preview images
        cv2.imwrite(previewPath + str(counter), framePost)
        counter = counter + 1

# stop stream
stream.stop()
