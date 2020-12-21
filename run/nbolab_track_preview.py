#!/home/pi/uni/PHD/tracking_device/environments/bg_fg/bin/python3.8
from threading import Thread
import cv2, time
import sys
sys.path.append('../')
from bg_fg import *
import os
import glob
import csv
import socket
import argparse
from configparser import ConfigParser
from time import time as timer
import picamera
from vidgear.gears import PiGear

if __name__ == '__main__':
    # read the config file
    parser = ConfigParser()
    parser.read('/home/pi/uni/PHD/tracking_device/config/config.conf')

# read the arguments
parserArg = argparse.ArgumentParser(description='write frame track to csv')
parserArg.add_argument('--file_name', type=str, help='name of the file')
parserArg.add_argument('--background', type=str, help='specify the path of back')
parserArg.add_argument('--capture', type=int, help='set the camera stream vide0 as default')
parserArg.add_argument('--fps', type=int, help='set frames per second for processing')
args = parserArg.parse_args()


# set the background
if args.background is not None:
    bg = cv2.imread(args.background, cv2.IMREAD_GRAYSCALE)
    print("loaded background from " + args.background)
    print("background image size: " + str(bg.shape))
else:
    bg = cv2.imread('/home/pi/uni/PHD/tracking_device/background/bg.png', cv2.IMREAD_GRAYSCALE)
    print("load default path for background")
    print("background image size: " + str(bg.shape))

## set the capture device
#if args.capture is not None:
#    cap = cv2.VideoCapture(args.capture)
#    _, aux_frame = cap.read()
#    print("user defined cam selected: " + str(args.capture))
#    print("cam image size: " + str(aux_frame.shape))
#else:
#    cap = cv2.VideoCapture(0)
#    _, aux_frame = cap.read()
#    print("default camera selected")
#    print("cam image size: " + str(aux_frame.shape))
#
## set fps control
#if args.fps is not None:
#    fps = args.fps
#    fps /= 1000
#    print("user defined FPS: " + str(args.fps))
#else:
#    fps = cap.get(cv2.CAP_PROP_FPS)
#    fps /= 1000
#    print("selected camara default FPS:" + str(fps*1000))

## MULTI ##

#class VideoGet:
#    """
#    Class that continuously gets frames from a VideoCapture object
#    with a dedicated thread.
#    """
#
#    def __init__(self, src=0):
#        #self.stream = cv2.VideoCapture(src)
#        self.stream = PiCamera()
#        #self.stream.set(3, 320)
#        #self.stream.set(4, 240)
#        self.stream.resolution = (320, 240)
#        #(self.grabbed, self.frame) = self.stream.read()
#        self.stream.capture(self.frame)
#        self.stopped = False
#
#    def start(self):    
#        Thread(target=self.get, args=()).start()
#        return self
#
#    def get(self):
#        while not self.stopped:
#            if not self.grabbed:
#                self.stop()
#            else:
#                (self.grabbed, self.frame) = self.stream.read()
#
#    def stop(self):
#        self.stopped = True
#
#
#video_getter = VideoGet(0).start()
#bg = cv2.resize(bg, (128, 128), interpolation = cv2.INTER_AREA)
stream = PiGear(resolution=(320, 240), framerate=60, colorspace='COLOR_BGR2GRAY').start()
d=parser.getint('preprocess', 'filter_size')
sigma1=parser.getint('preprocess', 'sigma_color')
sigma2=parser.getint('preprocess', 'sigma_space')
kx=parser.getint('postprocess', 'kernelx')
ky=parser.getint('postprocess', 'kernely')

while(True):
    # start timer for fps control
    foreground = stream.read()
    fg_filter = preprocess_image(foreground, d, sigma1, sigma2)
    img_diff = bgfg_diff(bg, fg_filter, d, sigma1, sigma2)
    contours = contour_extraction(img_diff)
    post_proc = postprocess_image(contours, kx, ky)
    start = timer()
    #frame, _, _, points = body_tracking(post_proc)
    #body = skfmm.distance(post_proc)
    #bx, by = np.unravel_index(body.argmax(), body.shape)
    body = cv2.moments(post_proc)
    bx = int(body['m10'] / body['m00'])
    by = int(body['m01'] / body['m00'])
    diff = timer() - start

    # plot stuff
    img_jpg = cv2.circle(post_proc, (bx, by), radius=8, color=(0, 0, 255), thickness=-1)
    #img_jpg = cv2.circle(frame, points[1], radius=8, color=(0, 0, 255), thickness=-1)
    #img_jpg = cv2.circle(frame, points[2], radius=8, color=(0, 0, 255), thickness=-1)
    #cv2.putText(img_jpg, str(diff), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)
    cv2.imshow('frame', post_proc)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #cv2.imwrite('/home/pi/uni/PHD/tracking_device/stream/stream.jpg', img_jpg) 
    print('{:.7f}'.format(diff))
    # end timer for fps control
stream.stop()
cap.release()
cv2.destroyAllWindows()
