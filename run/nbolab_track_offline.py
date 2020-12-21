#!/home/nicoluarte/uni/PHD/tracking_device/environments/bg_fg/bin/python3.8
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


if __name__ == '__main__':
    # read the config file
    parser = ConfigParser()
    parser.read('../config/config.conf')

# read the arguments
parserArg = argparse.ArgumentParser(description='write frame track to csv')
parserArg.add_argument('--frames_path', type=str, help='where are the frames')
parserArg.add_argument('--background', type=str, help='specify the path of back')
parserArg.add_argument('--filename', type=str, help='specify filename')
args = parserArg.parse_args()


# frames path
if args.frames_path is not None:
    path = str(args.frames_path)
else:
    path = str(socket.gethostname()) + "_" + \
        datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# file name
if args.frames_path is not None:
    label = str(args.filename)
else:
    label = str(socket.gethostname()) + "_" + \
        datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# set the background
if args.background is not None:
    bg = cv2.imread(args.background)
    print("loaded background from " + args.background)
else:
    bg = cv2.imread('../background/bg.png')
    print("load default path for background")

# load frames
frames_list = sorted(glob.glob(path + '/*.png'))
print(len(frames_list))

 # start writing into csv
with open('../csv_bak/' + label + '.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["YEAR", "MONTH", "DAY", "HOUR", "MINUTE", "SECOND", "MICROSECOND", "body_x", "body_y", "tail_x", "tail_y", "head_x", "head_y", "FRAME_PATH"])
    for i in frames_list:
        # read a single frame
        frame = cv2.imread(i)
        # process the image
        fp, _, _, points = body_tracking(image_full_process(
            background=bg,
            foreground=frame,
            d=parser.getint('preprocess', 'filter_size'),
            sigma1=parser.getint('preprocess', 'sigma_color'),
            sigma2=parser.getint('preprocess', 'sigma_space'),
            kx=parser.getint('postprocess', 'kernelx'),
            ky=parser.getint('postprocess', 'kernely')
            ))
        time_stamp = datetime.datetime.now().strftime("%Y %m %d %H %M %S %f")
        log = list(map(int, time_stamp.split())) + [item for i in points for item in i] + [os.path.basename(i)]
        writer.writerow(log)
