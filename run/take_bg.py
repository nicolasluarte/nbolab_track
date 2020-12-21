import sys
from tracking_functions import *
import os
import glob
import csv
import socket
import argparse
from configparser import ConfigParser
from time import time as timer
from os.path import expanduser
from pathlib import Path


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
repo = home + '/uni/PHD/tracking_device'
config = repo + '/config/config.conf'
backgrounds = repo + '/background'
hostname = os.popen('hostname').read().rstrip('\n')
csv_files = repo + '/csv_bak'

print(config)

"""
    Read the parameters in the configuration file
    NOTE: file must by built with python
"""
parser = ConfigParser()
parser.read(config)

"""
    Read the arguments passed for program execution
"""
parserArg = argparse.ArgumentParser(description='take background picture')
parserArg.add_argument('-d', '--file_dest', type=str, help='destination of background')
parserArg.add_argument('-c', '--capture', type=int, help='set the camera stream vide0 as default')
args = parserArg.parse_args()

"""
    Set the arguments passed or set to defaults
"""
if args.file_dest is not None:
    path = str(args.file_name) + '/bg_' + hostname + '.png'
else:
    print("saving picture as default...")
    path = backgrounds + '/bg_' + str(hostname) + '.png'

# set cam
if args.capture is not None:
    cap = int(args.capture)
else:
    print("set to the default camera")
    cap = 0

take_background(path,
        cap,
        d=parser.getint('preprocess', 'filter_size'),
        sigma1=parser.getint('preprocess', 'sigma_color'),
        sigma2=parser.getint('preprocess', 'sigma_space'),
        height=240,
        width=340)
