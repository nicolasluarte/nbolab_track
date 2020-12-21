#!/bin/sh
# remember to start v4l2loopback
# sudo modprobe v4l2loopback
ffmpeg -re -i "$1" -map 0:v -f v4l2 /dev/"$2"
