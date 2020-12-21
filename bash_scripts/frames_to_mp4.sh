#!/bin/sh

# only use inside the folder with the redlight videos

ffmpeg -framerate "$1" -pattern_type glob -i '*_*_c.png' \
-c:v libx264 -r 30 -pix_fmt yuv420p "$2"
