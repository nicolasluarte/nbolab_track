#!/bin/sh

# remember to install rsync in pi

# get all background pics
rsync -a "$1":"$2" $HOME/uni/PHD/tracking_device/background
