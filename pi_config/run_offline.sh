#!/bin/bash


while getopts m:b:f:w:h: flag
do
	case "${flag}" in
		m) mode=${OPTARG};;
		b) background=${OPTARG};;
		f) folder=${OPTARG};;
		w) width=${OPTARG};;
		h) height=${OPTARG};;
	esac
done

IP=($(awk -F ',' '{print $1}' ipfile))
HOSTNAME=($(awk -F ',' '{print $2}' ipfile))

for ((i=0; i<${#IP[@]}; i++))
do
	ssh ${IP[$i]} "cd nbolab_track/run &&
		nohup python3 nbolab_track.py --nopi 0 --mode $mode \
		--background $background \
		--folder $folder \
		--width $width \
		--height $height >/dev/null 2>/dev/null &" & disown
done
