#!/bin/bash

# get passwordless access in each one
# assign proper hostnames
# remember to put all ips and hostnames in the ipfile

IP=($(awk -F ',' '{print $1}' ipfile))
HOSTNAME=($(awk -F ',' '{print $2}' ipfile))

# variables for running previews
MAIN='nbolab_track.py/run/nbolab_track.py'
BG='uni/PHD/tracking_device/run/take_bg.py'
STREAM='nbolab_track.py/stream'

if [[ "$1" == "full" ]] || [[ "$1" == "all" ]]
then
	echo "Performing full setup including hostnames and passwordless login"
	for ((i=0; i<${#IP[@]}; i++))
	do 
		ssh-copy-id ${IP[$i]}
		ssh ${IP[$i]} "sudo hostnamectl set-hostname ${HOSTNAME[$i]}"
	done
fi

# broadcast reboot
if [[ "$1" == "reboot" ]] || [[ "$1" == "all" ]]
then
	echo "Performing reboot"
	for ((i=0; i<${#IP[@]}; i++))
	do 
		ssh ${IP[$i]} "sudo reboot"
	done
fi

# broadcast update
if [[ "$1" == "update" ]] || [[ "$1" == "all" ]]
then
	echo "Performing update"
	for ((i=0; i<${#IP[@]}; i++))
	do 
		ssh ${IP[$i]} "sudo apt-get update"
	done
fi

# broadcast python install
# also install requierements
if [[ "$1" == "python" ]] || [[ "$1" == "all" ]]
then
	echo "Performing python installation"
	for ((i=0; i<${#IP[@]}; i++))
	do 
		ssh ${IP[$i]} "sudo apt install python3"
		ssh ${IP[$i]} "sudo apt-get install git"
		ssh ${IP[$i]} "sudo rm -rf uni &&
			git clone https://github.com/nicolasluarte/uni.git && 
			cd ~/uni/PHD/tracking_device &&
			pip3 install --user -r requirements.txt" 
	done
fi

# broadcast cam setup
if [[ "$1" == "cam" ]] || [[ "$1" == "all" ]]
then
	echo "Performing cam installation"
	for ((i=0; i<${#IP[@]}; i++))
	do 
		ssh ${IP[$i]} "sudo apt-get install cmake libjpeg8-dev &&
			sudo apt-get install gcc g++ &&
			git clone https://github.com/jacksonliam/mjpg-streamer ||
			cd mjpg-streamer/mjpg-streamer-experimental &&
			make &&
			sudo make install"
	done
fi

# broadcast cam preview
if [[ "$1" == "campreview" ]] || [[ "$1" == "all" ]]
then
	echo "Performing cam preview"
	for ((i=0; i<${#IP[@]}; i++))
	do 

		ssh ${IP[$i]} "python3 $MAIN --mode preview & export LD_LIBRARY_PATH=. & mjpg_streamer -i 'input_file.so -f $STREAM -n stream.jpg -d 0.1' -o 'output_http.so -w /usr/local/www'"
	done

fi

# update repo
if [[ "$1" == "repo" ]] || [[ "$1" == "all" ]]
then
	echo "Performing repo update"
	for ((i=0; i<${#IP[@]}; i++))
	do 
		ssh ${IP[$i]} "sudo rm -rf nbolab_track &&
			git clone https://github.com/nicolasluarte/nbolab_track.git"
	done
fi

# main program run
if [[ "$1" == "main" ]] || [[ "$1" == "all" ]]
then
	echo "Performing main program"
	for ((i=0; i<${#IP[@]}; i++))
	do 
		ssh ${IP[$i]} "python3 $MAIN --capture 0"
	done

fi

# take background
if [[ "$1" == "take_bg" ]] || [[ "$1" == "all" ]]
then
	echo "taking background"
	for ((i=0; i<${#IP[@]}; i++))
	do 
		ssh ${IP[$i]} "python3 $BG --capture 0"
	done

fi
