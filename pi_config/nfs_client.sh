#!/bin/bash

echo "enter the -h flag followed by the host ip address"

while getopts h:f: flag
do
	case "${flag}" in
		h) host_ip=${OPTARG};;
		f) host_shared_folder=${OPTARG};;
	esac
done

IP=($(awk -F ',' '{print $1}' ipfile))
HOSTNAME=($(awk -F ',' '{print $2}' ipfile))


for ((i=0; i<${#IP[@]}; i++))
do
	ssh ${IP[$i]} "sudo apt update &&
		sudo apt install nfs-common &&
		sudo mkdir -p ~/nfs &&
		sudo mount -o v3 $host_ip:$host_shared_folder ~/nfs && 
		sudo mkdir -p ~/nfs/$HOSTNAME &&
		sudo mkdir -p ~/nfs/$HOSTNAME/preview &&
		sudo mkdir -p ~/nfs/$HOSTNAME/csv &&
		sudo mkdir -p ~/nfs/$HOSTNAME/background"
done
