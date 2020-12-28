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
	ssh ${IP[$i]} "sudo apt update && echo 'done update' &&
		sudo apt install nfs-common && echo 'done nfs install' &&
		sudo mkdir -p ~/nfs && echo 'done making directory' &&
		sudo mount -o v3 $host_ip:$host_shared_folder ~/nfs && echo 'done mounting' &&
		sudo mkdir -p ~/nfs/$HOSTNAME 'done creating hostname folder'"
done
