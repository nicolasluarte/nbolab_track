#!/bin/bash

IP=($(awk -F ',' '{print $1}' ipfile))
HOSTNAME=($(awk -F ',' '{print $2}' ipfile))

for ((i=0; i<${#IP[@]}; i++))
do
	ssh ${IP[$i]} "pkill python3"
done
