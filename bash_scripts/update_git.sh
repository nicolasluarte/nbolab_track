#!/bin/sh

# checks if repo exists if it does not exist it creates, else 
# pulls the repo, check how to do this more robustly
cd $HOME
cd uni && git pull || git clone https://github.com/nicolasluarte/uni
# pip3 -r requirements.txt
