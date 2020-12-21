#!/home/nicoluarte/uni/PHD/tracking_device/environments/bg_fg/bin/python3.8
from bg_fg import remote_pi_command
import csv


"""
ip = 0
port = 1
name = 2
passwd = 3
"""
with open('pi_list.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)

for f in data:
    remote_pi_command(ip=f[0], 
            port=f[1],
            username=f[2],
            password=f[3],
            command = 'nohup python3 /home/pi/uni/PHD/tracking_device/remote_cam.py </dev/null >command.log 2>&1 &')
    print('DONE')
