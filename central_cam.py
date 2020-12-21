#!/home/nicoluarte/uni/PHD/tracking_device/environments/bg_fg/bin/python3.8
import cv2
import csv
import imagezmq
from bg_fg import *


bg = cv2.imread('/home/nicoluarte/uni/PHD/tracking_device/background.png')
image_hub = imagezmq.ImageHub()
while True:
    try:
        rpi_name, image = image_hub.recv_image()
        _, _, _, points = body_tracking(image_full_process(bg, image))
        print(points)
        cv2.imshow(rpi_name, image)
        cv2.waitKey(1)
        image_hub.send_reply(b'OK')
    
    except KeyboardInterrupt:
        break


with open('pi_list.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)

for f in data:
    remote_pi_command(ip=f[0], 
            port=f[1],
            username=f[2],
            password=f[3],
            command = 'pkill -f /home/pi/uni/PHD/tracking_device/remote_cam.py')
    print('DONE')
