#!/home/nicoluarte/uni/PHD/tracking_device/environments/bg_fg/bin/python3.8
import socket
import time
from imutils.video import VideoStream
import imagezmq

sender = imagezmq.ImageSender(connect_to='tcp://192.168.1.100:5555')

rpi_name = socket.gethostname()
picam = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
while True:
    try:
        image = picam.read()
        sender.send_image(rpi_name, image)
    except KeyboardInterrupt:
        picam.release()
        cv2.destroyAllWindows()
        print("BYE BYE")
        break


picam.release()
