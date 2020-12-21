#!/home/nicoluarte/uni/PHD/tracking_device/environments/bg_fg/bin/python3.8
import sys
sys.path.append('../')
from bg_fg import *
import matplotlib.pyplot as plt
import os
import glob
import csv
import socket


""" function only made to show an animation """
def update(i):
    p.set_data(live_preview(cap, bt))
    n.set_offsets(live_points(cap, bt))


""" read from virtual camera """
bg = cv2.imread('/home/nicoluarte/Downloads/background.jpeg')
cap = cv2.VideoCapture(0)
im = image_full_process(bg, cap, 3, 150, 150, 5, 5)
bt, tail, head, points = body_tracking(im)
points = live_points(cap, bt)
p = plt.imshow(live_preview(cap, bt))
n = plt.scatter(*zip(*points), marker="x", color="red")
ani = FuncAnimation(plt.gcf(), update, interval=50)
plt.show()
cap.release()
cv2.destroyAllWindows()

""" read from virtual camera and write to csv """
# csv label, so every record is distinct
label = str(socket.gethostname()) + "_" + \
        datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
bg = cv2.imread('/home/nicoluarte/Downloads/background.jpeg')
cap = cv2.VideoCapture(2)
with open('../csv_bak/' + label + '.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["YEAR", "MONTH", "DAY", "HOUR", "MINUTE", "SECOND", "MICROSECOND", "body_x", "body_y", "tail_x", "tail_y", "head_x", "head_y"])
    while(True):
        points = live_points(cap, bg)
        time_stamp = datetime.datetime.now().strftime("%Y %m %d %H %M %S %f")
        log = list(map(int, time_stamp.split())) + [item for i in points for item in i]
        writer.writerow(log)
cap.release()
cv2.destroyAllWindows()


### TEST AREA ###
# final_image = image_full_process(bg, fg)
# body, tail, head, points = body_tracking(final_image)
# remote_stream('192.168.1.117', '22', 'pi', 'dw4yb26f')
# end_remote_stream('192.168.1.117', '22', 'pi', 'dw4yb26f')

### DISPLAY AREA ###
# plt.imshow(head)
# plt.scatter(*zip(*points))
# plt.show()

