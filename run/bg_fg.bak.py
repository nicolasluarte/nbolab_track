#!/home/nicoluarte/uni/PHD/tracking_device/environments/bg_fg/bin/python3.8
import cv2
import numpy as np
import skfmm
import paramiko
import time, datetime
from matplotlib.animation import FuncAnimation



def preprocess_image(image, d, sigma1, sigma2):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bl_filter_image = cv2.bilateralFilter(gray_image, d, sigma1, sigma2) # test out parameters
    return bl_filter_image

""" the image gets dilated (if one pixel under the kernel is '1' the a determined pixel is also '1') and then followed by erosion (a pixel is 1 if all pixels under the kernel are 1) """
def postprocess_image(image, kx, ky):
   #kernel = np.ones((17,17), np.uint8) 
   # ellipse kernel works better
   kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kx,ky))
   close_operation = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
   _, thresholded = cv2.threshold(close_operation, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
   #blur = cv2.medianBlur(thresholded, 5)
   # didnt use blur, better results without it
   blur = thresholded
   return blur

""" takes the bigger contour area, all small areas are considered noise as there's is to be only one animal; it also requires a gray image"""
def contour_extraction(image):
    ret, thresh = cv2.threshold(image, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    extraction = cv2.drawContours(np.zeros((image.shape[0], image.shape[1])), [max(contours, key = cv2.contourArea)], -1, 255, thickness=-1)
    return extraction


""" outputs the difference between the background and background + animal, so animal can be isolated """
def bgfg_diff(background, foreground, d, sigma1, sigma2):
    bg = preprocess_image(background, d, sigma1, sigma2)
    fg = preprocess_image(foreground, d, sigma1, sigma2)
    diff = cv2.absdiff(bg, fg)
    return diff

""" this function does all the pipeline """
def image_full_process(background, foreground, d, sigma1, sigma2, kx, ky):
    # convert rgb to gray scale
    diff = bgfg_diff(foreground, background, d, sigma1, sigma2)
    # post process the difference to remove noise
    diff_postproc = postprocess_image(diff, kx, ky)
    # extract the biggest contour area and paste it in an empty canvas
    final_image = contour_extraction(diff_postproc)
    return final_image

""" function to get the head, tail and body position
    requires the final_image"""
def body_tracking(image):
    # first we get the geodesic distance
    # because mice and rats are fluffy the torso is the furthest away distance
    distance = skfmm.distance(image)
    # get the highest point
    bx, by = np.unravel_index(distance.argmax(), distance.shape)
    # the furthest away point from this is the tail
    # create a boolen mask
    mask = ~image.astype(bool)
    # we mark with 0 the body
    tail = np.ones_like(image)
    tail[bx][by] = 0
    t = np.ma.MaskedArray(tail, mask)
    # get the distance from 0
    # the highest one is the tip of the tail
    tail_distance = skfmm.distance(t)
    # same to get the head, but calculate it from the tip of the tail
    head = np.ones_like(image)
    tx, ty = np.unravel_index(tail_distance.argmax(), tail_distance.shape)
    head[tx][ty] = 0
    h = np.ma.MaskedArray(head, mask)
    head_distance = skfmm.distance(h)
    # get the exact point
    hx, hy = np.unravel_index(head_distance.argmax(), head_distance.shape)
    body_points = (by, bx)
    tail_points = (ty, tx)
    head_points = (hy, hx)
    points = [body_points, tail_points, head_points]
    return distance, tail_distance, head_distance, points

"""function to send command to remote pi"""
def remote_pi_command(ip, port, username, password, command):
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(ip, port=port, username=username, password=password)
        stdin, stdout, stderr = ssh.exec_command(command)
        print(stdout.readlines())

    finally:
        if ssh:
            ssh.close()

""" starts a remote cam server can be opened with vlc http://192.168.1.117:8000/stream.wmw"""
def remote_stream(ip, port, username, password):
    cmd = 'cvlc v4l2:///dev/video0 :v4l2-standard= :input-slave=alsa://hw:0,0 :live-caching=300 :sout="#transcode{vcodec=WMV2,vb=800,scale=1,acodec=wma2,ab=128,channels=2,samplerate=44100}:http{dst=:8000/stream.wmv}"'
    remote_pi_command(ip, port=port, username=username, password=password, command=cmd)

""" ends the stream by killing vlc """
def end_remote_stream(ip, port, username, password):
    cmd = 'pkill vlc'
    remote_pi_command(ip, port=port, username=username, password=password, command=cmd)

""" takes a picture """
def take_background(path, capture):
    cam = cv2.VideoCapture(capture)   # should be the infrared cam
    cam.set(3, 320)
    cam.set(4, 240)
    ret, frame = cam.read()
    gray_background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if ret:    # frame captured without any errors
        cv2.imwrite(path, gray_background) #save image
        cam.release()

""" live preview with geodesic distance takes the body as reference """
def live_preview(cap, body_track):
    ret, frame = cap.read()
    frame, _, _, _ =  body_track
    return frame

""" live preview but gets the points """
def live_points(cap, body_track):
    ret, frame = cap.read()
    _, _, _, points =  body_track
    return points

""" function only made to show an animation """
def update(i):
    p.set_data(live_preview(cap, bg))
    n.set_offsets(live_points(cap, bg))


""" to see what is being printed into the csv """
def test(cap, bg, n):
    for i in range(n):
        ret, frame = cap.read()
        _, _, _, points = body_tracking(image_full_process(bg, frame))
        time_stamp = datetime.datetime.now()
        f = [time_stamp, points]
        print(f)
