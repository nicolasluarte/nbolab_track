import cv2
import numpy as np

def preprocess_image(image):
    bilateral_filtered_image = cv2.bilateralFilter(image, 7, 150, 150)
    gray_image = cv2.cvtColor(bilateral_filtered_image, cv2.COLOR_BGR2GRAY)
    return gray_image


cap = cv2.VideoCapture('rat.mp4')
img = cv2.imread('/home/nicoluarte/background.png')
bg = preprocess_image(img)
kernel = np.ones((5,5),np.uint8)

while(1):
    ret, frame = cap.read()
    frame = preprocess_image(frame)
    image_sub = cv2.absdiff(bg, frame)
    close_operated_image = cv2.morphologyEx(image_sub, cv2.MORPH_CLOSE, kernel)
    _, thresholded = cv2.threshold(close_operated_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    frame = cv2.medianBlur(thresholded, 5)

    fgmask = frame

    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(0) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
