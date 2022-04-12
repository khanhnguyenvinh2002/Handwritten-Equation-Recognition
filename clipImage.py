from PIL import Image
import numpy as np
import cv2

# clipt image into box using OpenCV
im = cv2.imread("./test_1.png")
imgrey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgrey, 127, 255, 0)
contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(im, (x,y), (x+w, y+h), (0,255,0), 2)

result = Image.fromarray(im)
result.save("test.png")