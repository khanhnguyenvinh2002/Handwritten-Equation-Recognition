from PIL import Image
import numpy as np
import cv2

# clipt image into box using OpenCV
im = cv2.imread("./test_1.png")
imgrey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgrey, 127, 255, 0)
contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

tempRes = []
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    tempRes.append([(x,y), (x+w, y+h)])

finalRes = []
tempRes.sort()
i = 0
while (i < len(tempRes) - 1):
    (x, y), (xw, yh) = tempRes[i]
    (x1, y1), (xw1, yh1) = tempRes[i + 1]
    
    equation = (yh - y) / (xw - x) < 0.2 and (yh1 - y1) / (xw1 - x1) < 0.2 and abs(x1 - x) < 20
    division = (abs(x1 - x) < min(abs(xw - x) / 2, abs(xw1 - x1) / 2)
                and abs(y1 - yh) < max(abs(yh - y) / 2, abs(yh1 - y1) / 2)
                and (yh - y) / (xw - x) > 0.5 and (yh1 - y1) / (xw1 - x1) > 0.5)
    letterI = (((yh - y) / (xw - x) > 5 and (yh1 - y1) / (xw1 - x1) < 2)
               or ((yh - y) / (xw - x) < 2 and (yh1 - y1) / (xw1 - x1) > 5) and abs(x1 - x) < 2)
    divisionMark = False
    if i < len(tempRes) - 2:
        (x2, y2), (xw2, yh2) = tempRes[i + 2]
        divisionMark = ((yh - y) / (xw - x) < 0.2 and 0.7 < (yh1 - y1) / (xw1 - x1) < 1.3
                        and 0.7 < (yh2 - y2) / (xw2 - x2) < 1.3 and x < x1 < x2 < xw and max(y1, y2) > y
                        and min(y1, y2) < y and max(y1, y2) - min(y1, y2) < 1.2 * abs(xw - x))

    if equation or letterI and not division:
        finalRes.append([(min(x, x1), min(y, y1)), (max(xw, xw1), max(yh, yh1))])
        i += 2
    elif divisionMark and not division:
        finalRes.append([(min(x, x1, x2), min(y, y1, y2)), (max(xw, xw1, xw2), max(yh, yh1, yh2))])
        i += 3
    else:
        finalRes.append(tempRes[i])
        i += 1

while i < len(tempRes):
    finalRes.append(tempRes[i])
    i += 1
    
for (left, right) in finalRes:
    #print([left, right])
    cv2.rectangle(im, left, right, (0,255,0), 2)
    
Image.fromarray(im).show()
