import PIL.ImageOps
import numpy as np
import cv2
import glob
import os

from PIL import Image, ImageDraw

# clipt image into box using OpenCV

# TA input size: 1696 * 117
# pm  - not tried
# isFraction  - not updated

# detect if input boundingBox contains a dot
def isDot(boundingBox):
    (x, y), (xw, yh) = boundingBox
    return abs(xw - x) < 15 and abs(yh - y) < 15 and 0.75 < (xw - x)/(yh - y) < 1.5  # 15 is migical number

# detect if input boundingBox contains a vertical bar
def isVerticalBar(boundingBox):
    (x, y), (xw, yh) = boundingBox
    return (yh - y) / (xw - x) > 2

# detect if a given boundingBox contains a horizontal bar
def isHorizontalBar(boundingBox):
    (x, y), (xw, yh) = boundingBox
    return (xw - x) / (yh - y) > 2

# detect if input boundingBox contains a square (regular letters, numbers, operators)
def isSquare(boundingBox):
    (x, y), (xw, yh) = boundingBox
    return (xw - x) > 8 and (yh - y) > 8 and 0.75 < (xw - x)/(yh - y) < 1.5

# detect if input three boundingBoxes are a division mark
def isDivisionMark(boundingBox, boundingBox1, boundingBox2):
    (x, y), (xw, yh) = boundingBox
    (x1, y1), (xw1, yh1) = boundingBox1
    (x2, y2), (xw2, yh2) = boundingBox2
    return (isHorizontalBar(boundingBox) and isDot(boundingBox1) and isDot(boundingBox2)
            and x < x1 < x2 < xw and max(y1, y2) > y and min(y1, y2) < y
            and max(y1, y2) - min(y1, y2) < 1.2 * abs(xw - x))

# detect if input two boundingBoxes are a lowercase i
def isLetterI(boundingBox, boundingBox1):
    (x, y), (xw, yh) = boundingBox
    (x1, y1), (xw1, yh1) = boundingBox1
    return (((isDot(boundingBox) and isVerticalBar(boundingBox1)) or (isDot(boundingBox1) and isVerticalBar(boundingBox)))
            and abs(x1 - x) < 10)  # 10 is a magical number

# detect if input two boundingBoxes are an equation mark
def isEquationMark(boundingBox, boundingBox1):
    (x, y), (xw, yh) = boundingBox
    (x1, y1), (xw1, yh1) = boundingBox1
    return isHorizontalBar(boundingBox) and isHorizontalBar(boundingBox1) and abs(x1 - x) < 20  # 20 is a migical number

# detect if input three boundingBoxes are a ellipsis (three dots)
def isDots(boundingBox, boundingBox1, boundingBox2):
    (x, y), (xw, yh) = boundingBox
    (x1, y1), (xw1, yh1) = boundingBox1
    (x2, y2), (xw2, yh2) = boundingBox2
    return (isDot(boundingBox) and isDot(boundingBox1) and isDot(boundingBox2) and abs(max(yh, yh1, yh2, y, y1, y2) - min(yh, yh1, yh2, y, y1, y2)) < 20)  # 20 is a migical number

# return raw bounding boxes of input image
def rawBoundingBoxes(im):
    '''input: image; return: raw rectangles as list'''
    imgrey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgrey, 127, 255, 0)
    temp, contours, hierachy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    res = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        res.append([(x,y), (x+w, y+h)])
    return res

# take in raw bounding boxes and detect components should be connected
def connect(im, res):
    '''input: image, raw rectangles; return: joint rectangles indicating detected symbols'''
    finalRes = []
    res.sort()
    i = 0
    while (i < len(res) - 1):
        (x, y), (xw, yh) = res[i]
        (x1, y1), (xw1, yh1) = res[i+1]
        print([(x, y), (xw, yh)], [(x1, y1), (xw1, yh1)])

        equation = isEquationMark(res[i],  res[i + 1])
        # this one needs to refine, and use isSquare()
        division = (abs(x1 - x) < min(abs(xw - x) / 2, abs(xw1 - x1) / 2)
                    and abs(y1 - yh) < max(abs(yh - y) / 2, abs(yh1 - y1) / 2)
                    and (yh - y) / (xw - x) > 0.5 and (yh1 - y1) / (xw1 - x1) > 0.5)
        letterI = isLetterI(res[i], res[i+1])
        divisionMark = False
        dots = False
        # hard to be decided by bounding box
        # will try
        pm = False
        if i < len(res) - 2:
            (x2, y2), (xw2, yh2) = res[i+2]
            print([(x2, y2), (xw2, yh2)])
            divisionMark = isDivisionMark(res[i], res[i+1], res[i+2])
            dots = isDots(res[i], res[i+1], res[i+2])

        if (equation or letterI) and not division:
            finalRes.append([(min(x, x1), min(y, y1)), (max(xw, xw1), max(yh, yh1))])
            i += 2
        elif (divisionMark or dots) and not division:
            finalRes.append([(min(x, x1, x2), min(y, y1, y2)), (max(xw, xw1, xw2), max(yh, yh1, yh2))])
            i += 3
        else:
            finalRes.append(res[i])
            i += 1

    while i < len(res):
        print([res[i][0], res[i][1]])
        finalRes.append(res[i])
        i += 1

    return finalRes

# draw bounding boxes on image
def drawBoxes(im, boxes):
    ''' draw boxes on im'''
    for (left, right) in boxes:
        print([left, right])
        cv2.rectangle(im, left, right, (0,255,0), 2)

# remove noises of input image
def removeNoise(im):
    '''remove noises of input image; return: None'''

# return initial bounding boxes of input image
def initialBoxes(im):
    '''input: image; return: None'''
    # create grey image for retrieving contours
    # convert the image into white and black image
    im[im >= 127] = 255
    im[im < 127] = 0
    # set the morphology kernel size, the number in tuple is the bold pixel size
    kernel = np.ones((4,4),np.uint8)
    im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
    # create grey image for retrieving contours
    imgrey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgrey, 127, 255, 0)
    contours, hierachy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # RETR_EXTERNAL for only bounding outer box
    # bounding rectangle outside the individual element in image
    res = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        # exclude the whole size image and noisy point
        if x == 0: continue
        if w*h < 25: continue
        res.append([(x,y), (x+w, y+h)])
    return res


def saveImages(im, boxes):
    # make a tmpelate image for next crop
    image = Image.fromarray(im)
    num = 1
    boxes = sorted(boxes, key=lambda box: (box[1][1]-box[0][1]) * (box[1][0]-box[0][0]))
    for box in boxes:
        (x, y), (xw, yh) = box
        x -= 1
        y -= 1
        xw += 1
        yh += 1
        # save rectangled element
        symbolImage = image.crop((x, y, xw, yh))
        symbolImage.save("./testResults/" + "test" + "_" + str(num) + "_" + str(y) + "_" + str(yh) + "_" + str(x) + "_" + str(xw) + ".png")
        # fill the found part with black to reduce effect to other crop
        draw = ImageDraw.Draw(image)
        draw.rectangle((x, y, xw, yh), fill = 'black')
        # draw rectangle around element in image for confirming result
        cv2.rectangle(im, (x,y), (xw, yh), (0,255,0), 2)
        num = num + 1

# run the code
def main():
    image_list = glob.glob("./test-images/*.*")
    for im_name in image_list:
        im = cv2.imread(im_name)  # specify the image to process
        rawRes =  initialBoxes(im)  # raw bounding boxes
        finalRes = connect(im, rawRes)  # connect i, division mark, equation mark, ellipsis
        drawBoxes(im, finalRes)  # draw finalRes on im for debug
        result = Image.fromarray(im)  # show image
        head, tail = os.path.split(im_name)
        result.save("./testResult/" + tail)
        #saveImages(im, finalRes)
        #Image.fromarray(im).show()  # show image

if __name__ == "__main__":
    main()