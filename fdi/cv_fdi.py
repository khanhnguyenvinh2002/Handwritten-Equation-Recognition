import numpy as np
import cv2

from PIL import Image, ImageDraw

# clipt image into box using OpenCV

# can almost combine all symbols except very bad hand writings
# TA input size: 1696 * 117

# detect if input boundingBox contains a dot
def isDot(boundingBox):
    (x, y), (xw, yh) = boundingBox
    area = (yh - y) * (xw - x)
    return area < 850 and 0.5 < (xw - x)/(yh - y) < 2 and abs(xw - x) < 40 and abs(yh - y) < 40  # 100 is migical number

# detect if input boundingBox contains a vertical bar
def isVerticalBar(boundingBox):
    (x, y), (xw, yh) = boundingBox
    return (yh - y) / (xw - x) > 1.5

# detect if a given boundingBox contains a horizontal bar
def isHorizontalBar(boundingBox):
    (x, y), (xw, yh) = boundingBox
    return (xw - x) / (yh - y) > 1.5

# detect if input boundingBox contains a square (regular letters, numbers, operators)
def isSquare(boundingBox):
    (x, y), (xw, yh) = boundingBox
    return (xw - x) > 8 and (yh - y) > 8 and (0.5 < (xw - x)/(yh - y) < 2 or 0.5 < (yh - y)/(xw - x) < 2)

# detect if input three boundingBoxes are a division mark
def isDivisionMark(boundingBox, boundingBox1, boundingBox2):
    boundingBox, boundingBox1, boundingBox2 = sorted([boundingBox, boundingBox1, boundingBox2], key = lambda box: box[0][0])
    (x, y), (xw, yh) = boundingBox
    (x1, y1), (xw1, yh1) = boundingBox1
    (x2, y2), (xw2, yh2) = boundingBox2
    cenX1 = min(x1, xw1) + abs(xw1 - x1) / 2
    cenX2 = min(x2, xw2) + abs(xw2 - x2) / 2
    cenY1 = min(y1, yh1) + abs(yh1 - y1) / 2
    cenY2 = min(y2, yh2) + abs(yh2 - y2) / 2
    caseBase = isHorizontalBar(boundingBox) and isDot(boundingBox1) and isDot(boundingBox2)
    #caseBase1 = isHorizontalBar(boundingBox) and (xw1 - x1) < abs(xw - x)/2 and (xw2 - x2) < abs(xw - x)/2
    caseRelation = x < x1 < xw1 < xw and x < x2 < xw2 < xw  #and max(cenY1, cenY2) > yh and min(cenY1, cenY2) < y
    #caseDistance = max(cenY1, cenY2) - min(cenY1, cenY2) < 1.5 * abs(yh - y)
    return caseBase and caseRelation #and caseDistance

# detect if input two boundingBoxes are a lowercase i
def isLetterI(boundingBox, boundingBox1):
    (x, y), (xw, yh) = boundingBox
    (x1, y1), (xw1, yh1) = boundingBox1
    return (((isDot(boundingBox) and isVerticalBar(boundingBox1)) or (isDot(boundingBox1) and isVerticalBar(boundingBox)))
            and abs(x1 - x) < 30)  # 10 is a magical number

# detect if input two boundingBoxes are an equation mark
def isEquationMark(boundingBox, boundingBox1):
    (x, y), (xw, yh) = boundingBox
    (x1, y1), (xw1, yh1) = boundingBox1
    cenX = x + (xw - x) / 2
    cenX1 = x1 + (xw1 - x1) / 2
    case1 = x < cenX1 < xw
    case2 = x1 < cenX < xw1
    return isHorizontalBar(boundingBox) and isHorizontalBar(boundingBox1) and (case1 and case2)

# detect if input three boundingBoxes are a ellipsis (three dots)
def isDots(boundingBox, boundingBox1, boundingBox2):
    (x, y), (xw, yh) = boundingBox
    (x1, y1), (xw1, yh1) = boundingBox1
    (x2, y2), (xw2, yh2) = boundingBox2
    cenY = y + (yh - y) / 2
    cenY1 = y1 + (yh1 - y1) / 2
    cenY2 = y2 + (yh2 - y2) / 2
    caseBase = isDot(boundingBox) and isDot(boundingBox1) and isDot(boundingBox2)
    return caseBase and max(cenY, cenY1, cenY2) - min(cenY, cenY1, cenY2) < 50  # 30 is a migical number

# detect if input two boundingBoxes are a plus-minus
def isPM(boundingBox, boundingBox1):
    (x, y), (xw, yh) = boundingBox
    (x1, y1), (xw1, yh1) = boundingBox1
    cenX = x + (xw - x) / 2
    cenX1 = x1 + (xw1 - x1) / 2
    case1 = isHorizontalBar(boundingBox) and isSquare(boundingBox1) and x < cenX1 < xw and -15 < y - yh1 < 35 and xw - cenX1 < 50
    case2 = isSquare(boundingBox) and isHorizontalBar(boundingBox1) and x1 < cenX < xw1 and -15 < y1 - yh < 35 and xw1 - cenX < 50
    return case1 or case2  # magical number

# detect if input three boundingBoxes are a fraction
def isFraction(boundingBox, boundingBox1, boundingBox2):
    (x, y), (xw, yh) = boundingBox
    (x1, y1), (xw1, yh1) = boundingBox1
    (x2, y2), (xw2, yh2) = boundingBox2
    cenX = x + (xw - x) / 2
    cenX1 = x1 + (xw1 - x1) / 2
    cenX2 = x2 + (xw2 - x2) / 2
    case1 = isSquare(boundingBox) and isSquare(boundingBox1) and isHorizontalBar(boundingBox2) and (y < y2 < yh1 or y1 < y2 < yh)
    case2 = isSquare(boundingBox2) and isSquare(boundingBox) and isHorizontalBar(boundingBox1) and (y2 < y1 < yh or y < y1 < yh2)
    case3 = isSquare(boundingBox1) and isSquare(boundingBox2) and isHorizontalBar(boundingBox) and (y1 < y < yh2 or y2 < y < yh1)
    return (case1 or case2 or case3) and  max(cenX, cenX1, cenX2) - min(cenX, cenX1, cenX2) < 50  # 30 is a migical number

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
    kernel = np.ones((2,2),np.uint8)
    im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
    # create grey image for retrieving contours
    imgrey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgrey, 127, 255, 0)
    im2, contours, hierachy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # RETR_EXTERNAL for only bounding outer box
    # bounding rectangle outside the individual element in image
    res = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        # exclude the whole size image and noisy point
        if x is 0: continue
        if w*h < 25: continue
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
        letterI = isLetterI(res[i], res[i+1])
        pm = isPM(res[i], res[i+1])
        divisionMark = False
        dots = False
        fraction = False
        if i < len(res) - 2:
            (x2, y2), (xw2, yh2) = res[i+2]
            print([(x2, y2), (xw2, yh2)])
            divisionMark = isDivisionMark(res[i], res[i+1], res[i+2])
            dots = isDots(res[i], res[i+1], res[i+2])
            fraction = isFraction(res[i], res[i+1], res[i+2])

        # PM os really hard to determine, mixed with fraction
        if (divisionMark or dots) and not fraction:
            finalRes.append([(min(x, x1, x2), min(y, y1, y2)), (max(xw, xw1, xw2), max(yh, yh1, yh2))])
            i += 3
        elif (equation or letterI or pm) and not fraction:
            finalRes.append([(min(x, x1), min(y, y1)), (max(xw, xw1), max(yh, yh1))])
            i += 2
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
    '''draw boxes on im; return: None'''
    for (left, right) in boxes:
        print([left, right])
        cv2.rectangle(im, left, right, (0,255,0), 2)

# slices im into smaller images based on boxes
def saveImages(im, boxes):
    '''input: image, boxes; return: None'''
    # make a tmpelate image for next crop
    image = Image.fromarray(im)
    num = 1
    boxes = sorted(boxes, key = lambda box: (box[1][1]-box[0][1]) * (box[1][0]-box[0][0]))
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
    #'''
    im = cv2.imread("SKMBT_36317040717260_eq13.png")
    rawRes =  initialBoxes(im)
    finalRes = connect(im, rawRes)
    drawBoxes(im, finalRes)
    Image.fromarray(im).show()
    '''
    image_list = glob.glob("./special/*.*")
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
    '''

if __name__ == "__main__":
    main()