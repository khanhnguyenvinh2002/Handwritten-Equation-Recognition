from PIL import Image
import cv2

# clipt image into box using OpenCV

'''
equation = (yh - y) / (xw - x) < 0.2 and (yh1 - y1) / (xw1 - x1) < 0.2 and abs(x1 - x) < 20
division = (abs(x1 - x) < min(abs(xw - x) / 2, abs(xw1 - x1) / 2)
                and abs(y1 - yh) < max(abs(yh - y) / 2, abs(yh1 - y1) / 2)
                and (yh - y) / (xw - x) > 0.5 and (yh1 - y1) / (xw1 - x1) > 0.5)
letterI = (((yh - y) / (xw - x) > 5 and (yh1 - y1) / (xw1 - x1) < 2)
               or ((yh - y) / (xw - x) < 2 and (yh1 - y1) / (xw1 - x1) > 5) and abs(x1 - x) < 2)
divisionMark = False
if i < len(res) - 2:
    (x2, y2), (xw2, yh2) = res[i + 2]
    divisionMark = ((yh - y) / (xw - x) < 0.2 and 0.7 < (yh1 - y1) / (xw1 - x1) < 1.3
                    and 0.7 < (yh2 - y2) / (xw2 - x2) < 1.3 and x < x1 < x2 < xw and max(y1, y2) > y
                    and min(y1, y2) < y and max(y1, y2) - min(y1, y2) < 1.2 * abs(xw - x))
'''

# 下一步进一步优化要用TA给的大小尺寸
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
    return (isDot(boundingBox) and isDot(boundingBox1) and isDot(boundingBox2)
            and abs(max(yh, yh1, yh2, y, y1, y2) - min(yh, yh1, yh2, y, y1, y2)) < 20)  # 20 is a migical number

# return raw bounding boxes of input image
def rawBoundingBoxes(im):
    '''input: image; return: raw rectangles'''
    imgrey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgrey, 127, 255, 0)
    temp, contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
        # this one needs to refind, and use isSquare()
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

# draw boxes on image
def drawBoxes(im, boxes):
    ''' draw boxes on im'''
    for (left, right) in boxes:
        print([left, right])
        cv2.rectangle(im, left, right, (0,255,0), 2)

# run the code
def main():
    im = cv2.imread("./test_1.png")  # specify the image to process
    rawRes = rawBoundingBoxes(im)
    finalRes = connect(im, rawRes)
    drawBoxes(im, finalRes)
    Image.fromarray(im).show()  # show image
    #result.save("test.png")

if __name__ == "__main__":
    main()