import boundingBox
import sys
import tensorflow as tf
from PIL import Image, ImageFilter
import os
import pickle
import glob
import pprint
import operator

sy = ['dots', 'tan', ')', '(', '+', '-', 'sqrt', '1', '0', '3', '2', '4', '6', 'mul', 'pi', '=', 'sin', 'pm', 'A',
'frac', 'cos', 'delta', 'a', 'c', 'b', 'bar', 'd', 'f', 'i', 'h', 'k', 'm', 'o', 'n', 'p', 's', 't', 'y', 'x', 'div']
slash_sy = ['tan', 'sqrt', 'mul', 'pi', 'sin', 'pm', 'frac', 'cos', 'delta', 'bar', 'div','^','_']
brules = {}
for i in range(0,len(sy)):
    brules[i] = sy[i]

def predictint():
    # Define the model (same as when creating the model file)
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 40]))
    b = tf.Variable(tf.zeros([40]))

    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1,28,28,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 40])
    b_fc2 = bias_variable([40])

    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    init_op = tf.initialize_all_variables()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, os.getcwd()+"/model.ckpt")
        print ("Model restored.")
        nf = open("result.txt", 'w')
        updated_nf = open("updated_result.txt", 'w')

        number = 0
        hit = 0

        test_equal_path = "./data/annotated_test_Equal/"
        test_data_path = "./data/annotated_test_Equal_boxes/"
        result_data_path = "./data/annotated_test_result_boxes/"
        test_data_list = glob.glob(test_equal_path+ '/*.*')

        for test_data in test_data_list:
            nf.write("predict for equation %s\n" %(test_data)) # write the result
            updated_nf.write("predict for equation %s\n" %(test_data)) # write the result
            test_symbol_list = boundingBox.createSymbol(test_data)

            test_symbol_list = sorted(test_symbol_list, key=operator.itemgetter(2, 3))
            for i in range(len(test_symbol_list)):
                test_symbol = test_symbol_list[i]
                imvalue, image = imageprepare(test_symbol[0])
                prediction = tf.argmax(y_conv, 1)
                predint = prediction.eval(feed_dict={x: [imvalue], keep_prob: 1.0}, session=sess)
                if test_symbol[1] != "dot":
                    predict_result = brules[predint[0]]
                else:
                    predict_result = "dot"
                test_symbol = (test_symbol[0], predict_result, test_symbol[2], test_symbol[3], test_symbol[4], test_symbol[5])
                test_symbol_list[i] = test_symbol
                nf.write("\t%s\t[%d, %d, %d, %d]\n" %(test_symbol[1], test_symbol[2], test_symbol[3], test_symbol[4], test_symbol[5])) # write the result

            updated_symbol_list = update(test_data, test_symbol_list)
            for updated_symbol in updated_symbol_list:
                updated_nf.write("\t%s\t[%d, %d, %d, %d]\n" %(updated_symbol[0], updated_symbol[1], updated_symbol[2], updated_symbol[3], updated_symbol[4])) # write the result

            equation = toLatex(updated_symbol_list)
            updated_nf.write("%s\n" %(equation)) # write the result

        nf.close()

        print ("see result is in result.txt")
        print ("see result is in updated_result.txt")
#        print "Accuracy is ", (hit/float(number))

def isDot(boundingBox):
    (x, y), (xw, yh) = boundingBox
    area = (yh - y) * (xw - x)
    return area < 850 and 0.5 < (xw - x)/(yh - y) < 2 and abs(xw - x) < 40 and abs(yh - y) < 40  # 100 is migical number

def imageprepare(image):
    im = image.convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (0)) #creates black canvas of 28x28 pixels

    if width > height: #check which dimension is bigger
        nheight = int(round((28.0/width*height),0)) #resize height according to ratio width
        img = im.resize((28,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight)/2),0)) #caculate horizontal pozition
        newImage.paste(img, (0, wtop)) #paste resized image
    else:
        nwidth = int(round((20.0/height*width),0)) #resize width according to ratio height
        img = im.resize((nwidth,20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth)/2),0)) #caculate vertical pozition
        newImage.paste(img, (wleft, 4)) #paste resized image on
    tv = list(newImage.getdata()) #get pixel values
    #normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [ 1-(255-x)*1.0/255.0 for x in tv]
    return tva, newImage


# detect if input three boundingBoxes are a division mark
def isDivisionMark(boundingBox, boundingBox1, boundingBox2, res, res1, res2):
    (x, y), (xw, yh) = boundingBox
    (x1, y1), (xw1, yh1) = boundingBox1
    (x2, y2), (xw2, yh2) = boundingBox2
    cenX1 = min(x1, xw1) + abs(xw1 - x1) / 2
    cenX2 = min(x2, xw2) + abs(xw2 - x2) / 2
    cenY1 = min(y1, yh1) + abs(yh1 - y1) / 2
    cenY2 = min(y2, yh2) + abs(yh2 - y2) / 2
    #caseBase = (res == '-' and res1 == 'dot' and res2 == 'dot')
    caseBase = (res == '-' and (isDot(boundingBox1) or res1 == 'dot') and (isDot(boundingBox2) or res2 == 'dot'))
    caseRelation = x < x1 < xw1 < xw and x < x2 < xw2 < xw # and max(cenY1, cenY2) > yh and min(cenY1, cenY2) < y
    #caseDistance = max(cenY1, cenY2) - min(cenY1, cenY2) < 1.5 * abs(yh - y)
    return caseBase and caseRelation # and caseDistance

# detect if input two boundingBoxes are a lowercase i
def isLetterI(boundingBox, boundingBox1, res, res1):
    (x, y), (xw, yh) = boundingBox
    (x1, y1), (xw1, yh1) = boundingBox1
    return (((isDot(boundingBox) and res1 == '1') or (isDot(boundingBox1) and res == '1'))
            and abs(x1 - x) < 30)  # 10 is a magical number

# detect if input two boundingBoxes are an equation mark
def isEquationMark(boundingBox, boundingBox1, res, res1):
    (x, y), (xw, yh) = boundingBox
    (x1, y1), (xw1, yh1) = boundingBox1
    cenX = x + (xw - x) / 2
    cenX1 = x1 + (xw1 - x1) / 2
    case1 = x < cenX1 < xw
    case2 = x1 < cenX < xw1
    return res == '-' and res1 == '-' and (case1 and case2)

# detect if input three boundingBoxes are a ellipsis (three dots)
def isDots(boundingBox, boundingBox1, boundingBox2, res, res1, res2):
    (x, y), (xw, yh) = boundingBox
    (x1, y1), (xw1, yh1) = boundingBox1
    (x2, y2), (xw2, yh2) = boundingBox2
    cenY = y + (yh - y) / 2
    cenY1 = y1 + (yh1 - y1) / 2
    cenY2 = y2 + (yh2 - y2) / 2
    caseBase = isDot(boundingBox) and isDot(boundingBox1) and isDot(boundingBox2)
    return caseBase and max(cenY, cenY1, cenY2) - min(cenY, cenY1, cenY2) < 50  # 30 is a migical number

# detect if input two boundingBoxes are a plus-minus
def isPM(boundingBox, boundingBox1, res, res1):
    (x, y), (xw, yh) = boundingBox
    (x1, y1), (xw1, yh1) = boundingBox1
    cenX = x + (xw - x) / 2
    cenX1 = x1 + (xw1 - x1) / 2
    case1 = res == '-' and res1 == '+' and x < cenX1 < xw and -15 < y - yh1 < 35 and xw - cenX1 < 50
    case2 = res == '+' and res1 == '-' and x1 < cenX < xw1 and -15 < y1 - yh < 35 and xw1 - cenX < 50
    return case1 or case2  # magical number


def update(im_name, symbol_list):
    # symbol = <PIL.Image.Image image mode=RGB size=79x69 at 0x115BB3CD0>, 'c', 1145, 46, 1224, 115
    # output: list of (res, x, y, xw, yh)
    im = Image.open(im_name)
    list_len = len(symbol_list)

    finalRes = []

    i = 0
    while (i < list_len):

        equation = False
        letterI = False
        pm = False
        divisionMark = False
        dots = False
        fraction = False

        symbol = symbol_list[i]
        res, x, y, xw, yh = symbol[1:]
        box = ((x, y), (xw, yh))

        if i < list_len - 1:
            symbol1 = symbol_list[i + 1]
            res1, x1, y1, xw1, yh1 = symbol1[1:]
            box1 = ((x1, y1), (xw1, yh1))
            equation = isEquationMark(box, box1, res, res1)
            letterI = isLetterI(box, box1, res, res1)
        if i < list_len - 2:
            symbol2 = symbol_list[i + 2]
            res2, x2, y2, xw2, yh2 = symbol2[1:]
            box2 = ((x2, y2), (xw2, yh2))
            divisionMark = isDivisionMark(box, box1, box2, res, res1, res2)
            dots = isDots(box, box1, box2, res, res1, res2)

        if (divisionMark or dots):
            if divisionMark:
                res = 'div'
            elif dots:
                res = 'dots'
            finalRes.append((res, min(x, x1, x2), min(y, y1, y2), max(xw, xw1, xw2), max(yh, yh1, yh2)))
            i += 3
        elif (equation or letterI or pm):
            if equation:
                res = '='
            elif letterI:
                res = 'i'
            elif pm:
                res = 'pm'
            finalRes.append((res, min(x, x1), min(y, y1), max(xw, xw1), max(yh, yh1)))
            i += 2
        else:
            finalRes.append((res, x, y, xw, yh))
            i += 1

    return finalRes

def toLatex(symbol_list):
    s = []
    for i in range(len(symbol_list)):
        symbol = symbol_list[i]
        value = symbol[1]

        if value in slash_sy:
            s.append('\\' + value)
        elif i > 0 and (s[len(s) - 1] in slash_sy):
            s.append('{'+value+'}')
        elif i < len(symbol_list) - 1 and isUpperSymbol(symbol, symbol_list[i+1]):
            s.append(value)
            s.append('^')
        elif i < len(symbol_list) - 1 and isLowerSymbol(symbol, symbol_list[i+1]):
            s.append(value)
            s.append('_')
        else:
            s.append(value)
    return "".join(s)

def isUpperSymbol(cur, next):
    cur_center = cur[3] + (cur[5] - cur[3])/2
    next_center = next[3] + (next[5] - next[3])/2
    cur_center_x = cur[2] + (cur[4] - cur[2])/2
    if next_center < cur_center - (next[5] - next[3])/2 and next[2] > cur_center_x: return True
    else: return False

def isLowerSymbol(cur, next):
    cur_center = cur[3] + (cur[5] - cur[3])/2
    next_center = next[3] + (next[5] - next[3])/2
    cur_center_x = cur[2] + (cur[4] - cur[2])/2
    if next_center > cur_center + (next[5] - next[3])/2 and next[2] > cur_center_x: return True
    else: return False

def area(symbol):
    return (symbol[4] - symbol[2]) * (symbol[5] - symbol[3])

def main():
    predint = predictint()

if __name__ == "__main__":
    main()