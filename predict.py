import sys
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from PIL import Image, ImageFilter
import os
import pickle
import boundingBox
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
        #print ("Model restored.")
        nf = open("result.txt", 'w')
        tfile = open("test.pkl","rb")
        updated_nf = open("updated_result.txt", 'w')
        data = pickle.load(tfile)

        number = 0
        hit = 0
        for test_data in data["images"]:
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
                updated_nf.write("\t%s\t[%d, %d, %d, %d]\n" %(updated_symbol[1], updated_symbol[2], updated_symbol[3], updated_symbol[4], updated_symbol[5])) # write the result
                
            equation = toLatex(updated_symbol_list)
            updated_nf.write("%s\n" %(equation)) # write the result
            
        nf.close()

        print ("see result is in result.txt")
        print ("Accuracy is ", (hit/float(number)))

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


def update(im_name, symbol_list):
    im = Image.open(im_name)
    list_len = len(symbol_list)
    for i in range(list_len):
        if i >= len(symbol_list): break
        
        symbol = symbol_list[i]
        predict_result = symbol[1]
        
        # deal with equal mark
        if predict_result == "-":
            if i < (len(symbol_list) - 1):
                s1 = symbol_list[i+1]
                if s1[1] == "-" and ((s1[2] - symbol[2]) < 20 or (s1[4] - symbol[4]) < 20):
                    new_x = min(symbol[2], s1[2])
                    new_xw = max(symbol[3], s1[3])
                    new_y = min(symbol[4], s1[4])
                    new_yh = max(symbol[5], s1[5])
                    new_symbol = (im.crop((new_x, new_y, new_xw, new_yh)), "=", new_x, new_y, new_xw, new_yh)
                    symbol_list[i] = new_symbol
                    symbol_list.pop(i+1)
                    continue
        
        # deal with division mark
        if predict_result == "-":
            if i < (len(symbol_list) - 2):
                s1 = symbol_list[i+1]
                s2 = symbol_list[i+2] 
                if s1[3] < symbol[3] and s2[3] > symbol[3] and (s2[2] - s1[2]) < 30 and (area(s1) < 1600 or area(s2) < 1600):
                    new_x = min(symbol[2], s1[2], s2[2])
                    new_xw = max(symbol[3], s1[3], s2[3])
                    new_y = min(symbol[4], s1[4], s2[4])
                    new_yh = max(symbol[5], s1[5], s2[5])
                    new_symbol = (im.crop((new_x, new_y, new_xw, new_yh)), "div", new_x, new_y, new_xw, new_yh)
                    symbol_list[i] = new_symbol
                    symbol_list.pop(i+2)
                    symbol_list.pop(i+1)
                    continue
        
        # deal with dots
        if predict_result == "dot":
            if i < (len(symbol_list) - 2):
                s1 = symbol_list[i+1]
                s2 = symbol_list[i+2]
                if symbol_list[i+1][1] == "dot" and symbol_list[i+2][1] == "dot":
                    new_x = min(symbol[2], s1[2], s2[2])
                    new_xw = max(symbol[3], s1[3], s2[3])
                    new_y = min(symbol[4], s1[4], s2[4])
                    new_yh = max(symbol[5], s1[5], s2[5])
                    new_symbol = (im.crop((new_x, new_y, new_xw, new_yh)), "dots", new_x, new_y, new_xw, new_yh)
                    symbol_list[i] = new_symbol
                    symbol_list.pop(i+2)
                    symbol_list.pop(i+1)
                    continue
        
    return symbol_list

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