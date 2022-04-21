import sys
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from PIL import Image, ImageFilter
import os
import pickle

sy = ['dots', 'tan', ')', '(', '+', '-', 'sqrt', '1', '0', '3', '2', '4', '6', 'mul', 'pi', '=', 'sin', 'pm', 'A',
'frac', 'cos', 'delta', 'a', 'c', 'b', 'bar', 'd', 'f', 'i', 'h', 'k', 'm', 'o', 'n', 'p', 's', 't', 'y', 'x', 'div']
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
        nnfile = open("undesired.txt",'w')
        data = pickle.load(tfile)

        number = 0
        hit = 0
        for f in data["images"]:
            # print (fn)

            prediction=tf.argmax(y_conv,1)
            predint = prediction.eval(feed_dict={x: [data["images"][f]],keep_prob: 1.0}, session=sess)
            # print f
            # print brules[predint[0]]
            nf.write("%s\t%s\n" %(f,brules[predint[0]]))
            ins = f.split('.')[0].split('_')
            label = ins[3]
            if ins[3] == "o":
                label = "0"
            if ins[3] == "frac" or ins[3] == "bar":
                label = "-"
            if ins[3] == "mul":
                label = "x"
            if brules[predint[0]] == label:
                hit = hit +1
            else:
                nnfile.write("%s\t%s\n" %(f,brules[predint[0]]))
            number = number + 1
                # print f, (predint[0]) #first value in list
        nf.close()

        print ("see result is in result.txt")
        print ("Accuracy is ", (hit/float(number)))

# def imageprepare(argv):
#     im = Image.open(argv).convert('L')
#     width = float(im.size[0])
#     height = float(im.size[1])
#     newImage = Image.new('L', (28, 28), (0)) #creates black canvas of 28x28 pixels
#
#     if width > height: #check which dimension is bigger
#         nheight = int(round((20.0/width*height),0)) #resize height according to ratio width
#         img = im.resize((20,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
#         wtop = int(round(((28 - nheight)/2),0)) #caculate horizontal pozition
#         newImage.paste(img, (4, wtop)) #paste resized image
#     else:
#         nwidth = int(round((20.0/height*width),0)) #resize width according to ratio height
#         img = im.resize((nwidth,20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
#         wleft = int(round(((28 - nwidth)/2),0)) #caculate vertical pozition
#         newImage.paste(img, (wleft, 4)) #paste resized image on
#     tv = list(newImage.getdata()) #get pixel values
#     #normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
#     tva = [ 1-(255-x)*1.0/255.0 for x in tv]
#     return tva

def main():
    predint = predictint()

if __name__ == "__main__":
    main()