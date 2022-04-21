import os
import pickle
import pprint
import random
from PIL import Image, ImageFilter
import cv2
import numpy as np

sy = ['dots', 'tan', ')', '(', '+', '-', 'sqrt', '1', '0', '3', '2', '4', '6', 'mul', 'pi', '=', 'sin', 'pm', 'A',
'frac', 'cos', 'delta', 'a', 'c', 'b', 'bar', 'd', 'f', 'i', 'h', 'k', 'm', 'o', 'n', 'p', 's', 't', 'y', 'x', 'div']
rules = {}
# brules = {}
lst = [float(0)] * 40
for i in range(0,len(sy)):
    lst[i] = float(1)
    rules[sy[i]] = lst[:]
    # brules[i] = sy[i]
    lst[i] = float(0)
# print rules
#later we can do some merge rules: 0 and o, frac and bar and -, x and mul
#rules['o'] = rules['0']
#rules['frac'] = rules['-']
#rules['bar'] = rules['-']
#rules['x'] = rules['mul']

pp = pprint.PrettyPrinter(indent=4)
dataroot = os.getcwd() + "/data/annotated/"
symbol_test = {}
images_test = {}
data_test = {} #store symbol and images dic
symbol_train = {}
images_train = {}
data_train = {} #store symbol and images dic
symbol = {}
images = {}
data = {} #store symbol and images dic

def processImage(image_name):
    im = cv2.imread(dataroot +image_name)
    im[im >= 127] = 255
    im[im < 127] = 0
    image = Image.fromarray(im)

    head, tail = os.path.split(image_name)

    width = float(image.size[0])
    height = float(image.size[1])
    newImage = Image.new('L', (28, 28), (0))

    if width > height: #check which dimension is bigger
        #Width is bigger. Width becomes 20 pixels.
        nheight = int(round((28.0/width*height),0)) #resize height according to ratio width
        if (nheight == 0): #rare case but minimum is 1 pixel
            nheight = 1
        # resize and sharpen
        img = image.resize((28,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight)/2),0)) #caculate horizontal pozition
        newImage.paste(img, (0, wtop)) #paste resized image on white canvas
    else:
        #Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((28.0/height*width),0)) #resize width according to ratio height
        if (nwidth == 0): #rare case but minimum is 1 pixel
            nwidth = 1
         # resize and sharpen
        img = image.resize((nwidth,28), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth)/2),0)) #caculate vertical pozition
        newImage.paste(img, (wleft, 0)) #paste resized image on white canvas

    # newImage.save("./annotated_28x28/"+tail, quality=100)
    tv = list(newImage.getdata())
    tva = [x * 1.0/255.0 for x in tv]
    return tva

def marklabel(f):
   ins = f.split('.')[0].split('_')
   if len(ins) > 3: # exclude the equation png only individual symbol
    #    symbol[f] = ins[3]

         decider = random.randint(0,9)
         if decider >= 3:
             symbol_train[f] = rules[ins[3]]
             symbol[f] = symbol_train[f]
             image = processImage(f)
            #  image = np.reshape(image, (784))
             images_train[f] = image
             images[f] = images_train[f]
         else:
             symbol_test[f] = rules[ins[3]]
             symbol[f] = symbol_test[f]
            #  im = Image.open(dataroot + f)
            #  image = {
            #     'data':im.tobytes(),
            #     'size':im.size,
            #     'mode':im.mode
            #  }
             image = processImage(f)
            #  image = np.reshape(image, (784))
             images_test[f] = image
             images[f] = images_test[f]

def main():

    total = 0

    for f in os.listdir(dataroot):
        if f.endswith(".png"):
            marklabel(f)
            total = total + 1
    # print total
    # print (set(symbol.values()))
    data_train["images"] = images_train
    data_train["labels"] = symbol_train
    data_test["images"] = images_test
    data_test["labels"] = symbol_test
    data["images"] = images
    data["labels"] = symbol

    pfile = open('train.pkl','wb')
    pickle.dump(data_train, pfile)
    pfile.close()
    pfile = open('test.pkl','wb')
    pickle.dump(data_test, pfile)
    pfile.close()
    pfile = open('data.pkl','wb')
    pickle.dump(data, pfile)
    pfile.close()

if __name__ == "__main__":
    main()