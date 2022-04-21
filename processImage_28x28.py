import glob
import cv2
import os
from PIL import Image, ImageFilter

image_list = glob.glob("data/annotated/*.*")

for image_name in image_list:
    im = cv2.imread(image_name)
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

    newImage.save("./annotated_28x28/"+tail, quality=100)