import glob
import cv2
import os
from PIL import Image, ImageFilter

image_list = glob.glob("./train-symbol/*.*")

for image_item in image_list:
    im = cv2.imread(image_item)
    im[im >= 127] = 255
    im[im < 127] = 0
    image = Image.fromarray(im)
    
    head, tail = os.path.split(image_item)
    
    #read width of image
    width = float(image.size[0])
    #read height of image
    height = float(image.size[1])
    # create a new Image with size 28x28
    new_image = Image.new('L', (28, 28), (0))

    #check which dimension is bigger
    if width > height: 
        # width becomes 20 pixels.
        
        # resize height according to ratio width
        new_height = int(round((20.0/width*height),0)) 
        # edge case height is 0 => min 1
        if (new_height == 0):
            new_height = 1
        # resize and sharpen
        img = image.resize((20,new_height), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)

        #caculate horizontal pozition
        width_top = int(round(((28 - new_height)/2),0)) 

        #paste resized image on white canvas
        new_image.paste(img, (4, width_top)) 
    else:
        # Height is bigger. Heigth becomes 20 pixels.

        # resize width according to ratio height
        new_width = int(round((20.0/height*width),0)) 
        # edge case width is 0 => min 1
        if (new_width == 0): 
            new_width = 1
         # resize and sharpen
        img = image.resize((new_width,20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)

         #caculate vertical pozition
        width_left = int(round(((28 - new_width)/2),0))

         #paste resized image on white canvas
        new_image.paste(img, (width_left, 4))
        
    new_image.save("./annotated-20_28x28/"+tail, quality=100)