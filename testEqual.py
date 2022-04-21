import os
import cv2
from PIL import Image

class TestEqual:
    # def _init_(self):

    def getEqual(self):
        dataroot = os.getcwd() + "/data/annotated/"
        saveroot = os.getcwd() + "/data/TestEqual/"
        for f in os.listdir(dataroot):
            if f.endswith(".png"):
                ins = f.split('.')[0].split('_')
                if len(ins) <= 3: # exclude the equation png only individual symbol
                    im = Image.open(dataroot + f)
                    im.save(saveroot + f)

def main():
    x = TestEqual()
    x.getEqual()


if __name__ == "__main__":
    main()