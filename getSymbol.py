import pickle
from PIL import Image
import os

data_train_root = os.getcwd() + "/data/train/"
data_test_root = os.getcwd() + "/data/test/"

def gettrainsymbol():
    pfile = open("train.pkl","rb")
    lfile = open(data_train_root + "lable.txt", "wb")
    data_train = pickle.load(pfile)
    images_train = data_train["images"]
    label_train = data_train["labels"]

    for f in images_train:
        image = images_train[f]
        # im = Image.frombytes(**image)
        # im.save(data_train_root+f)
        image.save(data_test_root+f)
        lfile.write(f+ " "+ str(label_train[f])+"\n")

    lfile.close()

def gettestsymbol():
    pfile = open("test.pkl","rb")
    lfile = open(data_test_root + "lable.txt", "wb")
    data_test = pickle.load(pfile)
    images_test = data_test["images"]
    label_test = data_test["labels"]

    for f in images_test:
        image = images_test[f]
        # im = Image.frombytes(**image)
        image.save(data_test_root+f)
        # im.save(data_test_root+f)
        lfile.write(f+ " "+ str(label_test[f])+"\n")

    lfile.close()

def main():
    gettrainsymbol()
    gettestsymbol()

if __name__ == "__main__":
    main()