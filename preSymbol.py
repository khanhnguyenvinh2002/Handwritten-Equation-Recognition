import os
import pickle
from PIL import Image
import pprint
import random

sy = ['dots', 'tan', ')', '(', '+', '-', 'sqrt', '1', '0', '3', '2', '4', '6', 'mul', 'pi', '=', 'sin', 'pm', 'A',
'frac', 'cos', 'delta', 'a', 'c', 'b', 'bar', 'd', 'f', 'i', 'h', 'k', 'm', 'o', 'n', 'p', 's', 't', 'y', 'x', 'div']
rules = {}
brules = {}

for i in range(0,len(sy)):
    rules[sy[i]] = i
    brules[i] = sy[i]

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

def marklabel(f):
   ins = f.split('.')[0].split('_')
   if len(ins) > 3: # exclude the equation png only individual symbol
    #    symbol[f] = ins[3]

         decider = random.randint(0,9)
         if decider >= 3:
             symbol_train[f] = rules[ins[3]]
             symbol[f] = symbol_train[f]
             im = Image.open(dataroot + f)
             image = {
                'data':im.tobytes(),
                'size':im.size,
                'mode':im.mode
             }
             images_train[f] = image
             images[f] = images_train[f]
         else:
             symbol_test[f] = rules[ins[3]]
             symbol[f] = symbol_test[f]
             im = Image.open(dataroot + f)
             image = {
                'data':im.tobytes(),
                'size':im.size,
                'mode':im.mode
             }
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