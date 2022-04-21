import os
import numpy
import pprint

pp = pprint.PrettyPrinter(indent=4)
eq = {}

def marklabel(f):
   ins = f.split('.')[0].split('_')
   if ins[2] in eq:
       if len(ins) > 3:
           eq[ins[2]][ins[4]+"_"+ins[5]+"_"+ins[6]+"_"+ins[7]] = ins[3]
   else:
       eq[ins[2]] = {}


def main():
    dataroot = os.getcwd() + "/data/annotated"
    total = 0
    for f in os.listdir(dataroot):
        if f.endswith(".png"):
            marklabel(f)
            total = total + 1
    print (total)
    pp.pprint (eq)

if __name__ == "__main__":
    main()