# Handwritten-Equation-Recognition

# Overview
This project implement deep learning on handwriting mathmetic equation recognition using (CNN) convolutional neural network.

# Package PrerequisitesPytoh
1. OpenCV
<br>install OpenCV: https://pypi.python.org/pypi/opencv-python
<br>For Mac:
      brew tap homebrew/science<br>
      brew install opencv<br>
      export PYTHONPATH=/usr/local/lib/python2.7/site-packages:$PYTHONPATH<br>
1. Numpy
1. Scipy
1. PIL (Pillow)


preSymbol.py : to label each picture (exclude equation) and divide it into train and test data for models. its format: "images" -> "file_name" : its value(picture) "labels" -> "file_name" : its label (integer decided by rules)

getSymbol.py : get pictures for train and test data and their labels