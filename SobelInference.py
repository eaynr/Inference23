import json
import sys
import os
import time
import numpy as np
import onnx
import onnxruntime
from onnx import numpy_helper
import cv2

model="model_regression_peritonitis.onnx"
pathing = "imgs/after_sobel_filter/"
path=pathing + sys.argv[1]

#read image
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# Normalize the image
img = cv2.normalize(img, None, 0, 1.0,cv2.NORM_MINMAX, dtype=cv2.CV_32F)

#Ajust to model input parameters
img.resize(1,1,200,560)

session = onnxruntime.InferenceSession(model, None,)# providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
input_name = {session.get_inputs()[0].name: img}

result = session.run(None, input_name)

print(result)