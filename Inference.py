import json
import sys
import os
import time
import numpy as np
import onnx
import onnxruntime
from onnx import numpy_helper
import cv2

import time

model="model_regression_peritonitis.onnx"
pathing = "imgs/"
path=pathing + sys.argv[1]

#read image https://nvidia.box.com/shared/static/amhb62mzes4flhbfavoa73m5z933pv75.whl
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]) #GREYSCALE

#Adjust dynamic range
min_brt = 0
max_brt = 65000
img = np.where(img < min_brt, min_brt, img)
img = np.where(img > max_brt, max_brt, img)

#NORMALITZATION
img = img.astype(np.float32)
img -= min_brt
img = img / (max_brt - min_brt)
img *= 255
#img = cv2.normalize(img, None, 0, 1.0,cv2.NORM_MINMAX, dtype=cv2.CV_32F)

#ROI
subImg = img[800:1000, 100:660]

#SOBEL
subImg_filt = cv2.Sobel(subImg, cv2.CV_64F, 1, 0, None, 3, 1, 0)

# Normalize the image
subImg_filt_norm = cv2.normalize(subImg_filt, None, 0, 1.0,cv2.NORM_MINMAX, dtype=cv2.CV_32F)

#Ajust to model input parameters
subImg_filt_norm.resize(1,1,200,560)

#Inferencia
session = onnxruntime.InferenceSession(model, None)#, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])

#['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']

#session.set_providers('CPUExecutionProvider', 'CUDAExecutionProvider', 'TensorrtExecutionProvider')
input_name = {session.get_inputs()[0].name: subImg_filt_norm}

# get the start time
st = time.time()
result = session.run(None, input_name)

# get the end time
et = time.time()

#knowdevice
#print(onnxruntime.get_available_providers())
#print(onnxruntime.get_device())

# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')

print(result)

