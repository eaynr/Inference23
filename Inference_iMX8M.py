# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Example using TF Lite to classify a given image using an Edge TPU.

   To run this code, you must attach an Edge TPU attached to the host and
   install the Edge TPU runtime (`libedgetpu.so`) and `tflite_runtime`. For
   device setup instructions, see g.co/coral/setup.

   Example usage (use `install_requirements.sh` to get these files):
   ```
   python3 classify_image.py \
     --model models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite  \
     --labels models/inat_bird_labels.txt \
     --input images/parrot.jpg
   ```
"""
import argparse
import time

import tflite_runtime.interpreter as tflite
import platform

import sys
import time
import numpy as np
import cv2

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

def make_interpreter(model_file):
  model_file, *device = model_file.split('@')
  return tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB,
                               {'device': device[0]} if device else {})
      ])

def input_details(interpreter, key):
  """Returns input details by specified key."""
  return interpreter.get_input_details()[0][key]

def input_tensor(interpreter):
  """Returns input tensor view as numpy array of shape (height, width, 3)."""
  tensor_index = input_details(interpreter, 'index')
  return interpreter.tensor(tensor_index)()[0]

def set_input(interpreter, data):
  """Copies data to input tensor."""
  input_tensor(interpreter)[:, :] = data

def output_tensor(interpreter, dequantize=True):
  output_details = interpreter.get_output_details()[0]
  output_data = np.squeeze(interpreter.tensor(output_details['index'])())
  return output_data

def get_output(interpreter):
  score = output_tensor(interpreter)
  return score

def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '-m', '--model', required=True, help='File path of .tflite file.')
  parser.add_argument(
      '-i', '--input', required=True, help='Image to be classified.')
  args = parser.parse_args()

  interpreter = make_interpreter(args.model)
  interpreter.allocate_tensors()

  model="model_regression_peritonitis.onnx"
  pathing = "imgs/"
  path=pathing + args.input
  
  #read image
  img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
  
  #Adjust dynamic range
  min_brt = 0
  max_brt = 65000
  img = np.where(img < min_brt, min_brt, img)
  img = np.where(img > max_brt, max_brt, img)
  
  #NORMALITZATION
  img = img.astype(np.float32)
  img -= min_brt
  img /= (max_brt - min_brt)
  img *= 255
  
  #ROI
  subImg = img[800:1000, 100:660]
  
  #SOBEL
  subImg_filt = cv2.Sobel(subImg, cv2.CV_64F, 1, 0, None, 3, 1, 0)
  
  # Normalize the image
  subImg_filt_norm = cv2.normalize(subImg_filt, None, 0, 1.0,cv2.NORM_MINMAX, dtype=cv2.CV_32F)
  
  #Ajust to model input parameters
  subImg_filt_norm.resize(1,1,200,560)

  set_input(interpreter, subImg_filt_norm)

  print('----INFERENCE TIME----')
  print('Note: The first inference on Edge TPU is slow because it includes',
        'loading the model into Edge TPU memory.')
  
  start = time.perf_counter()
  interpreter.invoke()
  inference_time = time.perf_counter() - start
  result = get_output(interpreter)
  print('%.1fms' % (inference_time * 1000))

  print('-------RESULTS--------')
  print(result)

if __name__ == '__main__':
  main()