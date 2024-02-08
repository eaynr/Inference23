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
  output_data = interpreter.get_tensor(output_details['index'])
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
      '-c', '--count', type=int, default=5,
      help='Number of times to run inference')
  args = parser.parse_args()

  interpreter = make_interpreter(args.model)
  interpreter.allocate_tensors()

  pathing = "imgs/"
  imgs = ["img_40.png", "img_41.png", "img_42.png", "img_43.png", "img_44.png", "img_45.png", "img_46.png", "img_47.png", "img_48.png", "img_49.png"]
  
  for img in imgs:
    #img
    path=pathing + img
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
  
    start = time.perf_counter()
    interpreter.invoke()
    inference_time = time.perf_counter() - start
    result = get_output(interpreter)

    print(img)
    print(result)
    print("--Inference time--")
    print(inference_time)

if __name__ == '__main__':
  main()