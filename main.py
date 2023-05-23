# Main Inference Script
import onnx
import numpy as np
import onnxruntime as ort
import cv2
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import os
import json

with open('Utils/synset.txt', 'r') as f:
    labels = [l.rstrip() for l in f]

model_path = 'Models/model.onnx'
model = onnx.load(model_path)
img_directory = 'SampleData/Input/'
session = ort.InferenceSession(model.SerializeToString())

def get_image(path, show=False):
    print(path)
    img = cv2.imread(path)
    return img


def preprocess(img):
    img = img / 255.    
    img = rgb2gray(img)
    img = img.reshape(1,img.shape[0]*img.shape[1])
    return img

def predict(directory):
    # iterate over files in the image directory
    for filename in os.listdir(directory):
        # Split and get the file names
        filename_split = filename.split('.')
        input_image_name = filename_split[0]
        print(input_image_name)
        path = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(path):
            img = get_image(path, show=True)
            # BBox of the image
            x_min = 0
            y_min = 0
            x_max = img.shape[1]
            y_max = img.shape[0]
            
            # output
            output_json = {
                "prediction": [
                    {
                        "class_id": "",
                        "score": "",
                        "xmin": "",
                        "ymin": "",
                        "xmax": "",
                        "ymax": ""
                    }
                ]
            }

            img = preprocess(img)
            input_name = session.get_inputs()[0].name
            ort_inputs = {input_name: img.astype(np.float32)}            
            preds = session.run(None, ort_inputs)[0]
            preds = np.squeeze(preds)
            num_classes =10
            preds = preds/num_classes
            a = np.argsort(preds)[::-1]
            # Store the output of each file in the Runtime folder in the format <image-name-out>
            file1 = open('Runtime/' + input_image_name + "-out.json","a")
            # Modify the output json file
            output_json['prediction'][0]['class_id'] = str(labels[a[0]])
            output_json['prediction'][0]['score'] = str(preds[a[0]])
            output_json['prediction'][0]['xmin'] = x_min
            output_json['prediction'][0]['ymin'] = y_min
            output_json['prediction'][0]['xmax'] = x_max
            output_json['prediction'][0]['ymax'] = y_max

            file1.write(json.dumps(output_json))
            file1.close()
            # print('class=%s ; probability=%f' % (labels[a[0]], preds[a[0]]))

# Enter path to the inference image directory
predict(img_directory)
