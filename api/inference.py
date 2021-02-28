import sys
import glob

from PIL import Image
import numpy as np
import cv2

import torch
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    ])

transform_landmark = A.Compose([
    A.Resize(300, 300),
    A.Normalize(
        mean=(0.4452, 0.4457, 0.4464), 
        std=(0.2592, 0.2596, 0.2600)),
    ToTensorV2(),
])

def mnist_evaluation(inputs, weight_path, model):
    inputs = transform(inputs)
    inputs = torch.unsqueeze(inputs, dim=0)
    # inputs = inputs.view(1,1,28,28)
    # inputs = inputs[:,-1,:,:]
    model.load_state_dict(torch.load(weight_path, map_location=device)) #CPU Comparable. 
    model.eval()
    with torch.no_grad():
        inputs = inputs.to(device)

        preds = model(inputs)
        preds = torch.argmax(preds, 1)

    return inputs, preds


def quickdraw_evaluation(inputs, weight_path, model):
    inputs = transform(inputs)
    inputs = torch.unsqueeze(inputs, dim=0)
    # inputs = inputs.view(1,1,28,28)
    # inputs = inputs[:,-1,:,:]
    model.load_state_dict(torch.load(weight_path, map_location=device)) #CPU Comparable. 
    model.eval()
    with torch.no_grad():
        inputs = inputs.to(device)

        preds = model(inputs)
        preds = torch.argmax(preds, 1)

    return inputs, preds


def class_dict_extraction(path, fileformat):
    #Look for all files with specific format and put it in as dictionary
    files = sorted(glob.glob(path + '/*.' +fileformat))

    #Add files in dictionary
    dic = ClassDict()
    for key, value in enumerate(files):       
        #Get rid of the path and print label only
        label = value[len(path)+1 : -len(fileformat)-1]
        dic.add(key, label)
    return dic

class ClassDict(dict):
    def __init__(self):
        self = dict()

    def add(self, key, value):
        self[key] = value


def landmark_evaluation(inputs, weight_path, model):
    inputs = np.array(inputs)
    inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB)

    augmented = transform_landmark(image=inputs)
    inputs = augmented['image']
    
    inputs = torch.unsqueeze(inputs, dim=0)
    model.load_state_dict(torch.load(weight_path, map_location=device))

    model.eval()
    with torch.no_grad():
        inputs = inputs.to(device)

        preds = model(inputs)
        preds = torch.argmax(preds, 1)

    return inputs, preds
