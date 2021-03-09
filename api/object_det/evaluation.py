import os
from random import randint
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from matplotlib.collections import PatchCollection
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
from PIL import Image, ImageDraw
from skimage.measure import find_contours
from torchvision import datasets, models, transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from matplotlib.figure import Figure

COCO_CLASSES = {1: 'person',
2: 'bicycle',
3: 'car',
4: 'motorcycle',
5: 'airplane',
6: 'bus',
7: 'train',
8: 'truck',
9: 'boat',
10:	'traffic light',
11:	'fire hydrant',
12:	'street sign',
13:	'stop sign',
14:	'parking meter',
15:	'bench',
16:	'bird',
17:	'cat',
18:	'dog',
19:	'horse',
20:	'sheep',
21:	'cow',
22:	'elephant',
23:	'bear',
24:	'zebra',
25:	'giraffe',
26:	'hat',
27:	'backpack',
28:	'umbrella',
29:	'shoe',
30:	'eye glasses',
31:	'handbag',
32:	'tie',
33:	'suitcase',
34:	'frisbee',
35:	'skis',
36:	'snowboard',
37:	'sports ball',
38:	'kite',
39:	'baseball bat',
40: 'baseball glove',
41:	'skateboard',
42:	'surfboard',
43:	'tennis racket',
44:	'bottle',
45:	'plate',
46:	'wine glass',
47:	'cup',
48:	'fork',
49:	'knife',
50:	'spoon',
51:	'bowl',
52:	'banana',
53:	'apple',
54:	'sandwich',
55:	'orange',
56:	'broccoli',
57:	'carrot',
58:	'hot dog',
59:	'pizza',
60:	'donut',
61:	'cake',
62:	'chair',
63:	'couch',
64:	'potted plant',
65:	'bed',
66:	'mirror',
67:	'dining table',
68:	'window',
69:	'desk',
70:	'toilet',
71:	'door',
72:	'tv',
73:	'laptop',
74:	'mouse',
75:	'remote',
76:	'keyboard',
77:	'cell phone',
78:	'microwave',
79:	'oven',
80:	'toaster',
81:	'sink',
82:	'refrigerator',
83:	'blender',
84:	'book',
85:	'clock',
86:	'vase',
87:	'scissors',
88:	'teddy bear',
89:	'hair drier',
90:	'toothbrush',
91:	'hair brush'}

COLOR_MAP = {}
# class map color assign
for clss in COCO_CLASSES.keys():
    COLOR_MAP[clss] = (randint(0,200), randint(0,200), randint(0,200))

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

model.eval()

def frcnn_evaluation(img, filename):
    
    img_input = transforms.ToTensor()(img)
    img_input = img_input.unsqueeze(0)
    img_input = img_input.to(device)
    outputs = model(img_input)[0]

    scores = outputs['scores'].cpu().detach().numpy()
    boxes = outputs['boxes'].cpu().detach().numpy()
    labels = outputs['labels'].cpu().detach().numpy()
    font_size = 18

    for score, box, label in zip(scores, boxes, labels):
        if score > 0.6:
            #print(score, box, mask, label)
            draw = ImageDraw.Draw(img)
            draw.rectangle(box, outline = COLOR_MAP[label])
            #Text
            text_size = draw.textsize(COCO_CLASSES[label])
            draw.rectangle([box[0], box[1]-text_size[1]*2, box[0]+text_size[0], box[1]], fill = COLOR_MAP[label])
            draw.text([box[0], box[1]-font_size], COCO_CLASSES[label], fill='white')

    img.save(os.path.join('outputs/',"frcnn"+filename))
    new_filename = "frcnn"+filename
    return new_filename


