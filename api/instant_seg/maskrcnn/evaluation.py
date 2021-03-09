import os

from random import randint
import numpy as np

from PIL import Image, ImageFont, ImageDraw
# import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
from skimage.measure import find_contours

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

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

for clss in COCO_CLASSES.keys():
    r = randint(0,200)
    g = randint(0,200)
    b = randint(0,200)
    COLOR_MAP[clss] = ((r, g, b, 10), (r, g, b, 200))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def maskrcnn_evaluation(img, filename):
    model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model = model.to(device)

    model.eval()

    img_input = transforms.ToTensor()(img)
    img_input = img_input.unsqueeze(0)
    img_input = img_input.to(device)
    torch.save(img_input,"img_test_re")
    outputs = model(img_input)[0]

    scores = outputs['scores'].cpu().detach().numpy()
    boxes = outputs['boxes'].cpu().detach().numpy()
    masks = outputs['masks'].cpu().detach().numpy()
    labels = outputs['labels'].cpu().detach().numpy()

    background = Image.new('RGBA', img.size)
    background.paste(img)
    poly = Image.new('RGBA', img.size)
    font_size = 18

    for score, box, mask, label in zip(scores, boxes, masks, labels):
        if score > 0.6:
            padded_mask = np.where(mask[0] > 0.5, 1, 0)
            contour = find_contours(padded_mask, 0.1)
            contour = contour[0]
            contour = np.flip(contour, axis=1)
            contour = contour.flatten().tolist()

            print(COLOR_MAP[label])

            draw = ImageDraw.Draw(poly)
            draw.polygon(contour, fill=COLOR_MAP[label][0], outline=COLOR_MAP[label][1])
            background.paste(poly, mask=poly)
            
            text_size = draw.textsize(COCO_CLASSES[label])
            draw.rectangle([box[0], box[1]-text_size[1]*2, 
                box[0] + text_size[0], box[1]], fill=COLOR_MAP[label][1])
            draw.text([box[0], box[1]-font_size], COCO_CLASSES[label], fill='white')

            draw = ImageDraw.Draw(background)

    background = background.convert('RGB')
    background.save(os.path.join('outputs/', filename))

    box_list = boxes.tolist()
    label_list = labels.tolist()
    score_list = scores.tolist()
    mask_list = masks.tolist()
    result = {}
    #print(box_list)
    result = {"boxes": box_list, "labels": label_list, "scores": score_list, "masks": mask_list}
    return result
    
