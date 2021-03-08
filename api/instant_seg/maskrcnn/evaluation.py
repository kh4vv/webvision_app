import os

from random import randrange
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
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

def rgb_to_rgba(rgb, alpha = 0.3):
    rgb = np.asarray(rgb) / 255.
    c1 = np.insert(rgb, len(rgb), alpha).round(2)
    c2 = np.insert(rgb, len(rgb), 1.).round(2)
    return (tuple(c1), tuple(c2))

COLOR_MAP = {}

for clss in COCO_CLASSES.keys():
    COLOR_MAP[clss] = rgb_to_rgba((randrange(255), randrange(255), randrange(255)), 0.5)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def maskrcnn_evaluation(img, filename):
    model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model = model.to(device)

    model.eval()
    #img.show() 
    img_input = transforms.ToTensor()(img)
    img_input = img_input.unsqueeze(0)
    img_input = img_input.to(device)
    torch.save(img_input,"img_test_re")
    outputs = model(img_input)[0]

    scores = outputs['scores'].cpu().detach().numpy()
    boxes = outputs['boxes'].cpu().detach().numpy()
    masks = outputs['masks'].cpu().detach().numpy()
    labels = outputs['labels'].cpu().detach().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(11,5), dpi=300)

    for score, box, mask, label in zip(scores, boxes, masks, labels):
        #print(score, box, mask, label)
        if score > 0.6:
            padded_mask = np.where(mask[0] > 0.5, 1, 0)
            contour = find_contours(padded_mask, 0.1)
            contour = contour[0]
            contour = np.flip(contour, axis=1)
            polygon = Polygon(contour, fc=COLOR_MAP[label][0], ec=COLOR_MAP[label][1], lw=0.5)
            ax.add_patch(polygon)
            print(polygon)

            ax.annotate(COCO_CLASSES[label], (box[0], box[1]), color='w', weight='bold', 
                fontsize=3, ha='left', va='bottom', 
                bbox=dict(facecolor=COLOR_MAP[label][0], edgecolor=COLOR_MAP[label][1], pad=0.0))

    ax.axis('off')
    plt.savefig(filename)
    plt.show()

    box_list = boxes.tolist()
    label_list = labels.tolist()
    score_list = scores.tolist()
    mask_list = masks.tolist()
    result = {}
    #print(box_list)
    result = {"boxes": box_list, "labels": label_list, "scores": score_list, "masks": mask_list}
    return result
    


