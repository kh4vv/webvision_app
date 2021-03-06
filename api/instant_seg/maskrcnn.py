from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

#device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def maskrcnn_evaluation(img, classmap, filename):
    #Pre-trained Model Fine Tuning
    model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model = model.to(device)
    #eval
    model.eval()

    img_input = transforms.ToTensor()(img)
    img_input = img_input.unsqueeze(0)
    img_input = img_input.to(device)

    output = model(img_input)
    result = {}
    #Tensor -> List
    box_list = output[0]['boxes'].tolist()
    label_list = output[0]['labels'].tolist()
    score_list = output[0]['scores'].tolist()
    mask_list = output[0]['masks'].tolist()
    result = {"boxes": box_list, "labels": label_list, "scores": score_list, "masks": mask_list}
    return result
    


