import os
import numbers
import numpy as np
import cv2
from PIL import Image, ImageDraw
from albumentations.pytorch import ToTensorV2
import albumentations as A

import torch
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms as tv_tf

from .models.yolov3 import YoloV3
from .dataset.dataset import COCODataset
from .utils.utils import NMS, IOU


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def yolov3_evaluation(img, weight_path, class_name, filename):

    model = YoloV3()
    model.load_state_dict(torch.load(weight_path))
    model.to(device)
    model.eval()
    dataset = COCODataset(img, img_size=416)

    results = run_detection(model, dataset, device, 0.8, 0.4)
    print("done run detection")
    for img_i, result in enumerate(results):
        detections, _, _ = result
        img = draw_result(img, detections, class_names=class_name)
        img.save(os.path.join('outputs/', filename))
    print("image saved")


def run_detection(model, dataloader, device, conf_thres, nms_thres):
    """
    Run detection

    Args:
        model : Yolo v3 model
        dataloader :
        device : 'cuda' if gpu is available
        conf_thres: confidence threshold
        num_thres: NMS threshold
        num_class: number of classes

    Returns:
        results : detection result

    """
    results = []
    detection_time_list = []
    num_class = 80

    for index, batch in enumerate(dataloader):
        img_batch = batch[0].to(device)
        scales = batch[1]
        scales = torch.tensor([scales], dtype=float)
        scales = scales.to(device)
        paddings = batch[2]
        paddings = torch.from_numpy(paddings)
        paddings = torch.unsqueeze(paddings, dim=0)
        paddings = paddings.to(device)
        # Get detection
        with torch.no_grad():
            detections = model(img_batch)
        detections = post_process(num_class, detections, conf_thres, nms_thres)
        for detection, scale, padding in zip(detections, scales, paddings):
            # Transform the bbox from the scaled image back to the unsclaed image
            detection[..., :4] = untransform_box(
                detection[..., :4], scale, padding)
            # cxcywh to xywh
            detection[..., 0] -= detection[..., 2]/2
            detection[..., 1] -= detection[..., 3]/2

        results.extend(zip(detections, scales, padding))
        print(results)
        # return results
        break
    return results


def untransform_box(bboxes, scale, padding):
    """
    transform the bbox from the scaled image back to the unsclaed image
    """
    x = bboxes[..., 0]
    y = bboxes[..., 1]
    w = bboxes[..., 2]
    h = bboxes[..., 3]
    x /= scale
    y /= scale
    w /= scale
    h /= scale
    x -= padding[0]
    y -= padding[1]
    return bboxes


def post_process(num_class, result, conf_thres, nms_thres):
    """
    post process result values with confidence and nms thresholds

    Args:
        result : result raw data
        conf_thres: confidence threshold
        nms_thres : nms threshold

    Returns:
        results : result after processing
    """
    results = []
    for index, raw in enumerate(result):
        bboxes = raw[..., :4]
        scores = raw[..., 4]
        classes = raw[..., 5:]
        classes = torch.argmax(classes, dim=1)

        bboxes, scores, classes = NMS(
            bboxes, scores, classes, num_class, conf_thres, nms_thres, center=True)

        results_ = torch.cat((bboxes, scores.view(-1, 1),
                              classes.view(-1, 1).float()), dim=1)
        results.append(results_)
    return results


def draw_result(image, boxes, show=False, class_names=None):
    """
    Draw bounding boxes and labels of detections

    """
    if isinstance(image, torch.Tensor):
        transform = ToPILImage()
        image = transform(image)

    draw = ImageDraw.ImageDraw(image)
    show_class = (boxes.size(1) >= 6)

    for box in boxes:
        x, y, w, h = box[:4]
        x2 = x+w
        y2 = y+h
        draw.rectangle([x, y, x2, y2], outline='white', width=3)
        if show_class:
            class_id = int(box[5])
            class_name = class_names[class_id]
            font_size = 20
            text_size = draw.textsize(class_name)
            draw.rectangle([x, y-text_size[1]*2, x +
                            text_size[0], y], fill='white')
            draw.text([x, y-font_size], class_name, fill='black')
    if show:
        image.show()

    return image
