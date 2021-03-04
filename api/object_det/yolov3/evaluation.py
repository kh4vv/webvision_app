import sys
import os
import glob
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

from . models.yolov3 import YoloV3


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def yolov3_evaluation(img, weight_path, class_name, filename):

    model = YoloV3()
    model.load_state_dict(torch.load(weight_path))
    model.to(device)
    model.eval()
    dataset = ImageFolder(img, img_size=416)

    results = run_detection(model, dataset, device, 0.8, 0.4)
    print("done run detection")
    for img_i, result in enumerate(results):
        detections, _, _ = result
        img = draw_result(img, detections, class_names=class_name)
        img.save(filename)
    print("image saved")


class ImageFolder(Dataset):
    """The ImageFolder Dataset class."""

    def __init__(self, image, img_size=416, sort_key=None):
        self.img = image
        self.img_shape = (img_size, img_size)
        self._img_size = img_size
        self._transform = default_transform(img_size)

    def __getitem__(self, index):
        img = self.img
        w, h = img.size
        max_size = max(w, h)
        _padding = _get_padding(h, w)
        transformed_img_tensor, _ = self._transform(img)
        transformed_img_tensor = torch.unsqueeze(transformed_img_tensor, dim=0)
        scale = self._img_size / max_size
        return transformed_img_tensor, scale, np.array(_padding)

    def __len__(self):
        return len(self.files)


def _get_padding(h, w):
    """Generate the size of the padding given the size of the image,
    such that the padded image will be square.
    Args:
        h (int): the height of the image.
        w (int): the width of the image.
    Return:
        A tuple of size 4 indicating the size of the padding in 4 directions:
        left, top, right, bottom. This is to match torchvision.transforms.Pad's parameters.
        For details, see:
            https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.Pad
        """
    dim_diff = np.abs(h - w)
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    return (0, pad1, 0, pad2) if h <= w else (pad1, 0, pad2, 0)


def default_transform(img_size):
    return ComposeWithLabel([PadToSquareWithLabel(fill=(127, 127, 127)), ResizeWithLabel(img_size), tv_tf.ToTensor()])


class ComposeWithLabel(tv_tf.Compose):

    def __call__(self, img, label=None):
        import inspect
        for t in self.transforms:
            num_param = len(inspect.signature(t).parameters)
            if num_param == 2:
                img, label = t(img, label)
            elif num_param == 1:
                img = t(img)
        return img, label


class PadToSquareWithLabel(object):
    """
    Pad to square the given PIL Image with label.
    Args:
        fill (int or tuple): Pixel fill value for constant fill. Default = 0.If tuple of length 3, its RGB channels
        padding_mode: constant, edge, reflect or symmetric. Default is constant
            -constant: pads with a constant value
            -edge: pads with the last value at the edge of the image
            -reflect: pads with reflection of image without repeating the last value on the edge
            -symmetric: pads with reflection of image with repeating the last value on the edge

    """

    def __init__(self, fill=0, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        self.fill = fill
        self.padding_mode = padding_mode

    def __init__(self, fill=0, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        self.fill = fill
        self.padding_mode = padding_mode

    @ staticmethod
    def _get_padding(w, h):
        """
        Generate the size of the padding given the size of the image
        Args:
            w (int): the height of the image
            h (int): the width of the image

        Returns:
            A tuple of size 4 indicating (left, top, right, bottom)

        """
        dim_diff = np.abs(w-h)
        pad1, pad2 = dim_diff//2, dim_diff-dim_diff//2

        return(0, pad1, 0, pad2) if h <= w else (pad1, 0, pad2, 0)

    def __call__(self, img, label=None):
        w, h = img.size
        padding = self._get_padding(w, h)
        img = TF.pad(img, padding, self.fill, self.padding_mode)
        if label is None:
            return img, label
        label[..., 0] += padding[0]
        label[..., 1] += padding[1]
        return img, label


class ResizeWithLabel(tv_tf.Resize):

    def __init__(self, size, interpolation=Image.BILINEAR):
        super().__init__(size, interpolation)

    def __call__(self, img, label=None):
        w_old, h_old = img.size
        img = super(ResizeWithLabel, self).__call__(img)
        w_new, h_new = img.size
        if label is None:
            return img, label
        scale_w = w_new/w_old
        scale_h = h_new/h_old
        label[..., 0] *= scale_w
        label[..., 1] *= scale_h
        label[..., 2] *= scale_w
        label[..., 3] *= scale_h
        return img, label


def grouping_class(classes, num_class):
    """
    group the object with the same class into a list.
    Args:
        classes (tensor): classes tensor
        num_class (int): number of classes

    Returns:
        group_index : a list of group index

    """
    group_index = [[] for _ in range(num_class)]

    for index, class_ in enumerate(classes):
        group_index[torch.argmax(class_)].append(index)

    return group_index


def group_same_class_object(obj_classes, one_hot=True, num_classes=-1):
    """
    Given a list of class results, group the object with the same class into a list.
    Returns a list with the length of num_classes, where each bucket has the objects with the same class.
    :param
        obj_classes: (torch.tensor) The representation of classes of object.
         It can be either one-hot or label (non-one-hot).
         If it is one-hot, the shape should be: [num_objects, num_classes]
         If it is label (non-non-hot), the shape should be: [num_objects, ]
        one_hot: (bool) A flag telling the function whether obj_classes is one-hot representation.
        num_classes: (int) The max number of classes if obj_classes is represented as non-one-hot format.
    :return:
        a list of of a list, where for the i-th list,
        the elements in such list represent the indices of the objects in class i.
    """
    if one_hot:
        num_classes = obj_classes.shape[-1]
    else:
        assert num_classes != -1
    grouped_index = [[] for _ in range(num_classes)]
    if one_hot:
        for idx, class_one_hot in enumerate(obj_classes):
            grouped_index[torch.argmax(class_one_hot)].append(idx)
    else:
        for idx, obj_class_ in enumerate(obj_classes):
            grouped_index[obj_class_].append(idx)
    return grouped_index


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


def NMS(bboxes, scores, classes, num_class, conf_thres, nms_thres, center=False):
    """
    Apply non-max suppression to avoid overlapping bounding boxes

    Args:
        bboxes (tensor) : bounding boxes coordinates. shape:[num_priors, 4]
        scores (tensor) : class prediction score. shape: [num_priors]
        classes (tensor): predictions of each classes. shape: [num_priors]
        num_class (int): number of classes
        conf_thres (float) : confidence threshold
        nms_thres (float) : nms threshold
        center (boolean) : either cxcywh or xywh

    Returns:
        bboxes_result, score_result, class_result:the indeices of the survived boxes with respect to num_priors, and xywh format

    """
    num_prior = bboxes.shape[0]

    # if no objects, return raw result
    if num_prior == 0:
        return bboxes, scores, classes

    # Apply threshold
    if conf_thres > 0:
        conf_index = torch.nonzero(torch.ge(scores, conf_thres)).squeeze()

        bboxes = bboxes.index_select(0, conf_index)
        scores = scores.index_select(0, conf_index)
        classes = classes.index_select(0, conf_index)

    # If more than one class, divide them into groups
    # group_indices = grouping_(classes, num_class)
    group_indices = group_same_class_object(
        classes, one_hot=False, num_classes=num_class)
    final_indices = []

    for class_id, index in enumerate(group_indices):
        index_tensor = bboxes.new_tensor(index, dtype=torch.long)
        bboxes_class = bboxes.index_select(dim=0, index=index_tensor)
        scores_class = scores.index_select(dim=0, index=index_tensor)
        socres_class, sorted_indices = torch.sort(
            scores_class, descending=False)

        selected_indices = []

        while sorted_indices.size(0) != 0:
            index_ = sorted_indices[-1]
            selected_indices.append(index_)
            bbox = bboxes_class[index_]
            ious = IOU(bbox, bboxes_class[sorted_indices[:-1]], center=center)

            index__ = torch.nonzero(ious <= nms_thres).squeeze()
            sorted_indices = sorted_indices.index_select(dim=0, index=index__)
        final_indices.extend([index[i] for i in selected_indices])

    final_indices = bboxes.new_tensor(final_indices, dtype=torch.long)
    bboxes_result = bboxes.index_select(dim=0, index=final_indices)
    scores_result = scores.index_select(dim=0, index=final_indices)
    classes_result = classes.index_select(dim=0, index=final_indices)

    return bboxes_result, scores_result, classes_result


def IOU(bbox1, bbox2, center=False):
    """
    Calculate IOU for bbox1 with another group of bboxes (bbox2)

    """
    x1, y1, w1, h1 = bbox1
    x2 = bbox2[..., 0]
    y2 = bbox2[..., 1]
    w2 = bbox2[..., 2]
    h2 = bbox2[..., 3]

    if center:
        x1 = x1 - w1/2
        y1 = y1 - h1/2
        x2 = x2 - w2/2
        y2 = y2 - h2/2

    area1 = w1 * h1
    area2 = w2 * h2
    right1 = x1 + w1
    right2 = x2 + w2
    bot1 = y1 + h1
    bot2 = y2 + h2

    w_intersect = (torch.min(right1, right2) - torch.max(x1, x2)).clamp(min=0)
    h_intersect = (torch.min(bot1, bot2) - torch.max(y1, y2)).clamp(min=0)
    aoi = w_intersect * h_intersect
    iou = aoi / (area1 + area2 - aoi + 1e-10)  # 1e-10 to avoid 0 division.
    return iou


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
