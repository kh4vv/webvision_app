import os
import numbers
import numpy as np
from PIL import Image

import torch
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset
from torchvision import transforms as tv_tf


class COCODataset(Dataset):
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
