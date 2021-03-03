"""
Credit to Western digital Corporation
https://github.com/westerndigitalcorporation/YOLOv3-in-PyTorch
Copyright (c) 2019 Western Digital Corporation or its affiliates. All rights reserved

"""
import torch
import torch.nn.functional as F


def YoloLoss(preds, target, target_len, img_size):
    """
    Calculate the loss function given the prediction, the targets, and target length and image size

    Args:
        preds (tensor): the raw prediction tensor. Size is [Batch size, total number of predictions , Number attribues]
        target(tensor): the ground trouths. Size is [Batch size, Max number of targets in batch, Number attribues]
        target_len(tensor): a 1D tensor showing the number of the targets for each sample. Size is [Batch size, ...]
        img_size(int) : the size of input image

    Returns:
        total_loss : total loss value

    """
    #Generate the no-objectness mask. mask has size of [Batch size, total number of predictions]
    mask = mask_fn(preds, target)
    target1D, pred_index = preprocess(target, target_len, img_size)
    mask = mask_filter(mask, pred_index)

    #Calculate the no-objectness loss
    pred_conf = preds[...,4]
    target_zero = torch.zeros(pred_conf.size(), device = pred_conf.device)
    pred_conf = pred_conf - (1-mask)*1e7
    mask_loss = F.binary_cross_entropy_with_logits(pred_conf, target_zero, reduction='sum')

    #Select the predictions corresponding to the targets
    batch_size, n_preds, _ = preds.size()
    preds_1d = preds.view(batch_size * n_preds, -1)
    preds_obj= preds_1d.index_select(0, pred_index)

    #Calculate the coordinate loss
    coor_loss = F.mse_loss(preds_obj[..., :4], target1D[..., :4], reduction ='sum')

    #Calculate the objectness loss
    pred_conf_obj = preds_obj[..., 4]
    target_1 = torch.ones(pred_conf_obj.size(), device = pred_conf_obj.device)
    obj_loss = F.binary_cross_entropy_with_logits(pred_conf_obj, target_1, reduction='sum')

    #Calculate the classification loss
    class_loss = F.binary_cross_entropy_with_logits(preds_obj[..., 5:], target1D[..., 5:], reduction='sum')

    #Total lose
    #Mask loss coeffient and coordinate losee coeffient are the hyperparameters for the lose function
    #which is described in YOLOv1 paper which is 0.2 and 5 
    total_loss = mask_loss * 0.2 + obj_loss + class_loss + coor_loss * 5

    return total_loss



    


def mask_fn(preds, target):
    """
    Generate the no-objectness mask indicator

    Args:
        preds (tensor): the raw prediction tensor. Size is [Batch size, total number of predictions , Number attribues]
        target(tensor): the ground trouths. Size is [Batch size, Max number of targets in batch, Number attribues]

    Returns:
        indicator

    """
    num_batch, num_pred, num_attrib = preds.size()
    ious = IOU_batch(preds[...,:4], target[...,:4], center=True) #in cxcywh format
    #for each prediction bbox, find the target box with most IOU value
    max_ious, max_ious_index = torch.max(ious, dim=2)
    # if the max iou between the raw detection and all the ground truth is larger than 0.5 (hyperparameters), 
    # then we consider this detection will not contribute to the loss function.
    indicator = torch.where((max_ious - 0.5) > 0, torch.zeros_like(max_ious), torch.ones_like(max_ious))
    
    return indicator

def mask_filter(mask_indicator, obj_index):
    """
    Generate mask filter
    Args:
        mask_indicator (tensor): value from mask_fn function
        obj_index (tensor): value from preprocessing

    Returns:
        mask

    """
    batch_size, n_pred = mask_indicator.size()
    mask_indicator = mask_indicator.view(-1)
    filter_ = torch.zeros(mask_indicator.size(), device = mask_indicator.device)
    mask_indicator.scatter_(0, obj_index, filter_)
    mask_indicator = mask_indicator.view(batch_size, -1)
    return mask_indicator


def IOU_batch(bboxes1, bboxes2, center=False, zero_center=False):
    """
    Calculate the IOU's between bboxes1 and bboxes 2

    Args:
        bboxes1 (tensor): A 3D tensor representing the first group of boxes. Size is [Batch size, number of bboxes in the sampel, 4]
        bboxes2 (tensor): A 3D tensor representing the second group of boxes. size is same as bboxes1
        4 means x,y,w,h or cx, cy, w, h
        zero_center : zero_center mode on and off

    Returns:
        iou: A 3D tensor represetning IOU

    """
    x1 = bboxes1[..., 0]
    y1 = bboxes1[..., 1]
    w1 = bboxes1[..., 2]
    h1 = bboxes1[..., 3]

    x2 = bboxes2[..., 0]
    y2 = bboxes2[..., 1]
    w2 = bboxes2[..., 2]
    h2 = bboxes2[..., 3]

    area1 = w1 * h1
    area2 = w2 * h2

    if zero_center:
        w1.unsqueeze_(2)
        w2.unsqueeze_(1)
        h1.unsqueeze_(2)
        h2.unsqueeze_(1)
        w_intersect = torch.min(w1, w2).clamp(min =0)
        h_intersect = torch.min(h1, h2).clamp(min =0)

    else:
        if center:#center mode
            x1 = x1-w1/2
            y1 = y1-h1/2
            x2 = x2-w2/2
            y2 = y2-h2/2
        right1 = (x1+w1).unsqueeze(2)
        right2 = (x2+w2).unsqueeze(1)
        top1 = (y1+h1).unsqueeze(2)
        top2 = (y2+h2).unsqueeze(1)
        left1 = x1.unsqueeze(2)
        left2 = x2.unsqueeze(1)
        bot1 = y1.unsqueeze(2)
        bot2 = y2.unsqueeze(1)

        w_intersect = (torch.min(right1, right2) - torch.max(left1, left2)).clamp(min=0)
        h_intersect = (torch.min(top1, top2) - torch.max(bot1, bot2)).clamp(min=0)

    aoi = w_intersect * h_intersect

    iou = aoi / (area1.unsqueeze(2) + area2.unsqueeze(1) - aoi + 1e-9) # 1e-9 to avoid 0 Division

    return iou

def preprocess(target, target_length, img_size):
    """
    Get the index of the predictions corresponding to the targets
    and put targets from different sample into 1D and convert coordiantes to local.

    Args:
        target(tensor): the ground trouths. Size is [Batch size, Max number of targets in batch, Number attribues]
        target_len(tensor): a 1D tensor showing the number of the targets for each sample. Size is [Batch size, ...]
        img_size(int) : the size of input image

    Returns:
        target_flat (tensor): the 1D and local target. Size ie [total number of targets in the batch, Number attribues]
        indices (tensor): the tensor of the indeicies of the predictions corresponding to the targets. 
                            Size is [total number of targets in the batch, ...]
    
    """

    #Find the anchor box which has max IOU (zero-centered)  with the targets
    ANCHORS = [(10,13),(16,30),(33,23),(30,61),(62,45),(59,119),(116,90),(156,198),(373,326)]
    wh_anchor = torch.tensor(ANCHORS).to(target.device).float()
    n_anchor = wh_anchor.size(0)
    xy_anchor = torch.zeros((n_anchor, 2), device = target.device)
    bbox_anchor = torch.cat((xy_anchor, wh_anchor), dim=1)
    bbox_anchor.unsqueeze_(0)
    iou_anchor_target = IOU_batch(bbox_anchor, target[..., :4], zero_center = True)
    _, anchor_index = torch.max(iou_anchor_target, dim=1)

    #Find the corresponding prediction's index for the anchor box with the max IOU
    strides_choices = [8, 16, 32]
    scale = anchor_index //3
    anchor_index_scale = anchor_index - scale*3
    stride = 8 * 2 ** scale
    grid_x = (target[..., 0] // stride.float()).long()
    grid_y = (target[..., 1] // stride.float()).long()
    
    n_grid = img_size // stride

    obj_index = (scale <= 1).long() * (img_size//strides_choices[2])**2*3 + (scale <= 0).long() * (img_size//strides_choices[1])**2*3 + n_grid**2 + anchor_index_scale + n_grid*grid_y +grid_x

    #Calculate target_x and target_y
    t_x = (target[..., 0] / stride.float() - grid_x.float()).clamp(1e-9 , 1-1e-9) #1e-9 to avoid NaN
    t_x = torch.log(t_x / (1.0 - t_x))          #inverse of sigmoid
    t_y = (target[..., 1] /stride.float() - grid_y.float()).clamp(1e-9, 1-1e-9)
    t_y = torch.log(t_y/ (1.0 - t_y))

    #Calculate target_w and target_h
    w_anchor = wh_anchor[...,0]
    w_anchor = torch.index_select(w_anchor, 0, anchor_index.view(-1)).view(anchor_index.size())
    h_anchor = wh_anchor[...,1]
    h_anchor = torch.index_select(h_anchor, 0, anchor_index.view(-1)).view(anchor_index.size())
    
    t_w = torch.log((target[...,2] / w_anchor).clamp(min= 1e-9)) #1e-9 to avoid NaN. Inverse of exp = log
    t_h = torch.log((target[...,3] / h_anchor).clamp(min= 1e-9))

    #The raw target tensor
    target_t = target.clone().detach()

    target_t[..., 0] = t_x
    target_t[..., 1] = t_y
    target_t[..., 2] = t_w
    target_t[..., 3] = t_h

    #targets and the corresponding prediction index from idfferent batches in to 1D
    batch_size = target.size(0)
    n_pred  = sum([(img_size//s)**2 for s in strides_choices]) *3

    obj_index_1d = []
    target_t_1d =[]

    for batch in range(batch_size):
        v = obj_index[batch]
        t = target_t[batch]
        l = target_length[batch]
        obj_index_1d.append(v[:l] + batch*n_pred)
        target_t_1d.append(t[:l])

    obj_index_1d = torch.cat(obj_index_1d)
    target_t_1d = torch.cat(target_t_1d)

    return target_t_1d, obj_index_1d



