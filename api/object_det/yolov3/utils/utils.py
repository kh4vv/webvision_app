import torch


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