import torch
import torch.nn as nn
import torch.nn.functional as F
"""
Credit to Western digital Corporation
https://github.com/westerndigitalcorporation/YOLOv3-in-PyTorch
Copyright (c) 2019 Western Digital Corporation or its affiliates. All rights reserved

"""

class ConvoLayer(nn.Module):
    """
    Basic Conv2D layer with few parameters: channels, kernel size
    Then batch norm - layer and Leaky ReLu layer as follow
    Leaky ReLu has negative slope 0.1 as default

    """
    def __init__(self, channel_in, channel_out, kernel_size, stride = 1, neg_slope = 0.1):
        """
        Initalize the functions with various parameters

        Args:
            channel_in (int) : number of input - channels 
            channel_out(int) : number of output- channels
            kernel_size(int) : kernel size
            stride : 1 as default
            neg_slope: 0.1 as default
        
        """
        super().__init__()
        padding = (kernel_size-1)//2
        self.conv = nn.Conv2d(channel_in, channel_out, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(channel_out)
        self.lrelu = nn.LeakyReLU(neg_slope)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = self.lrelu(y)
        
        return y

class ResidulBlock(nn.Module):
    """
    Residual Block is consist with two convolutional layers and one residual layer 

    From YOLOv3: An Incremental Improvement (U of W) Paper,
    The first convolutional layers has half of the number of the filters as the second convolutional layers
    The First convolutional layers = 1 x 1 filter size and the Second convolutional layers = 3 x 3 filter size.
    
    """
    def __init__(self, channel_in):
        
        super().__init__()
        assert channel_in %2 == 0 # make sure channel_in is even number
        channel_half = channel_in//2
        self.conv1 =ConvoLayer(channel_in, channel_half, 1)
        self.conv2 =ConvoLayer(channel_half, channel_in, 3)

    def forward(self, x):
        residual = x
        y = self.conv1(x)
        y = self.conv2(y)
        y += residual
        return y

def DarkNetBlock(channel_in, channel_out, resblock_num):
    """
    In DarkNet Backbone, the format is usually : one convolutional layer + residual block + residual blocks + ...
    We can add these layer and blocks to make one bigger block called DarkNet Block.

    The one convolutional layer before residual block has size = 3 , stride =2

    Args:
        channel_in (int): number of input channels 
        channel_out (int): number of output channels
        resblock_num (int): how many residual blocks after one convolutional layer

    Returns:
        block: Sequential with all layers combined (convolutional layer + residual blocks)

    """
    block = nn.Sequential()
    block.add_module('conv', ConvoLayer(channel_in, channel_out, 3, stride=2))
    for i in range(resblock_num):
        block.add_module('res{}'.format(i), ResidulBlock(channel_out))
    return block

#Anchors list from Darknet Backbone
ANCHORS_LIST = [(10,13), (16,30), (33,23), (30,61), (62,45), (59,119), (116,90), (156, 198), (373, 326)]

class YoloLayer(nn.Module):
    """
    Create Yolo Layer from the Darknet Backbone
    Anchors list is writen above
    There are three masks: 0,1,2 and 3,4,5 and 6,7,8

    """
    def __init__(self, mask_num, stride):
        """
        Args:
            mask_num : you can choose it from 1,2,3 which is corresponding to (0,1,2) (3,4,5) and (6,7,8)
            class_num: number of classes (default: coco dataset = 80)
            stride : stride number
        """
        super().__init__()
        if mask_num == 1:
            index = (0,1,2)
        elif mask_num ==2:
            index = (3,4,5)
        elif mask_num ==3:
            index = (6,7,8)
        else:
            index = None

        self.anchors= torch.tensor([ANCHORS_LIST[i] for i in index])
        self.stride = stride
        self.class_num = 80 #coco dataset has 80 class number

    def forward(self, x):
        batch_size = x.size(0)
        grid_size = x.size(2)
        num_anchors = 3
        num_attrib = 5+self.class_num #5 + number of classes which is 80 so 85 total
            
        if self.training:
            output = x.view(batch_size, num_anchors, num_attrib, grid_size, grid_size).permute(0,1,3,4,2).contiguous().view(batch_size, -1, num_attrib)
            return output
        else:
            prediction = x.view(batch_size, num_anchors, num_attrib, grid_size, grid_size).permute(0,1,3,4,2).contiguous()
            self.anchors = self.anchors.to(x.device).float()
            #Calculate new Offset 
            grid_tensor = torch.arange(grid_size, dtype = torch.float, device = x.device).repeat(grid_size,1)
            grid_x = grid_tensor.view([1,1,grid_size, grid_size])
            grid_y = grid_tensor.t().view([1,1,grid_size, grid_size])
            anchor_w = self.anchors[:,0:1].view((1,-1,1,1))
            anchor_h = self.anchors[:,1:2].view((1,-1,1,1))

            #Get Output
            torch.save(grid_x, "grid_x_c")
            torch.save(grid_y, "grid_y_c")
            x_pred = (torch.sigmoid(prediction[...,0]) + grid_x) * self.stride                         # Center x
            y_pred = (torch.sigmoid(prediction[...,1]) + grid_y) * self.stride                         # Center y
            w_pred = torch.exp(prediction[...,2])*anchor_w                                           # Width
            h_pred = torch.exp(prediction[...,3])*anchor_h                                           # Height
            pred_conf = torch.sigmoid(prediction[...,4]).view(batch_size, -1, 1)                # Conf
            pred_cls = torch.sigmoid(prediction[...,5:]).view(batch_size, -1, self.class_num)   # Class prediction

            pred_bbox = torch.stack((x_pred, y_pred, w_pred, h_pred),dim=4).view(batch_size,-1,4)      # Bounding Box (cx, cy, w, h)
            output = torch.cat((pred_bbox, pred_conf, pred_cls), -1)
            return output



class DetectionBlock(nn.Module):
    """
    Detection Block consists of 6 Convolutional layers, 1 Conv 2D layer, and 1 Yolo layer
    first 6 layers have filter size 1, 3 ,1 ,3, 1 ,3 
    The 2D conv layer filter size = 1 x 1 x 225
    
    """
    def __init__(self, channel_in, channel_out, mask_num, stride):

        super().__init__()
        channel_half = channel_out//2

        self.conv1 = ConvoLayer(channel_in, channel_half, 1)
        self.conv2 = ConvoLayer(channel_half, channel_out, 3)
        self.conv3 = ConvoLayer(channel_out, channel_half, 1)
        self.conv4 = ConvoLayer(channel_half, channel_out, 3)
        self.conv5 = ConvoLayer(channel_out, channel_half, 1)
        self.conv6 = ConvoLayer(channel_half, channel_out, 3)

        self.conv7 = nn.Conv2d(channel_out, 3*(80+5), 1, bias=True)  # anchor number = 3 and numb of attribe = (class_num +5) 
        self.yolo = YoloLayer(mask_num, stride)

    def forward(self, x):
        y_ = self.conv1(x)
        y_ = self.conv2(y_)
        y_ = self.conv3(y_)
        y_ = self.conv4(y_)
        self.branch = self.conv5(y_)
        y_ = self.conv6(self.branch)
        y_ = self.conv7(y_)
        y = self.yolo(y_)
        return y

class Darknet53Backbone(nn.Module):
    """
    Darnet53 backbone consists of one Convolutional layer and five Darknet Blocks

    """
    def __init__(self):
        
        super().__init__()
        self.conv1 = ConvoLayer(3,32,3)
        self.db1 = DarkNetBlock(32, 64, 1)
        self.db2 = DarkNetBlock(64, 128, 2)
        self.db3 = DarkNetBlock(128,256, 8)
        self.db4 = DarkNetBlock(256,512, 8)
        self.db5 = DarkNetBlock(512,1024,4)

    def forward(self, x):
        y = self.conv1(x)
        y = self.db1(y)
        y = self.db2(y)
        y3 = self.db3(y)
        y2 = self.db4(y3)
        y1 = self.db5(y2)

        return y1, y2, y3

class YoloNetTail(nn.Module):
    """
    It will take Darkent53Bacbone and do some upsampling and concatentation.

    """
    def __init__(self):

        super().__init__()
        self.d1 = DetectionBlock(1024, 1024, 3, 32)
        self.conv1 = ConvoLayer(512, 256, 1)
        self.d2 = DetectionBlock(768, 512, 2, 16)
        self.conv2 = ConvoLayer(256, 128, 1)
        self.d3 = DetectionBlock(384,256,1, 8)


    def forward(self, x1, x2, x3):
        y1 = self.d1(x1)
        b1 = self.d1.branch
        y = self.conv1(b1)
        y = F.interpolate(y, scale_factor = 2)
        y = torch.cat((y, x2), 1)
        y2 = self.d2(y)
        b2 = self.d2.branch
        y = self.conv2(b2)
        y = F.interpolate(y, scale_factor = 2)
        y = torch.cat((y, x3),1)
        y3 = self.d3(y)

        return y1, y2, y3

class YoloV3(nn.Module):
    """
    Final Yolo v3 model
    Combining Darknet53Backbone and YoloNetTail

    """
    def __init__(self):
        super().__init__()
        self.darknet = Darknet53Backbone()
        self.yolotail = YoloNetTail()
        #self.nms = NMS
        #self.post_process= post

    def forward(self, x):
        y1_, y2_, y3_ = self.darknet(x)
        y1, y2, y3 = self.yolotail(y1_, y2_, y3_)
        y = torch.cat((y1,y2,y3),1)
        return y

    def yolo_last_layers(self):
        _layers = [self.yolotail.d1.conv7, self.yolotail.d2.conv7, self.yolotail.d3.conv7]
        return _layers
