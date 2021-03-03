import math

import numpy as np
import torch
from torch import nn, Tensor
import torch.optim as optim
from torchvision import models 
import torch.nn.functional as F
import efficientnet_pytorch


class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
    
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.c5 = nn.Linear(16*4*4, 120)
        self.f6 = nn.Linear(120, 84)
        self.output = nn.Linear(84, num_classes)
 
    def forward(self, x):
        x = self.c1(x)
        x = F.max_pool2d(x, kernel_size=2) # S2포함
        x = self.c3(x)
        x = F.max_pool2d(x, kernel_size=2) # S4포함
        x = x.view(-1, 16*4*4)
        x = self.c5(x)
        x = self.f6(x)
        x = self.output(x)
        return x

# model = LeNet().to(device)
# print(summary(model, input_size=(1,28,28), batch_size=batch_size, device=device))

class QDCNN(nn.Module):
    def __init__(self, num_classes = 345):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv2d(1, 5, kernel_size=3, stride=1, padding = 1), nn.ReLU(inplace=True))
        self.c2 = nn.Sequential(nn.Conv2d(5, 5, kernel_size=3, stride=1, padding = 1), nn.ReLU(inplace=True))
        self.c3 =nn.Sequential(nn.Conv2d(5, 5, kernel_size=3, stride=1, padding =1), nn.ReLU(inplace=True))
        self.l1 = nn.Linear(5*14*14, 700)
        self.l2 = nn.Linear(700, 500)
        self.l3 = nn.Linear(500, 400)
        self.l4 = nn.Linear(400, 345)
    
    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 5*14*14)
        x = self.l1(x)
        #x = nn.ReLU(x)
        x = self.l2(x)
        #x = nn.ReLU(x)
        x = self.l3(x)
        #x = nn.ReLU(x)
        x = self.l4(x)
        return x

class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class SwishModule(nn.Module):
    def forward(self, x):
        return Swish.apply(x)

    
class DenseCrossEntropy(nn.Module):
    def forward(self, x, target):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()


class ArcMarginProductSubcenter(nn.Module):
    def __init__(self, in_features, out_features, k=3):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features*k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        
    def forward(self, features):
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine   


class ArcFaceLossAdaptiveMargin(nn.Module):
    def __init__(self, margins, s=30.0):
        super().__init__()
        self.crit = DenseCrossEntropy()
        self.s = s
        self.margins = margins
            
    def forward(self, logits, labels, out_dim):
        ms = []
        ms = self.margins[labels.cpu().numpy()]
        cos_m = torch.from_numpy(np.cos(ms)).float().cuda()
        sin_m = torch.from_numpy(np.sin(ms)).float().cuda()
        th = torch.from_numpy(np.cos(math.pi - ms)).float().cuda()
        mm = torch.from_numpy(np.sin(math.pi - ms) * ms).float().cuda()
        labels = F.one_hot(labels, out_dim).float()
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * cos_m.view(-1,1) - sine * sin_m.view(-1,1)
        phi = torch.where(cosine > th.view(-1,1), phi, cosine - mm.view(-1,1))
        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = self.crit(output, labels)
        return loss


class EfficientNetLandmark(nn.Module):
    def __init__(self, depth, num_classes=1049):
        super().__init__()
        # self.enet = geffnet.create_model(enet_type.replace('-', '_'), pretrained=True)
        self.base = efficientnet_pytorch.EfficientNet.from_pretrained(f'efficientnet-b{depth}')
        self.linear = nn.Linear(self.base._fc.in_features, 512) # 1280, 512
        self.swish = SwishModule()
        self.classifier = ArcMarginProductSubcenter(512, num_classes)
        self.base._fc = nn.Identity()

    def forward(self, x):
        x = self.base(x)
        x = self.linear(x)
        x = self.swish(x)
        x = self.classifier(x)
        return x

# num_features = model.fc.in_features
# model.fc = nn.Linear(num_features, num_classes)
# model = model.to(device)

class ResNext101Landmark(nn.Module):
    def __init__(self, num_classes=1049):
        super().__init__()
        self.base = models.resnext101_32x8d(pretrained=True)
        self.linear = nn.Linear(self.base.fc.in_features, 512) # 2048, 512
        self.swish = SwishModule()
        self.classifier = ArcMarginProductSubcenter(512, num_classes)
        self.base.fc = nn.Identity()

    def forward(self, x):
        x = self.base(x)
        x = self.linear(x)
        x = self.swish(x)
        x = self.classifier(x)
        return x


# if __name__ == '__main__':
#     from torchsummary import summary
#     model = EfficientNetLandmark(0)
#     print(summary(model, input_data=(3, 300, 300)))
