import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import torchvision.models as models
from modules import *
from data_loader import DATASET_CONFIGS

class MyModel(nn.Module):

    def __int__(self):
        super(MyModel, self).__init__()
        self.model = resnet20(config.planes, DATASET_CONFIGS[config.dataset], config.num_caps, config.caps_size, config.depth, mode=self.mode).to(device)
        self.cnn_model = models.resnet18(pretrained=True).to(device)
        self.alpha = nn.Parameter(torch.tensor(0.5),requires_grad=True)
        # self.linear = nn.Linear(1000,num_classes)

    def forward(self,inputs):
        
        out1 = self.model(inputs)
        out2 = self.cnn_model(inputs)
        return self.alpha*out1 +(1-self.alpha)*out2