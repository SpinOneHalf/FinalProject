import numpy as np
import torch.nn
import numpy as np
from torch.utils.data import DataLoader
from FinalProject.datasets import TrainingSet
from FinalProject.networks import UNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import matplotlib.pyplot as plt
from tqdm import tqdm


class SimpleSegmentationNet(nn.Module):
    def __init__(self, num_classes):
        super(SimpleSegmentationNet, self).__init__()
        # Define the layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.classifier = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1)
    def forward(self, x):
        # Forward pass
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = self.upsample(x)
        x = self.upsample(x)
        x = self.classifier(x)
        return x
tr_db=TrainingSet("/home/charles/PycharmProjects/MachineLearningHomework/Dataset_Student/train")
val_db=TrainingSet("/home/charles/PycharmProjects/MachineLearningHomework/Dataset_Student/val")
criteria=torch.nn.CrossEntropyLoss()
device=torch.device(0)
#model=UNet(3,49).to(device)
model=torch.load("class.pt")
data_loader=DataLoader(tr_db,batch_size=16,pin_memory=True,num_workers=12)
val_loader=DataLoader(val_db,batch_size=16*2,pin_memory=True)
optim=torch.optim.Adamax(model.parameters(),lr=.00001)
jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49)
def convert_out_put(net_out):
    net_out=net_out.cpu()
    r=torch.zeros((output.shape[0],160,240))
    for i in range(49):
        r[net_out[:,i,:,:]>.5]=i
    return r
for e in range(100):
    for batch in data_loader:
        output=model(batch[0].to(device)/255)
        optim.zero_grad()
        loss=criteria(output,batch[1].to(device))
        loss.backward()
        optim.step()
    running_iou=0
    with torch.no_grad():
        for batch in val_loader:
            output = model(batch[0].to(device) / 255)
            running_iou+=jaccard(batch[2], convert_out_put(output))
    print(f"DONE:{running_iou/len(data_loader)},{e}")
torch.save(model,"class.pt")
