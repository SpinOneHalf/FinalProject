from networks import Autoencoder
from datasets import UnlabeldImageDataset
from plottingtools import PlotManager
import torch.optim as optim
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=.1):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()).to(tensor.device) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
class DropOut(object):
    def __init__(self,p=.2):
        self.p=p
    def __call__(self,tensor):
        index=torch.rand(tensor.shape[1:])<self.p
        tensor[:,index]=0
        return tensor
device="cuda:0"
test_=UnlabeldImageDataset("unlabeled/")
net =Autoencoder()
training_loader=torch.utils.data.DataLoader(test_,batch_size=64,shuffle=True,num_workers=0)
criterion=nn.MSELoss()
optimizer=optim.Adam(net.parameters(),lr=.0001)
net.to(device)
test_manager=PlotManager()
beta=.01
counter=0
tr_trainsformer=transforms.RandomChoice([transforms.ElasticTransform(alpha=150.),AddGaussianNoise(), DropOut()])
for epoch in range(100):
    running_loss = 0.0
    for i, data in enumerate(training_loader, 0):
        # zero the parameter gradients
        optimizer.zero_grad()
        data=data.to(device)/255
        input_data=tr_trainsformer(torch.clone(data))
        # forward + backward + optimize
        outputs = net(input_data.float())
        loss = criterion(outputs, data.float())
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % 20 == 0:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}')
            if counter!=0:

                outputs=net(data.float())
                test_manager.update_loss(running_loss/20,counter)
                test_manager.update_images(outputs[0].detach().cpu(),data[0].detach().cpu())
                net.train()
            counter+=1
            running_loss = 0.0
del test_manager
model_scripted = torch.jit.script(net)  # Export to TorchScript
model_scripted.save('og_scripted.pt')