from networks import VariationalAutoencoder,CustomAutoEncoder
from datasets import UnlabeldImageDataset
from plottingtools import PlotManager
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision.transforms as transforms


#
if __name__=="__main__":
    device="cuda:0"
    test_=UnlabeldImageDataset("unlabeled_data/")
    net =CustomAutoEncoder()
    training_loader=torch.utils.data.DataLoader(test_,batch_size=14,shuffle=True,num_workers=0)
    criterion=nn.MSELoss()
    optimizer=optim.Adam(net.parameters(),lr=.0001)
    net.to(device)
    test_manager=PlotManager()
    beta=.01
    counter=0
    tr_trainsformer=transforms.Compose([transforms.ElasticTransform(alpha=250.)])
    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(training_loader, 0):
            # zero the parameter gradients
            optimizer.zero_grad()
            data=data.to(device)/255
            input_data=tr_trainsformer(data)
            # forward + backward + optimize
            outputs= net(input_data.float())
            loss = criterion(outputs, data.float())
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 20 == 0:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}')
                if counter!=0:

                    outputs=net(input_data.float())
                    test_manager.update_loss(running_loss/20,counter)
                    test_manager.update_images(outputs[0].detach().cpu(),input_data[0].detach().cpu())
                    net.train()
                counter+=1
                running_loss = 0.0
    del test_manager
    model_scripted = torch.jit.script(net)  # Export to TorchScript
    model_scripted.save('smallauto_scripted.pt')  # Save

    test_ = UnlabeldImageDataset("unlabeled_data/")
    net = VariationalAutoencoder()
    training_loader = torch.utils.data.DataLoader(test_, batch_size=14, shuffle=True, num_workers=0)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=.001)
    net.cuda()
    test_manager = PlotManager(
    )
    beta = .01
    counter = 0
    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(training_loader, 0):
            # zero the parameter gradients
            optimizer.zero_grad()
            data = data.to("cuda") / 255
            input_data = tr_trainsformer(data)
            # forward + backward + optimize
            outputs, encoded = net(input_data.float())
            loss = criterion(outputs, data.float())
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 20 == 0:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}')
                if counter != 0:
                    outputs, _ = net(input_data.float())
                    test_manager.update_loss(running_loss / 20, counter)
                    test_manager.update_images(outputs[0].detach().cpu(), data[0].detach().cpu())
                    net.train()
                counter += 1
                running_loss = 0.0
    del test_manager
    model_scripted = torch.jit.script(net)  # Export to TorchScript
    model_scripted.save('high_auto_scripted.pt')  # Save