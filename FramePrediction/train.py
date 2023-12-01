import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import Compose, ConvertImageDtype, Normalize
from simvp_model import SimVP_Model
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
#from tqdm import tqdm
from VideoDataset import VideoDataset


def main():
    # Calculated from the training + unlabeled data set
    mean = torch.tensor([0.5072, 0.5056, 0.5021])
    std = torch.tensor([0.0547, 0.0544, 0.0587])

    # Define transformations with calculated mean and std
    transforms = Compose([
        ConvertImageDtype(torch.float),
        Normalize(mean=mean, std=std)
    ])

    train_dataset = VideoDataset(root_dirs=['Dataset_Student/train', 'Dataset_Student/unlabeled'], transform=transforms, sequence_length=11)
    eval_dataset = VideoDataset(root_dirs=['Dataset_Student/val'], transform=transforms, sequence_length=11)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    eval_loader = DataLoader(eval_dataset, batch_size=16, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SimVP_Model(in_shape=(11, 3, 240, 160), hid_S=16, hid_T=256, N_S=4, N_T=4, model_type='gSTA',
                 mlp_ratio=8., drop=0.0, drop_path=0.0, spatio_kernel_enc=3,
                 spatio_kernel_dec=3)
    model.to(device)

    num_epochs = 200
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_loader), epochs=num_epochs)
    criterion = torch.nn.MSELoss(reduction='mean')

    # Training loop
    min_val_loss = None
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        #train_bar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{num_epochs}')

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
        
        average_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {average_train_loss:.4f}")
            
        # Validation loop
        model.eval()
        total_val_loss = 0
        #val_bar = tqdm(eval_loader, desc='Validation')

        with torch.no_grad():
            for inputs, targets in eval_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                predictions = model(inputs)

                val_loss = criterion(predictions, targets)
                total_val_loss += val_loss.item()

        average_val_loss = total_val_loss / len(eval_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {average_val_loss:.4f}")

        if min_val_loss is None:
            min_val_loss = average_val_loss
        elif average_val_loss < min_val_loss:
            min_val_loss = average_val_loss
            torch.save(model.state_dict(), 'pred_model_checkpoint.pth')


if __name__ == '__main__':
    main()