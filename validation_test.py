
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
import os
import re
from torchvision.transforms import Compose, ConvertImageDtype, Normalize
import numpy as np
import torchmetrics

from FramePrediction.simvp_model import SimVP_Model
mean_standard = torch.tensor([0.5072, 0.5056, 0.5021])
std_standard = torch.tensor([0.0547, 0.0544, 0.0587])
def unnormalize(tensor, mean=mean_standard, std=std_standard):
    """Revert the normalization process."""
    for k in range(tensor.shape[1]):
        for i in range(tensor.shape[0]):
            for j in range(3):
                tensor[i,k,j]=tensor[i,k,j].mul_(std[j]).add_(mean[j])  # Multiply by std and add the mean
    return tensor

class MaskedVideoDataset(Dataset):
    def __init__(self, root_dirs, transform=None, sequence_length=11):
        self.transform = transform
        self.sequence_length = sequence_length
        self.frame_paths = []
        self.mask_paths = []
        self.max_class=49

        for root_dir in root_dirs:
            all_dirs = sorted(os.listdir(root_dir))
            video_dirs = [
                os.path.join(root_dir, d) for d in all_dirs
                if d.startswith('video_') and os.path.isdir(os.path.join(root_dir, d))
            ]

            for video_dir in video_dirs:
                all_files = os.listdir(video_dir)
                frames = [
                    os.path.join(video_dir, f) for f in all_files
                    if re.match(r'image_\d+\.png', f)
                ]
                frames = sorted(frames, key=lambda x: int(re.search(r'image_(\d+)\.png', os.path.basename(x)).group(1)))

                assert len(frames) == 22, f"Expected 22 frames, but found {len(frames)} in {video_dir}"
                self.frame_paths.append(frames[:22])
                self.mask_paths.append(os.path.join(video_dir,"mask.npy"))

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        #Get the images for predictor
        input_frames = self.frame_paths[idx][:self.sequence_length]
        input_frames = torch.stack([self.transform(read_image(frame)) for frame in input_frames])
        #Get lask mask
        target_masks_raw=np.load(self.mask_paths[idx])[-1]
        mask = torch.zeros(self.max_class, 160, 240)
        raw_mask_items = set(np.unique(target_masks_raw))
        for lil_mask in raw_mask_items:
            _temp_array = np.zeros((160, 240))
            try:
                _temp_array[target_masks_raw == lil_mask] = 1
                mask[lil_mask, :, :] = torch.tensor(_temp_array)
            except IndexError:
                print("HERE")
        return input_frames, mask,target_masks_raw

def convert_out_put_new(net_out):
    net_out=net_out.cpu()
    r=torch.zeros((net_out.shape[0],160,240))
    for j in range(160):
        for k in range(240):
            out_class=torch.argmax(net_out[:,:,j,k])
            out_class= 0 if out_class<.5 else out_class
            r[0,j,k]=out_class
    return r
def convert_out_put(net_out):
    net_out=net_out.cpu()
    r=torch.zeros((net_out.shape[0],160,240))
    for i in range(49):
        r[net_out[:,i,:,:]>.5]=i
    return r
# Define transformations with calculated mean and std
transforms = Compose([
    ConvertImageDtype(torch.float),
    Normalize(mean=mean_standard, std=std_standard)
])

device="cuda"
val_db=MaskedVideoDataset(["Dataset_Student/val"],transforms)
val_db=DataLoader(val_db,batch_size=1,pin_memory=True)

pred_model=torch.load("predict_model_one.pt")
mask_network=torch.load("masked_model_one.pt")
optim=torch.optim.RMSprop(list(pred_model.parameters())+list(mask_network.parameters()),lr=.0000000003)
criteria=torch.nn.CrossEntropyLoss()
jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49)
running_iou=0
running_iou=0
with torch.no_grad():
    for batch in val_db:
        input_images, mask, _ = batch
        predicted_out = pred_model(input_images.to(device))
        predicted_out = unnormalize(predicted_out)
        predicted_mask = mask_network(predicted_out[:, -1, :, :, :])
        running_iou+=jaccard(convert_out_put(mask), convert_out_put(predicted_mask))
print(f"DONE:{running_iou / len(val_db)}")
print("DONE")
