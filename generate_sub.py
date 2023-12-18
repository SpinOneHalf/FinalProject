from torch.utils.data import DataLoader,Dataset
from torchvision.io import read_image
import os
import re
from torchvision.transforms import Compose, ConvertImageDtype, Normalize
import numpy as np
import torchmetrics
import torch

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

                assert len(frames) == 11, f"Expected 11 frames, but found {len(frames)} in {video_dir}"
                self.frame_paths.append(frames)

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        #Get the images for predictor
        input_frames = self.frame_paths[idx][:self.sequence_length]
        input_frames = torch.stack([self.transform(read_image(frame)) for frame in input_frames])
        return input_frames

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

pred_model=torch.load("predict_model_one.pt").cuda()
mask_network=torch.load("masked_model_one.pt").cuda()
data_Set=MaskedVideoDataset(["hidden_set_for_leaderboard_1"],transforms)
results=[]
results=np.zeros((2000,160,240),dtype=np.int8)
with torch.no_grad():
    for i,input_frames in enumerate(data_Set):
        net_out=pred_model(input_frames.reshape(-1,*input_frames.shape).cuda())
        mask=mask_network(unnormalize(net_out[:,-1,:,:]))
        mask=convert_out_put(mask)
        results[i,:,:]=mask[0]
        print(i)

np.save("final_result.npy",results)

