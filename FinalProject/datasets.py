import glob
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset

class UnlabeldImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        super().__init__()
        self.imgs_folder = glob.glob(img_dir+"*")
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.imgs_folder)

    def __getitem__(self, idx):
        img_path=self.imgs_folder[idx]
        try :
            image = read_image(img_path)
        except:
            print(f"Failed on {idx}")
            image=torch.zeros((3,224,224)).float()
        if self.transform:
            image = self.transform(image)
        return image