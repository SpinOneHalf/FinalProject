import pathlib
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset

class UnlabeldImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        super().__init__()
        imgs_folder = list(pathlib.Path(img_dir).rglob("*"))
        image_files=[]
        for folder in imgs_folder:
            image_files+=list(folder.glob("*png"))
        self.imgs_folder=image_files
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