import pathlib
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
import numpy as np

class TrainingSet(Dataset):
    def __init__(self,path_to_files):
        super().__init__()
        file_paths = list(pathlib.Path(path_to_files).rglob("*"))
        self.max_class=49
        self.len_files_paths=len(file_paths)
        self.file_paths=file_paths
    def __len__(self):
        return self.len_files_paths
    def __getitem__(self, item):
        #Get the coressponding image files and masks
        folder_index=item//22
        image_index=item%22
        image_path=next(self.file_paths[folder_index].glob(f"image_{image_index}.png"))
        image_tensor=read_image(str(image_path))
        raw_mask=next(self.file_paths[folder_index].glob("*.npy"))
        raw_mask=np.load(raw_mask)
        mask=torch.zeros(self.max_class, 160, 240)
        raw_mask_items = set(np.unique(raw_mask))
        for lil_mask in raw_mask_items:
            _temp_array = np.zeros((160, 240))
            _temp_array[raw_mask[image_index] == lil_mask] = 1
            try:
                mask[lil_mask,:,:]=torch.tensor(_temp_array)
            except:
                print("O NO")
        return image_tensor,mask,raw_mask[image_index]