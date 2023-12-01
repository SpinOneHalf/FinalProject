import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import re

class VideoDataset(Dataset):
    def __init__(self, root_dirs, transform=None, sequence_length=11):
        self.transform = transform
        self.sequence_length = sequence_length
        self.frame_paths = []

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

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        input_frames = self.frame_paths[idx][:self.sequence_length]
        target_frames = self.frame_paths[idx][self.sequence_length:]
        input_frames = torch.stack([self.transform(read_image(frame)) for frame in input_frames])
        target_frames = torch.stack([self.transform(read_image(frame)) for frame in target_frames])
        return input_frames, target_frames
