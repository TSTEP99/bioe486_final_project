from torch.utils.data import Dataset
from torchvision.io import read_image
import clip
import torch

class ROCODataset(Dataset):
    def __init__(self, csv):
        self.img_paths = csv["image_path"].values
        self.captions = csv["caption"].values
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        caption = clip.tokenize(self.captions[idx])
        image = read_image(img_path)

        return image, caption


