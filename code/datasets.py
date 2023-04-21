from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image
import clip
import os
import torch

class ROCODataset(Dataset):
    def __init__(self, csv, preprocess, data_dir = "/mnt/ssd_4tb_0/teja/roco_data"):
        self.data_dir = data_dir
        self.preprocess = preprocess
        self.img_paths = csv["image_path"].values
        self.captions = csv["caption"].values
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.img_paths[idx])
        caption = clip.tokenize(self.captions[idx], context_length = 77).squeeze(dim=0)
        image = self.preprocess(Image.open(img_path)) #May need to add preprocessing code

        return image, caption


