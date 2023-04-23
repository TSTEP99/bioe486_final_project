from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image
import clip
import pandas as pd
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
        image = self.preprocess(Image.open(img_path))

        return image, caption
    
class HemorrhageDataset(Dataset):
    def __init__(self, preprocess, data_dir = "/mnt/ssd_4tb_0/teja/hemorrhage/"):
        self.preprocess = preprocess
        self.label_csv = pd.read_csv(os.path.join(data_dir,"labels.csv"))
        self.img_dir = os.path.join(data_dir, "head_ct/head_ct/")
    def __len__(self):
        return len(self.label_csv)
    def __getitem__(self, idx):
        img_index = idx
        img_path = os.path.join(self.img_dir,"%03d.png"%(idx))
        image = self.preprocess(Image.open(img_path))
        img_label = self.label_csv[" hemorrhage"][idx]
        return image, img_label



