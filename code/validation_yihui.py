# -*- coding: utf-8 -*-
"""project_run.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jjr2nB5Mnx9gUqpnyZcVKYZWw515MKe1
"""

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import sys
sys.path.append('/content/drive/MyDrive/project/bioe486_final_project-main/code')

# !unzip -u "/content/drive/MyDrive/project/bioe486_final_project.zip" -d "/content/drive/MyDrive/project"

# !unzip -u "/content/drive/MyDrive/project/val_data.zip" -d "/content/drive/MyDrive/project/val_data"

# !unzip -u "/content/drive/MyDrive/project/bioe486_final_project-main/code/checkpoints-20230430T193022Z-001" -d "/content/drive/MyDrive/project/bioe486_final_project-main/code"

# !unzip -u "/content/drive/MyDrive/project/all_data.zip" -d "/content/drive/MyDrive/project/all_data"

!pip install ftfy regex tqdm
!pip install git+https://github.com/openai/CLIP.git

from datasets import ROCODataset
from losses import OG_CLIP_loss
from torch.utils.data import DataLoader
from tqdm import tqdm
import clip
import math
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image
import clip
import os
import torch

class CTDataset(Dataset):
    def __init__(self, csv, preprocess, data_dir = "/content/drive/MyDrive/project/val_data/"):
        self.data_dir = data_dir
        self.preprocess = preprocess
        self.img_paths = csv["image_path"].values
        self.labels = csv["image_label"].values
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.img_paths[idx])
        image = self.preprocess(Image.open(img_path))
        img_label = self.labels[idx]
        return image, img_label

torch.cuda.is_available()

#Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
#load model try
model_fine = torch.load("/content/drive/MyDrive/project/bioe486_final_project-main/code/best_model.pth", map_location=device)
model, preprocess = clip.load("ViT-B/32", device=device)
tokenizer = clip.simple_tokenizer.SimpleTokenizer()

print(model.ResidualAttentionBlock)

print(model)

from tqdm import tqdm
import clip
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import sklearn.model_selection
import torch
import sklearn.metrics

def multi_class(dataloader, model, device, n_splits = 5):
    size = len(dataloader.dataset)
    total_features = []
    total_labels = []
    correct = 0

    with torch.no_grad():
        for (images, labels) in tqdm(dataloader):
            total_labels.append(labels)
            total_features.append(model.encode_image(images.to(device)).detach().cpu().numpy())
    total_features = np.concatenate(total_features, axis = 0)
    total_labels = np.concatenate(total_labels, axis = 0)
    kf = sklearn.model_selection.KFold(n_splits = n_splits)

    auc=0
    accuracy=0
    for (train_index, test_index) in kf.split(total_features):
        train_features = total_features[train_index]
        test_features = total_features[test_index]  
        train_labels = total_labels[train_index]
        test_labels = total_labels[test_index]

        classifier = KNeighborsClassifier(n_neighbors=5)
        clf = classifier.fit(train_features, train_labels)
        preds = clf.predict(test_features)
        y_pred = np.zeros((preds.shape[0],3))
        y_true = np.zeros((preds.shape[0],3))
        y_pred[preds=='COVID19',0]=1
        y_pred[preds=='NORMAL',1]=1
        y_pred[preds=='PNEUMONIA',2]=1
        y_true[test_labels=='COVID19',0]=1
        y_true[test_labels=='NORMAL',1]=1
        y_true[test_labels=='PNEUMONIA',2]=1

        auc += sklearn.metrics.roc_auc_score(y_true, y_pred, average='weighted', multi_class='ovo')
        accuracy += sklearn.metrics.accuracy_score(test_labels, preds)

    return auc/n_splits, accuracy/n_splits

def val_test(dataloader, model, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    correct = 0
    preds = []
    total_labels = []
    with torch.no_grad():
      for (images, labels) in tqdm(dataloader):
        images = images.to(device)
        label_class = np.array(['normal', 'pneumonia', 'COVID'])
        captions = torch.cat([clip.tokenize(f"Chest X-ray of a {c} patient") for c in label_class]).to(device)
        logits_per_image, _ = model(images, captions)
        preds.append((torch.argmax(logits_per_image, dim = 1)).cpu().numpy())
        total_labels.append(labels)
    total_labels = np.concatenate(total_labels, axis = 0)
    preds = np.concatenate(preds, axis = 0)
    test_labels = np.zeros(preds.shape)
    test_labels[total_labels=='NORMAL'] = 0
    test_labels[total_labels=='PNEUMONIA'] = 1
    test_labels[total_labels=='COVID19'] = 2
    
    y_pred = np.zeros((preds.shape[0],3))
    y_true = np.zeros((preds.shape[0],3))
    y_pred[preds==2,2]=1
    y_pred[preds==0,0]=1
    y_pred[preds==1,1]=1
    y_true[test_labels==2,2]=1
    y_true[test_labels==0,0]=1
    y_true[test_labels==1,1]=1
    auc = sklearn.metrics.roc_auc_score(y_true, y_pred, average='weighted', multi_class='ovo')
    accuracy = sklearn.metrics.accuracy_score(test_labels, preds)

    return auc, accuracy

path_t = "/content/drive/MyDrive/project/bioe486_final_project-main/data/"
ct_csv = pd.read_csv(path_t + "CT_data.csv")
BATCH_SIZE = 64
ct_ds = CTDataset(ct_csv, preprocess)
ct_dataloader = DataLoader(ct_ds, batch_size = BATCH_SIZE, shuffle=True)

# original
auc, accuracy = val_test(ct_dataloader, model, device)
print("Model name: original CLIP, Accuracy:%.2f, AUC: %.2f" % (accuracy, auc))
auc, accuracy = multi_class(ct_dataloader, model, device)
print("Model name: Classifier with original CLIP, Accuracy:%.2f, AUC: %.2f" % (accuracy, auc))

# fine-tuned 
auc, accuracy = val_test(ct_dataloader, model_fine, device)
print("Model name: fine-tuned CLIP, Accuracy:%.2f, AUC: %.2f" % (accuracy, auc))
auc, accuracy = multi_class(ct_dataloader, model_fine, device)
print("Model name: Classifier with fine-tuned CLIP, Accuracy:%.2f, AUC: %.2f" % (accuracy, auc))

# original CLIP
total_features = []
total_labels = []
correct = 0

with torch.no_grad():
    for (images, labels) in tqdm(ct_dataloader):
        total_labels.append(labels)
        total_features.append(model.encode_image(images.to(device)).detach().cpu().numpy())
total_features = np.concatenate(total_features, axis = 0)
total_labels = np.concatenate(total_labels, axis = 0)
color_in=np.zeros(total_labels.shape, dtype = object)
color_in[total_labels=='NORMAL']='r'
color_in[total_labels=='PNEUMONIA']='g'
color_in[total_labels=='COVID19']='b'
# embedding plot here
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
pca = PCA(n_components=2)
in_pca = pca.fit_transform(total_features)
plt.figure(figsize=(8, 8))
plt.scatter(x = in_pca[:,0], y=in_pca[:,1], color=color_in)
plt.show()

# fine-tuned CLIP
total_features = []
total_labels = []
correct = 0

with torch.no_grad():
    for (images, labels) in tqdm(ct_dataloader):
        total_labels.append(labels)
        total_features.append(model_fine.encode_image(images.to(device)).detach().cpu().numpy())
total_features = np.concatenate(total_features, axis = 0)
total_labels = np.concatenate(total_labels, axis = 0)
color_in=np.zeros(total_labels.shape, dtype = object)
color_in[total_labels=='NORMAL']='r'
color_in[total_labels=='PNEUMONIA']='g'
color_in[total_labels=='COVID19']='b'
# embedding plot here
pca = PCA(n_components=2)
in_pca = pca.fit_transform(total_features)
plt.figure(figsize=(8, 8))
plt.scatter(x = in_pca[:,0], y=in_pca[:,1], color=color_in)
plt.show()