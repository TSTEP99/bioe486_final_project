from datasets import ROCODataset
from losses import OG_CLIP_loss
from torch.utils.data import DataLoader
from tqdm import tqdm
import clip
import math
import pandas as pd
import torch

def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (images, captions) in tqdm(enumerate(dataloader)):
        # Compute prediction and loss
        # image_features = model.encode_image(images.to(device))
        # text_features = model.encode_text(captions.to(device))
        logits_per_image, logits_per_text = model(images.to(device), captions.to(device))
        loss = loss_fn(logits_per_image, logits_per_text)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(images)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for (images, captions) in tqdm(dataloader):
            # Compute prediction and loss
            # image_features = model.encode_image(images.to(device))
            # text_features = model.encode_text(captions.to(device))
            logits_per_image, logits_per_text = model(images.to(device), captions.to(device))
            loss = loss_fn(logits_per_image, logits_per_text)
            test_loss += loss

    test_loss /= num_batches
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")
    return test_loss

if __name__ == "__main__":

    #Set device
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    #Defining Parameters
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 1e-4

    #Load Pretrained Model
    model, preprocess = clip.load("ViT-B/32", device=device)
    # model.context_length = 654
    # transformer_width = model.transformer.width
    # model.positional_embedding = torch.nn.Parameter(torch.empty(model.context_length, transformer_width, requires_grad=True, device = device))
    # torch.nn.init.normal_(model.positional_embedding, std=0.01)

    #Import Training Dataset
    train_csv = pd.read_csv("../data/train_data.csv")
    train_csv = train_csv[train_csv["valid"].values]
    train_ds = ROCODataset(train_csv, preprocess)
    
    #Import Validation Dataset
    val_csv = pd.read_csv("../data/val_data.csv")
    val_csv = val_csv[val_csv["valid"].values]
    val_ds = ROCODataset(val_csv, preprocess)
    
    #Import Test Dataset
    test_csv = pd.read_csv("../data/test_data.csv")
    test_csv = test_csv[test_csv["valid"].values]
    test_ds = ROCODataset(test_csv, preprocess)

    #Creating Dataloaders
    train_dataloader = DataLoader(train_ds, batch_size = BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size = BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size = BATCH_SIZE, shuffle=True)

    #Creating Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 1)
    best_val_loss  = math.inf

    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, OG_CLIP_loss, optimizer, device)
        val_loss = test_loop(val_dataloader, model, OG_CLIP_loss, device)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, "../checkpoints/best_model.pth")  
    print("Done!")