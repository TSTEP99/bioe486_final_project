from datasets import ROCODataset
from tqdm import tqdm
import clip
import pandas as pd

train_csv = pd.read_csv("../data/train_data.csv")
train_csv = train_csv[train_csv["valid"].values]
train_ds = ROCODataset(train_csv)
val_csv = pd.read_csv("../data/val_data.csv")
val_csv = val_csv[val_csv["valid"].values]
val_ds = ROCODataset(val_csv)
test_csv = pd.read_csv("../data/test_data.csv")
test_csv = test_csv[test_csv["valid"].values]
test_ds = ROCODataset(test_csv)

tokenizer = clip.simple_tokenizer.SimpleTokenizer()

max_length = 0

for image, caption in tqdm(train_ds):
    token_length = len(tokenizer.encode(caption))

    if token_length > max_length:
        max_length = token_length

for image, caption in tqdm(val_ds):
    token_length = len(tokenizer.encode(caption))

    if token_length > max_length:
        max_length = token_length

for image, caption in tqdm(test_ds):
    token_length = len(tokenizer.encode(caption))

    if token_length > max_length:
        max_length = token_length
print(max_length) #654