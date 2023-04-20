from datasets import ROCODataset
import pandas as pd

train_csv = pd.read_csv("../data/train_data.csv")
ds = ROCODataset(train_csv)

for image, caption in ds:
    print(caption)
