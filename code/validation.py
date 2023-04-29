from datasets import HemorrhageDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import clip
import numpy as np
import sklearn
import sklearn.linear_model
import sklearn.model_selection
import torch


def linear_probing(dataloader,model, device, n_splits = 5):
    size = len(dataloader.dataset)
    total_features = []
    total_labels = []
    correct = 0

    with torch.no_grad():
        for (images, labels) in dataloader:
            total_labels.append(labels.detach().cpu().numpy())
            total_features.append(model.encode_image(images.to(device)).detach().cpu().numpy())
    total_features = np.concatenate(total_features, axis = 0)
    total_labels = np.concatenate(total_labels, axis = 0)
    kf = sklearn.model_selection.KFold(n_splits = n_splits)

    for (train_index, test_index) in kf.split(total_features):
        train_features = total_features[train_index]
        test_features = total_features[test_index]  
        train_labels = total_labels[train_index]
        test_labels = total_labels[test_index]

        clf = sklearn.linear_model.LogisticRegression().fit(train_features, train_labels)
        preds = clf.predict(test_features)

        correct += np.sum(preds == test_labels)
    return correct/size


def val_test(dataloader, model, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    correct = 0

    with torch.no_grad():
        for (images, labels) in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)  
            captions = clip.tokenize(["computed tomography of a normal patient", "computed tomography of a hemmorhaging patient"]).to(device)
            logits_per_image, _ = model(images, captions)
            preds = torch.argmax(logits_per_image, dim = 1)
            print(preds)
            correct += torch.sum(preds == labels)
    return correct/size

if __name__ == "__main__":
    #Set parameters
    BATCH_SIZE = 64
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    #Load model and preprocessing
    #model = torch.load("../checkpoints/best_model1.pth", map_location = device)
    model, preprocess = clip.load("ViT-B/32", device=device)



    #Validation Dataset
    val_ds = HemorrhageDataset(preprocess)

    #Creating Dataloaders
    val_dataloader = DataLoader(val_ds, batch_size = BATCH_SIZE, shuffle=True)

    #Performing out validation test
    accuracy = val_test(val_dataloader, model, device)
    #accuracy = linear_probing(val_dataloader, model, device)

    print("Accuracy: ", accuracy)
