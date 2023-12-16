import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import copy
import math
from models import load_models_using, get_model_results
import pandas as pd
import random
import numpy as np

# Seed Data
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Apply data augmentation (random resizing and horizontal flipping) and normalization for Training dataset
# Apply normalization only for Validation dataset
data_augmented = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load data without any data augmentation
data_original = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Set up data path, loaders, class names, device type, and loss function
data_dir = './data/ecg_images'
batch_size = 136
image_datasets = {x: datasets.ImageFolder(f"{data_dir}/{x}", data_original[x]) for x in ['train', 'val']}

train_loader = DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True)
val_loader = DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False)
print(len(train_loader), len(val_loader))

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

class_names = [d for d in os.listdir(data_dir + "/train") if d[0] != 0]
num_classes = len(class_names)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss()


if __name__ == '__main__':
    total_times = []
    results = []
    columns = ["Train_Loss", "Val_Loss", "Train_Acc", "Val_Acc"]
    outpath = './outputs/original_data'
    num_epochs = 1000
    max_patience = num_epochs

    model_names, model_list, model_optimizers = load_models_using(device, num_classes)

    for i in range(len(model_list)):
        model = model_list[i]
        name = model_names[i]
        optimizer = model_optimizers[i]

        train_losses = []
        val_losses = []
        train_acc = []
        val_acc = []
        class_accuracies = {class_name: [] for class_name in class_names}

        last_loss = float('inf')
        val_loss = float('inf')
        patience_counter = 0
        best_model_weights = None
        loss_is_nan = False

        print(f"Training {name}...")
        start_time = time.time()
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            class_correct = {classname: 0 for classname in class_names}
            class_total = {classname: 0 for classname in class_names}

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                    data_loader = train_loader
                else:
                    model.eval()   # Set model to evaluate mode
                    data_loader = val_loader

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in data_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)


                        # Backward + optimize only if in training phase, collect class acc in val
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                        else:
                            for label, pred in zip(labels, preds):
                                if label == pred:
                                    class_correct[class_names[label]] += 1
                                class_total[class_names[label]] += 1

                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                # Calculate Loss/Accuracy
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                # Check if loss values are not a number
                if math.isnan(epoch_loss):
                    loss_is_nan = True
                    break

                if phase == 'train':
                    train_losses.append(round(epoch_loss,2))
                    train_acc.append(round(epoch_acc.cpu().item(), 2))
                else:
                    val_loss = epoch_loss
                    val_losses.append(round(epoch_loss,2))
                    val_acc.append(round(epoch_acc.cpu().item(),2))
                
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            for classname in class_names:
                accuracy = round(100 * float(class_correct[classname]) / class_total[classname], 2)
                class_accuracies[classname].append(accuracy)

            # Calculate Average Losses
            avg_loss = val_loss / len(val_loader)
            if avg_loss < last_loss:
                last_loss = avg_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
            elif loss_is_nan:
                print("Early stop triggered. Loss values are NaN!")
                break
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print("Early stop triggered. No validation loss improvement")
                    break 

        class_results = pd.DataFrame(class_accuracies)
        class_results.index.name = 'Epoch'
        class_results.index += 1
        class_results.to_csv(f'{outpath}/{name}_val_classes.csv')

        model_results = [train_losses, val_losses, train_acc, val_acc]
        results.append(get_model_results(model_results, name, columns))

        torch.save(best_model_weights, f"./weights/{name}.pt")
        end_time = time.time()

        print('Training complete')
        time_format = "%H:%M:%S"
        total_time = time.strftime(time_format, time.gmtime(end_time - start_time))
        total_times.append(total_time)
        print(f"Time: {total_time}\n")


        print("Plotting...")
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label ='Training Loss')
        plt.plot(val_losses, label ='Validation Loss')
        plt.title(f'{name} Loss per Epoch ({total_time})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'{outpath}/{name}_Loss.png')
        plt.clf()

        plt.figure(figsize=(10, 6))
        plt.plot(train_acc, label ='Training Accuracy')
        plt.plot(val_acc, label ='Validation Accuracy')
        plt.title(f'{name} Accuracy per Epoch ({total_time})')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.legend()
        plt.savefig(f'{outpath}/{name}_Accuracy.png')
        plt.clf()
        print("Done!\n")
    
    ekg_training_stats = pd.concat([*results], axis=1)
    ekg_training_stats.to_csv(f"{outpath}/results.csv")

