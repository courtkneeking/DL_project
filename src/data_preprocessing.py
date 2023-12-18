# src/data_preprocessing.py
import torch
from torchvision import datasets, transforms
from torchvision.transforms import Grayscale
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Dataset, random_split

def load_data():
    data_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    # Update paths to your dataset
    train_dataset = datasets.ImageFolder(root='/u/erdos/csga/aalfatemi/DLFP/data/train', transform=data_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    return train_loader  
#, val_loader

# Custom dataset for combining real and generated images
class CombinedDataset(Dataset):
    def __init__(self, real_data, generated_data):
        self.data = torch.cat([real_data[0], generated_data[0]], dim=0)
        self.labels = torch.cat([real_data[1], generated_data[1]], dim=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
# Function to generate data in batches
def generate_data_in_batches(generator, latent_dim, num_samples, batch_size=50, device= 'cpu'):
    generated_images = []
    generated_labels = []

    for _ in range(num_samples // batch_size):
        z = torch.randn(batch_size, latent_dim).to(device)
        imgs = generator(z).detach()  # Detach to avoid tracking gradients
        generated_images.append(imgs)
        labels = torch.zeros(batch_size, dtype=torch.long).to(device)  # Labels for generated data
        generated_labels.append(labels)

    # Concatenate all batches
    return torch.cat(generated_images, dim=0), torch.cat(generated_labels, dim=0)

# Function to split dataset into training and validation sets
def split_dataset(dataset, train_ratio=0.8):
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    return random_split(dataset, [train_size, val_size])