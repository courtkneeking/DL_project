# main.py
import torch
from src.data_preprocessing import load_data
# , CombinedDataset, generate_data_in_batches, split_dataset
from src.gan import Generator, Discriminator, train_gan, generate_images
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
from torchvision import datasets, transforms
from src.multi_classification import ECG_CNN, train_classifier, evaluate_classifier
import os


# define the gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------- Hyperparameters ----------
# Define the dimensions of images
image_width, image_height = 128, 128

# Define the number of classes
number_of_classes = 5

# Define the latent space dimension
latent_dim = 300

# number of epochs 
num_epochs = 1000

# save_dir='/u/erdos/csga/aalfatemi/DLFP/dataset'
# Load data
train_loader = load_data()  # , val_loader 

# ---------------------
# Build the generator
generator = Generator(latent_dim, image_width, image_height).to(device)


# Build the discriminator
discriminator = Discriminator(image_width, image_height).to(device)


# Traing
losses_G, losses_D, imgs = train_gan(generator, discriminator, train_loader, num_epochs,latent_dim, device)

# # generate images
# generate_images(generator, latent_dim, device, num_images=10000)

# Plot the losses
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(losses_G,label="G")
plt.plot(losses_D,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('/u/erdos/csga/aalfatemi/DLFP/reports/figures/gan_losses.png')

# Load the model parameters
generator.load_state_dict(torch.load("generator.pth"))
discriminator.load_state_dict(torch.load("discriminator.pth"))

# Generate a batch of images
z = torch.randn(32, latent_dim).to(device)
generated_imgs = generator(z)


# Display a batch of generated images
plt.figure(figsize=(6,6))
plt.title("Fake Images")
for i in range(32):
    plt.subplot(4, 8, i+1)
    plt.imshow(generated_imgs[i,0].cpu().detach().numpy(), cmap='gray')
    plt.axis('off')
    # plt.title('Generated')
plt.tight_layout()
plt.savefig('/u/erdos/csga/aalfatemi/DLFP/reports/figures/fake_images.png')


# real images
plt.figure(figsize=(6,6))
plt.title("Real Images")
for i in range(imgs.size(0)):  
    plt.subplot(4, 8, i + 1)
    plt.imshow(imgs[i, 0].cpu().detach().numpy(), cmap='gray')
    plt.axis('off')
    # plt.title('Real')
plt.tight_layout()
plt.savefig('/u/erdos/csga/aalfatemi/DLFP/reports/figures/real_images.png')


# loading intgrated data
print('----------- Loading data -----------------')

# load data
data = '/u/erdos/csga/aalfatemi/DLFP/fakedata'

if not os.path.exists(data):
    raise FileNotFoundError(f"The specified dataset path '{data}' does not exist.")

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root=data, transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

loader_train = DataLoader(train_dataset, batch_size=64, shuffle=True)
loader_val = DataLoader(val_dataset, batch_size=64, shuffle=False)

# -------------------- train the classifier ---------------------------------
print('-----------train the classifier pleae wait -----------------')


# Classifier
classifier = ECG_CNN().to(device)

num_epoch = 25

# Train classifier with validation
training_losses, validation_losses = train_classifier(classifier, loader_train, loader_val, num_epoch, device)

# Plot training and validation losses
plt.figure(figsize=(10, 5))
plt.title("Classifier Loss During Training")
plt.plot(training_losses, label="Training Loss")
plt.plot(validation_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig('/u/erdos/csga/aalfatemi/DLFP/reports/figures/classifier_losses.png')


# Evaluate classifier
accuracy = evaluate_classifier(classifier, loader_val, device)
print("Test Accuracy:", accuracy)
