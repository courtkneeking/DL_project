# src/gan.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np
import os


# GANs for ECG images
class Generator(nn.Module):
    def __init__(self, latent_dim, image_width,image_height):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.image_shape = (1, image_width, image_height)

        self.init_size = image_width // 4
        self.l1 = nn.Sequential(nn.Linear(self.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img
 

class Discriminator(nn.Module):
    def __init__(self, image_width, image_height):
        super(Discriminator, self).__init__()
        self.image_shape = (1, image_width, image_height)

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )

        ds_size = image_width // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(256 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity


def train_gan(generator, discriminator, train_loader,num_epochs, latent_dim, device):
    # Define the loss functions
    adversarial_loss = torch.nn.BCELoss()

    # Learning rate for optimizers
    lr = 0.0002
    # Define the optimizers 
    optimizer_G = torch.optim.Adam(generator.parameters(), lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr, betas=(0.5, 0.999))


    # Define the number of steps to apply to the discriminator
    number_of_steps = 1

    # Define the number of epochs to wait before saving the model
    checkpoint_interval = 10

    # Lists to keep track of progress
    losses_G = []
    losses_D = []

    # Define the number of epochs to wait before saving the model
    checkpoint_interval = 10
    # ----------
    #  Training
    # ----------
    for epoch in range(num_epochs):
        for i, (imgs, _) in enumerate(train_loader):

            # Adversarial ground truths
            valid = torch.ones(imgs.size(0), 1).to(device)
            fake = torch.zeros(imgs.size(0), 1).to(device)

            # Configure input
            real_imgs = imgs.to(device)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = torch.randn(imgs.shape[0], latent_dim).to(device)

            # Generate a batch of images
            generated_imgs = generator(z)

            # Loss for real images
            real_loss = adversarial_loss(discriminator(real_imgs), valid)

            # Loss for fake images
            fake_loss = adversarial_loss(discriminator(generated_imgs.detach()), fake)

            # Total discriminator loss
            discriminator_loss = (real_loss + fake_loss) / 2

            # Calculate discriminator gradients
            discriminator_loss.backward()

            # Update discriminator weights
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            generated_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            generator_loss = adversarial_loss(discriminator(generated_imgs), valid)

            # Calculate gradients for generator
            generator_loss.backward()

            # Update generator weights
            optimizer_G.step()

            # Print training losses
            if i % 50 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (
                        epoch,
                        num_epochs,
                        i,
                        len(train_loader),
                        discriminator_loss.item(),
                        generator_loss.item(),
                    )
                )


            # If at sample interval save generated image samples
            if epoch % checkpoint_interval == 0:
                losses_G.append(generator_loss.item())
                losses_D.append(discriminator_loss.item())

                # Save losses
                np.save("losses_G.npy", np.array(losses_G))
                np.save("losses_D.npy", np.array(losses_D))

                # Save the model parameters
                torch.save(generator.state_dict(), "generator.pth")
                torch.save(discriminator.state_dict(), "discriminator.pth")
    return losses_G, losses_D, imgs


def generate_images(generator, latent_dim, device, num_images, save_dir='generated_images'):
    # Ensure the generator is in eval mode
    generator.eval()

    # Create directory for saving images if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Generate and save images
    with torch.no_grad():
        for i in range(num_images):
            z = torch.randn(1, latent_dim).to(device)
            generated_img = generator(z)

            save_file = os.path.join(save_dir, f'generated_image_{i}.png')
            save_image(generated_img, save_file, normalize=True)

    print(f"Generated and saved {num_images} images in '{save_dir}'.")
