import os
import math
import argparse
import numpy as np

# torchVision
import torchvision.transforms as transforms
from torchvision.utils import save_image

# torch
import torch

# models
from models.descriminator import Discriminator
from models.generator import Generator

# Parsing Arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument("--lr", type=float, default=5e-4, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
    args = parser.parse_args()
    return args

# main
def main():

    args = parse_args()
    print('Arguments : {}'.format(args))

    image_shape = (args.channels, args.img_size, args.img_size) # Set the shape of image [3, 64, 64]
    device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu") # Wheter GPU is available

    adversarial_loss = torch.nn.BCELoss() # Initialize Loss Function
    generator = Generator(args.latent_dim, image_shape) # Initialize Generator
    discriminator = Discriminator(image_shape) # Initialize Discriminator
    

    