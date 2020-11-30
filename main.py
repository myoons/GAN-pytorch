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

# dataloader & seed
from utils.dataloader import set_dataloader
from utils.seed import set_seed

# train function
from utils.train import train


# Parsing Arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=5e-4, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=48, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
    parser.add_argument("--seed", type=int, default=777, help="seed number")
    parser.add_argument("--object", type=str, default='person', help='which object to generate')
    args = parser.parse_args()
    print('Arguments : {}'.format(args))
    return args

# main
def main():
    
    # Set seed & dataloader
    set_seed(args)
    dataloader = set_dataloader(args)

    # Set the shape of image [3, 64, 64]
    image_shape = (args.channels, args.img_size, args.img_size)

    # Set device where training will be progressed & base Tensor type (dtype=float32)
    if torch.cuda.is_available():
        device = torch.device("cuda: 0")
        Tensor = torch.cuda.FloatTensor
    else:
        device = torch.device("cpu") 
        Tensor = torch.FloatTensor

    print('Current Device : {} \t Base Tensor : {}'.format(device, Tensor))

    # Initialize Loss Function & Models
    criterion = torch.nn.BCELoss().to(device)
    generator = Generator(args.latent_dim, image_shape).to(device)
    discriminator = Discriminator(image_shape).to(device)

    # Initialize Optimizer for Generator & Discriminator
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    train(args, device, dataloader, criterion, generator, discriminator, optimizer_G, optimizer_D, Tensor)

if __name__ == "__main__":
    global args
    args = parse_args()
    main()

    

    