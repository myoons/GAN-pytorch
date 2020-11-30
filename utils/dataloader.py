
import numpy as np

# torchvision
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

# torch
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

def set_data_loader(args):

    dataset = torchvision.datasets.ImageFolder(root='./data',
                                            transform=transforms.Compose([
                                            transforms.Resize((args.img_size, args.img_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5], [0.5])
                                            ]))

    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True
                                            )
    
    return dataloader