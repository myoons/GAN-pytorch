# torchvision
import torchvision
import torchvision.transforms as transforms

# torch
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

def set_dataloader(args):

    dataset = datasets.ImageFolder(root='./data',
                                            transform=transforms.Compose([
                                            transforms.Resize((args.img_size, args.img_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5], [0.5])
                                            ]))

    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=True
                            )
    
    return dataloader