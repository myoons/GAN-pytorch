# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# numpy
import numpy as np

class Generator(nn.Module):

    def __init__(self, latent_dim, image_shape):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.image_shape = image_shape

        def block(input_fea, output_fea, normalize=True):
            layers =[nn.Linear(input_fea, output_fea)]
            if normalize:
                layers.append(nn.BatchNorm1d(output_fea, 0.5))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(image_shape))),
            nn.Tanh()
        )
    
    def forward(self, z):
        image = self.model(z)
        image = image.view(image.size(0), *self.image_shape)
        return image
    