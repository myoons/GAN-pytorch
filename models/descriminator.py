# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# numpy
import numpy as np

class Discriminator(nn.Module):

    def __init__(self, image_shape):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(image_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, image):
        image_flat = image.view(image.size(0), -1)
        validity = self.model(image_flat)
        return validity