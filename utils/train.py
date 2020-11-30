import os
import numpy as np

# torch
import torch
from torch.autograd import Variable

# torchvision
import torchvision
from torchvision.utils import save_image

# time
import time

def train(args, device, dataloader, criterion, generator, discriminator, optimizer_G, optimizer_D, Tensor):

    os.makedirs('images', exist_ok=True)
    experiment_time = time.time()

    for epoch in range(args.n_epochs):
        for idx, data in enumerate(dataloader):
            images, labels = data[0].to(device), data[1].to(device)

            # Adversarial ground truths
            valid = Variable(Tensor(images.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(images.size(0), 1).fill_(0.0), requires_grad=False)

            # Configure Input
            real_images = Variable(images.type(Tensor))

            ###################
            # Train Generator #
            ###################

            optimizer_G.zero_grad()

            # Sampel noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (images.size(0), args.latent_dim))))

            # Generate a batch of images
            gen_images = generator(z)

            # Loss measures generator's ability to fool the discriminator
            loss_G = criterion(discriminator(gen_images), valid)

            loss_G.backward()
            optimizer_G.step()

            #######################
            # Train Discriminator #
            #######################

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            loss_real = criterion(discriminator(real_images), valid)
            loss_fake = criterion(discriminator(gen_images.detach()), fake)
            loss_D = (loss_real + loss_fake) / 2

            loss_D.backward()
            optimizer_D.step()

            if idx%20 == 0 :
                print('[Epoch {:d}/{:d}] \t [Batch {:d}/{:d}] \t [Loss_G : {:.4f}] \t [Loss_D : {:0.4f}]'
                    .format(epoch, args.n_epochs, idx, len(dataloader), loss_G.item(), loss_D.item()))

            batches_done = epoch * len(dataloader) + idx

            if batches_done % args.sample_interval == 0:
                print('Save sample Image')
                save_image(gen_images.data[:25], 'images/{}/{:d}.png'.format(experiment_time, batches_done), nrow=5, normalize=True)
    
    print('Everything Done.. Saving Model')

    # Setting the Path to save model
    PATH_base = './{}'.format(time.time())
    PATH_G = PATH_base + '/generator.pth'
    PATH_D = PATH_base + '/discriminator.pth'

    # Save Both Generator & Discriminator
    torch.save(generator.state_dict(), PATH_G)
    torch.save(discriminator.state_dict(), PATH_D)

