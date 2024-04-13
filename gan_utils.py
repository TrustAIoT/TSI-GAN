import torch
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import argparse
import datetime


def get_gradient(crit, real, fake, epsilon, device):
    '''
    Return the gradient of the critic's scores with respect to mixes of real and fake images.
    Parameters:
        crit: the critic model
        real: a batch of real images
        fake: a batch of fake images
        epsilon: a vector of the uniformly random proportions of real/fake per mixed image
    Returns:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    '''
    # print("real={}, fake={}, epsilon={}".format(real.shape, fake.shape, epsilon.shape))
    # Mix the images together
    mixed_images = Variable((real * epsilon + fake * (1 - epsilon)), requires_grad=True)

    # Calculate the critic's scores on the mixed images
    mixed_scores = crit(mixed_images)

    # Take the gradient of the scores with respect to the images
    # gradient = torch.autograd.grad(
    #     # Note: You need to take the gradient of outputs with respect to inputs.
    #     # This documentation may be useful, but it should not be necessary:
    #     # https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad
    #     #### START CODE HERE ####
    #     inputs=mixed_images,
    #     outputs=mixed_scores,
    #     #### END CODE HERE ####
    #     # These other parameters have to do with the pytorch autograd engine works
    #     grad_outputs=torch.ones_like(mixed_scores),
    #     create_graph=True,
    #     retain_graph=True,
    # )[0]
    # mixed_scores.mean().backward()
    # gradient = mixed_images.grad

    # adopt from Caogang: https://github.com/caogang/wgan-gp/blob/master/gan_toy.py
    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    # print(gradient.shape)
    return gradient


def gradient_penalty(gradient):
    '''
    Return the gradient penalty, given a gradient.
    Given a batch of image gradients, you calculate the magnitude of each image's gradient
    and penalize the mean quadratic distance of each magnitude to 1.
    Parameters:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    Returns:
        penalty: the gradient penalty
    '''
    # Flatten the gradients so that each row captures one image
    gradient = gradient.reshape(len(gradient), -1)

    # Calculate the magnitude of every row
    gradient_norm = gradient.norm(2, dim=1)

    # Penalize the mean squared distance of the gradient norms from 1
    penalty = ((gradient_norm - 1) ** 2).mean()
    # print(penalty)
    return penalty


def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    '''
    Return the loss of a critic given the critic's scores for fake and real images,
    the gradient penalty, and gradient penalty weight.
    Parameters:
        crit_fake_pred: the critic's scores of the fake images
        crit_real_pred: the critic's scores of the real images
        gp: the unweighted gradient penalty
        c_lambda: the current weight of the gradient penalty
    Returns:
        crit_loss: a scalar for the critic's loss, accounting for the relevant factors
    '''
    crit_loss = crit_fake_pred.mean() - crit_real_pred.mean() + (c_lambda * gp)
    # print(crit_loss)
    return crit_loss


def get_gen_loss(crit_x_fake_pred, crit_x_real_pred, mse):
    '''
    Return the loss of a generator given the critic's scores of the generator's fake images.
    Parameters:
        crit_fake_pred: the critic's scores of the fake images
    Returns:
        gen_loss: a scalar loss value for the current batch of the generator
    '''
    enc_loss = mse + crit_x_real_pred.mean() - crit_x_fake_pred.mean()
    return enc_loss


def get_enc_loss(crit_z_fake_pred, crit_z_real_pred, mse):
    dec_loss = mse + crit_z_real_pred.mean() - crit_z_fake_pred.mean()
    return dec_loss


class MyDataset(Dataset):
    def __init__(self, numpy_array, transform=None):
        self.data = torch.from_numpy(numpy_array)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        if self.transform:
            d = self.transform(d)
        return d.float()
