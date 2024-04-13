# -*- coding: utf-8 -*-

import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from dataset import MyDataset
from torch.autograd import Variable
from orion.evaluation import contextual_f1_score
from pyts.image import RecurrencePlot
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from hyperparameters import file_paths, signals, h_parameters, critic_x_cfg
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import os
import matplotlib.pyplot as plt
import csv
from evaluation import evaluate_model
from importlib import reload
from itertools import groupby
from operator import itemgetter
import threading
import datetime
import gc
import gan_model as gmodel
import gan_utils as gutil

gc.collect()
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True


# You initialize the weights to the normal distribution
# with mean 0 and standard deviation 0.02
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# noinspection PyShadowingNames
def train(data_group, file_name, train_dataset):
    z_dim = h_parameters['z_dim']
    device = h_parameters['device']
    im_chan = h_parameters['im_chan']
    c_lambda_x = h_parameters['c_lambda_x']
    c_lambda_z = h_parameters['c_lambda_z']
    crit_x_repeats = h_parameters['crit_x_repeats']
    crit_z_repeats = h_parameters['crit_z_repeats']
    batch_size = h_parameters['batch_size']
    epochs = h_parameters['epochs']
    # print(len(train_dataset))
    # epochs = int(iterations / (len(train_dataset) / batch_size))

    start_time = datetime.datetime.now()
    netG = gmodel.Generator(z_dim=h_parameters['z_dim'], im_chan=h_parameters['im_chan'],
                            hidden_dim=h_parameters['hidden_dim_G']).to(h_parameters['device'])
    netG.apply(gmodel.weights_init)
    # print(netG)

    netE = gmodel.Encoder(z_dim=h_parameters['z_dim'], im_chan=h_parameters['im_chan'],
                          hidden_dim=h_parameters['hidden_dim_E']).to(h_parameters['device'])
    netE.apply(gmodel.weights_init)
    # print(netE)

    netCx = gmodel.Critic_X(cfg=critic_x_cfg, im_chan=h_parameters['im_chan'],
                            hidden_dim=h_parameters['hidden_dim_Cx']).to(h_parameters['device'])
    netCx.apply(gmodel.weights_init)
    # print(netCx)

    netCz = gmodel.Critic_Z(z_dim=h_parameters['z_dim']).to(h_parameters['device'])
    netCz.apply(gmodel.weights_init)

    E_opt = torch.optim.RMSprop(netE.parameters(), lr=h_parameters['lr'], weight_decay=h_parameters['wd'])
    G_opt = torch.optim.RMSprop(netG.parameters(), lr=h_parameters['lr'], weight_decay=h_parameters['wd'])
    Cx_opt = torch.optim.RMSprop(netCx.parameters(), lr=h_parameters['lr'], weight_decay=h_parameters['wd'])
    Cz_opt = torch.optim.RMSprop(netCz.parameters(), lr=h_parameters['lr'], weight_decay=h_parameters['wd'])

    sched_E = torch.optim.lr_scheduler.MultiStepLR(E_opt, h_parameters['milestones'], gamma=h_parameters['gamma_sched'])
    sched_G = torch.optim.lr_scheduler.MultiStepLR(G_opt, h_parameters['milestones'], gamma=h_parameters['gamma_sched'])
    sched_Cx = torch.optim.lr_scheduler.MultiStepLR(Cx_opt, h_parameters['milestones'],
                                                    gamma=h_parameters['gamma_sched'])
    sched_Cz = torch.optim.lr_scheduler.MultiStepLR(Cz_opt, h_parameters['milestones'],
                                                    gamma=h_parameters['gamma_sched'])

    start_time = datetime.datetime.now()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    no_batches = len(train_loader)
    enc_losses = []
    gen_losses = []
    mse_x_losses = []
    mse_z_losses = []
    critic_x_losses = []
    critic_z_losses = []
    track_enc, track_gen, track_critic_x, track_critic_z = [], [], [], []
    for epoch in range(epochs):
        # train_loader = tqdm(train_loader)
        for i, real in enumerate(train_loader):
            cur_batch_size = len(real)
            # print(real.shape)
            real = real.to(h_parameters['device'])

            # PART 1: train Discriminators: netCx, netCz
            netCx.train()
            netCz.train()
            netE.eval()
            netG.eval()
            for p in netCz.parameters():
                p.requires_grad = True
            for p in netCx.parameters():
                p.requires_grad = True
            for p in netE.parameters():
                p.requires_grad = False
            for p in netG.parameters():
                p.requires_grad = False

            mean_iteration_critic_x_loss = 0
            mean_iteration_critic_z_loss = 0
            for iz in range(crit_z_repeats):
                Cz_opt.zero_grad()
                noise = torch.randn(cur_batch_size, h_parameters['z_dim'], 1, 1, device=h_parameters['device'])
                fake_z = netE(real)
                crit_z_fake_pred = netCz(fake_z.detach())
                crit_z_real_pred = netCz(noise)

                epsilon_z = torch.rand(len(real), h_parameters['z_dim'], 1, 1,
                                       device=h_parameters['device'], requires_grad=True)
                gradient_z = gutil.get_gradient(netCz, noise, fake_z.detach(), epsilon_z, device=h_parameters['device'])
                gp_z = gutil.gradient_penalty(gradient_z)

                crit_z_loss = crit_z_fake_pred.mean() - crit_z_real_pred.mean() + (c_lambda_z * gp_z)

                # Keep track of the average critic loss in this batch
                mean_iteration_critic_z_loss += crit_z_loss.item() / crit_z_repeats

                # writer.add_scalar('loss/crit_z', crit_z_loss, epoch*no_batches+i+iz)
                # Update gradients
                crit_z_loss.backward()
                # Update optimizer
                Cz_opt.step()

            for ix in range(crit_x_repeats):
                Cx_opt.zero_grad()
                noise = torch.randn(cur_batch_size, h_parameters['z_dim'], 1, 1, device=device)
                fake_x = netG(noise)
                crit_x_fake_pred = netCx(fake_x.detach())
                crit_x_real_pred = netCx(real)

                epsilon_x = torch.rand(cur_batch_size, im_chan, 1, 1, device=device, requires_grad=True)
                gradient_x = gutil.get_gradient(netCx, real, fake_x.detach(), epsilon_x, device=device)
                gp_x = gutil.gradient_penalty(gradient_x)

                crit_x_loss = crit_x_fake_pred.mean() - crit_x_real_pred.mean() + (c_lambda_x * gp_x)

                # Keep track of the average critic loss in this batch
                mean_iteration_critic_x_loss += crit_x_loss.item() / crit_x_repeats

                # writer.add_scalar('loss/crit_x', crit_x_loss, epoch*no_batches+i+ix)
                # Update gradients
                crit_x_loss.backward()

                # Update optimizer
                Cx_opt.step()
            critic_x_losses += [mean_iteration_critic_x_loss]
            critic_z_losses += [mean_iteration_critic_z_loss]

            # PART 2: Train Encoder and Generator
            # Fixed Critics:
            # set to exclude effect of dropout or batch_norm during evaluation.
            netCx.eval()
            netCz.eval()
            netE.train()
            netG.train()

            # set this to reduce memory usage, and prevent wrong loss if any.
            for p in netCz.parameters():
                p.requires_grad = False
            for p in netCx.parameters():
                p.requires_grad = False
            for p in netE.parameters():
                p.requires_grad = True
            for p in netG.parameters():
                p.requires_grad = True

            E_opt.zero_grad()
            G_opt.zero_grad()
            # generate noise from Gaussian distribution N(0,1)
            noise = torch.randn(cur_batch_size, h_parameters['z_dim'], 1, 1, device=device)
            fake_x = netG(noise)
            # crit_x_fake_pred = netCx(fake_x.detach())
            crit_x_fake_pred = netCx(fake_x)  # no detach to let gradient back propagate through netG
            crit_x_real_pred = netCx(real)

            fake_z = netE(real)
            # crit_z_fake_pred = netCz(fake_z.detach())
            crit_z_fake_pred = netCz(fake_z)  # no detach to let gradient back propagate through netE
            crit_z_real_pred = netCz(noise)

            loss = torch.nn.MSELoss()
            mse_x_loss = loss(real, netG(fake_z))
            mse_z_loss = loss(noise, netE(fake_x.detach()))  # detach fake_x to prevent backward in netG for this loss.

            gen_loss = -crit_x_fake_pred.mean() + mse_x_loss
            # enc_loss = crit_z_real_pred.mean() - crit_z_fake_pred.mean() + 1 * mse_loss
            enc_loss = - crit_z_fake_pred.mean() + mse_x_loss + mse_z_loss

            # retain_graph for mse_loss between E and G.
            # need retain_graph=True to let the next backward work,
            # otherwise, the tensors will be frozen after updating gradients
            gen_loss.backward(retain_graph=True)
            # enc_loss.backward(retain_graph=True)
            enc_loss.backward()

            # Update the weights
            G_opt.step()
            E_opt.step()

            # Keep track of the average generator loss
            enc_losses += [enc_loss.item()]
            gen_losses += [gen_loss.item()]
            mse_x_losses += [mse_x_loss.item()]
            mse_z_losses += [mse_z_loss.item()]

        sched_E.step()
        sched_G.step()
        sched_Cx.step()
        sched_Cz.step()
        enc_mean = np.mean(enc_losses)
        mse_x_mean = np.mean(mse_x_losses)
        gen_mean = np.mean(gen_losses)
        mse_z_mean = np.mean(mse_z_losses)
        track_enc += [enc_mean]
        track_gen += [gen_mean]
        track_critic_x += [mse_x_mean]
        track_critic_z += [mse_z_mean]
        crit_x_mean = np.mean(critic_x_losses)
        crit_z_mean = np.mean(critic_z_losses)
        print(f"Epoch {epoch}/{epochs} losses: E: {enc_mean}, G: {gen_mean}, "
              f"Cx: {crit_x_mean}, Cz: {crit_z_mean}, "
              f"MSE_X: {mse_x_mean}, MSE_Z:{mse_z_mean}")

    print(f"Time elapsed: {(datetime.datetime.now() - start_time).seconds} seconds")
    torch.save(netE.state_dict(), f'{file_paths["models_dir"]}/netE_{data_group}_{file_name}.pth')
    torch.save(netG.state_dict(), f'{file_paths["models_dir"]}/netG_{data_group}_{file_name}.pth')
    torch.save(netCx.state_dict(), f'{file_paths["models_dir"]}/netCx_{data_group}_{file_name}.pth')
    torch.save(netCz.state_dict(), f'{file_paths["models_dir"]}/netCz_{data_group}_{file_name}.pth')
    time_elapsed = (datetime.datetime.now() - start_time).seconds / 60
    threading.Thread(target=evaluate_model, args=(data_group, file_name, time_elapsed)).start()


if __name__ == '__main__':
    csv_file_path = file_paths['results_dir']
    with open(csv_file_path, 'w+', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["dataset_name", "signal", "expected", "predicted",
                             "f1_score",
                             "precision",
                             "recall",
                             "time_elapsed (minutes)"])
    threads = []
    for data_group, file_names in signals.items():
        for file_name in file_names:
            train_dataset = MyDataset(file_name)
            t = threading.Thread(target=train, args=(data_group, file_name, train_dataset))
            t.start()
            t.join()
    # for thread in threads:
    #     thread.start()
    #     thread.join()

