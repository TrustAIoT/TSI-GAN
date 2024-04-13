from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from scipy.stats import zscore
from itertools import groupby
from operator import itemgetter
from scipy.signal import find_peaks
import numpy as np
import pandas as pd
import torch
import csv
import glob
from hyperparameters import h_parameters, file_paths, critic_x_cfg, signals
import gan_model as gmodel
from dataset import MyDataset
from contextual import contextual_f1_score, contextual_precision, contextual_recall


def evaluate_model(data_group, file_name, time_elapsed):
    netE_path = glob.glob(f'{file_paths["models_dir"]}/netE_{data_group}_{file_name}.pth', recursive=True)[0]
    netG_path = glob.glob(f'{file_paths["models_dir"]}/netG_{data_group}_{file_name}.pth', recursive=True)[0]
    netCx_path = glob.glob(f'{file_paths["models_dir"]}/netCx_{data_group}_{file_name}.pth',
                           recursive=True)[0]
    netCz_path = glob.glob(f'{file_paths["models_dir"]}/netCz_{data_group}_{file_name}.pth',
                           recursive=True)[0]
    netE_state_dict = torch.load(netE_path)
    netG_state_dict = torch.load(netG_path)
    netCx_state_dict = torch.load(netCx_path)
    netCz_state_dict = torch.load(netCz_path)
    netE = gmodel.Encoder(z_dim=h_parameters['z_dim'], im_chan=h_parameters['im_chan'],
                          hidden_dim=h_parameters['hidden_dim_E']).to(h_parameters['device'])
    netG = gmodel.Generator(z_dim=h_parameters['z_dim'], im_chan=h_parameters['im_chan'],
                            hidden_dim=h_parameters['hidden_dim_G']).to(h_parameters['device'])
    netCx = gmodel.Critic_X(cfg=critic_x_cfg, im_chan=h_parameters['im_chan'],
                            hidden_dim=h_parameters['hidden_dim_Cx']).to(h_parameters['device'])
    netCz = gmodel.Critic_Z(z_dim=h_parameters['z_dim']).to(h_parameters['device'])
    netE.load_state_dict(netE_state_dict)
    netG.load_state_dict(netG_state_dict)
    netCx.load_state_dict(netCx_state_dict)
    netCz.load_state_dict(netCz_state_dict)
    netE = netE.cuda()
    netG = netG.cuda()
    netCx = netCx.cuda()
    netCz = netCz.cuda()
    write_to_csv(netE, netG, netCx, netCz, data_group, file_name, time_elapsed)


def get_score(netE, netG, netCx, netCz, test_dataset):
    netE.eval()
    netG.eval()
    netCx.eval()
    netCz.eval()
    rec_gaf = []
    rec_rp = []
    critic_score = []
    test_loader = torch.utils.data.DataLoader(test_dataset)

    with torch.no_grad():
        for i, real in enumerate(test_loader):
            real = real.to(h_parameters['device'])
            recon_image = netG(netE(real).detach())
            mse_ = torch.sum((real - recon_image) ** 2, dim=(2, 3))
            rec_gaf += list(mse_[:, 0].cpu().numpy())
            rec_rp += list(mse_[:, 1].cpu().numpy())
            # mse = torch.sum((real - recon_image) ** 2, dim=(1, 2, 3))
            critic_score.append(netCx(recon_image).squeeze().cpu().numpy())
        rec_gaf = MinMaxScaler((0, 10)).fit_transform(np.array(rec_gaf).reshape(-1, 1)).flatten()
        rec_rp = MinMaxScaler((0, 10)).fit_transform(np.array(rec_rp).reshape(-1, 1)).flatten()
        return rec_gaf, rec_rp


def get_anomalies(netE, netG, netCx, netCz, file_name):
    test_dataset = MyDataset(file_name, is_test=True)
    rec_gaf, rec_rp = get_score(netE, netG, netCx, netCz, test_dataset)
    _, s_rec_gaf = sm.tsa.filters.hpfilter(rec_gaf)
    _, s_rec_rp = sm.tsa.filters.hpfilter(rec_rp)
    peaks, _ = find_peaks(s_rec_gaf, height=0)
    gaf_peaks = np.sort(s_rec_gaf[peaks])[::-1]
    gaf_peak_diff = (gaf_peaks[0] - gaf_peaks[1]) / gaf_peaks[0]
    peaks, _ = find_peaks(s_rec_rp, height=0)
    rp_peaks = np.sort(s_rec_rp[peaks])[::-1]
    rp_peak_diff = (rp_peaks[0] - rp_peaks[1]) / rp_peaks[0]
    anomaly_score = (gaf_peak_diff * s_rec_gaf) + (rp_peak_diff * s_rec_rp)
    f_split = file_name.split('_')
    test_split = int(f_split[4])
    anomaly_start = int(f_split[5])
    anomaly_end = int(f_split[6].strip(''))
    predicted = np.argmax(anomaly_score) + test_split
    # predicted = pd.DataFrame([[predicted - 5, predicted + 5]], columns=['start', 'end'])
    # expected = pd.DataFrame([[anomaly_start, anomaly_end]], columns=['start', 'end'])
    predicted = [[predicted - 50, predicted + 50]]
    expected = [[anomaly_start, anomaly_end]]
    return expected, predicted


def write_to_csv(netE, netG, netCx, netCz, data_group, file_name, time_elapsed):
    predicted, expected = get_anomalies(netE, netG, netCx, netCz, file_name)
    f1_score = contextual_f1_score(expected, predicted, weighted=False)
    precision = contextual_precision(expected, predicted, weighted=False)
    recall = contextual_recall(expected, predicted, weighted=False)
    csv_file_path = f"{file_paths['results_dir']}"
    with open(csv_file_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([data_group, file_name, expected, predicted, f1_score, precision, recall, time_elapsed])
