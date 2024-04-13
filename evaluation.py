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
import re
import glob
from hyperparameters import h_parameters, file_paths, critic_x_cfg, signals
import gan_model as gmodel
from dataset import MyDataset
from contextual import contextual_f1_score, contextual_precision, contextual_recall


def load_dataset(file_name):
    f_split = file_name.split('_')
    # dataset_id = f_split[0]
    test_split = int(f_split[4])
    data_path = f"{file_paths['data_dir']}/{file_name}.txt"
    with open(data_path) as f:
        values = re.split(r"\s+", f.read())[1:-1]
    df = pd.DataFrame(values, columns=['value'], dtype=float)
    df['timestamp'] = df.index.astype(int)
    df = df[['timestamp', 'value']]
    df_train = df.iloc[:].reset_index(drop=True)
    df_test = df.iloc[test_split:-h_parameters['window_size'] + 1].reset_index(drop=True)
    return df_train, df_test


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
            # critic_score.append(netCx(recon_image).squeeze().cpu().numpy())
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
    predicted = detect_anomalies_sd(anomaly_score, file_name)
    f_split = file_name.split('_')
    anomaly_start = int(f_split[5])
    anomaly_end = int(f_split[6].strip(''))
    expected = pd.DataFrame([[anomaly_start, anomaly_end]], columns=['start', 'end'])
    return expected, predicted


def detect_anomalies_sd(score, file_name):
    _, df_test = load_dataset(file_name)
    mean_score = score.mean()
    std_score = score.std()

    pred_idx = []
    for i in range(len(score)):
        if score[i] > mean_score:
            pred_idx.append(True)
        else:
            pred_idx.append(False)
    pred_idx = np.array(pred_idx)
    anomaly_seqs = []
    threshold = 0.1
    for k, g in groupby(enumerate(df_test.loc[pred_idx == True].index), key=lambda x: x[0] - x[1]):
        anomaly_seqs.append(list(map(itemgetter(1), g)))
    if len(anomaly_seqs) > 0:
        max_score = [score[seq].max() for seq in anomaly_seqs]
        sort_idx = [i[0] for i in sorted(enumerate(max_score), key=lambda x: x[1], reverse=True)]
        max_score = sorted(max_score, reverse=True)
        anomaly_seqs = [anomaly_seqs[idx] for idx in sort_idx]
        # max_score.append(sorted(score[pred_idx==0], reverse=True)[0])

        descent_rate = [(max_score[idx - 1] - max_score[idx]) / max_score[idx - 1] for idx, _ in
                        enumerate(max_score, start=1) if idx != len(max_score)]
        for i, rate in enumerate(descent_rate):
            # and (max_score[i] < 4 * std_score) and (max_score[i] < 0.95 * max_score[0])
            if rate < threshold:
                del anomaly_seqs[i:]
                break
        start_end_idx = [[min(seq), max(seq)] for seq in anomaly_seqs]
        predicted = [[df_test['timestamp'].loc[idx[0]], df_test['timestamp'].loc[idx[1]]] for idx in start_end_idx]
        predicted = pd.DataFrame(predicted, columns=['start', 'end'])
    else:
        predicted = pd.DataFrame([], columns=['start', 'end'])
    return predicted


def write_to_csv(netE, netG, netCx, netCz, data_group, file_name, time_elapsed):
    predicted, expected = get_anomalies(netE, netG, netCx, netCz, file_name)
    f1_score = contextual_f1_score(expected, predicted, weighted=False)
    precision = contextual_precision(expected, predicted, weighted=False)
    recall = contextual_recall(expected, predicted, weighted=False)
    csv_file_path = f"{file_paths['results_dir']}"
    with open(csv_file_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([data_group, file_name, expected, predicted, f1_score, precision, recall, time_elapsed])
