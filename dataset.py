from torch.utils.data import Dataset
import torch
import requests
import pandas as pd
import os
import numpy as np
import re
import json
import zipfile
from hyperparameters import file_paths, h_parameters
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def find_start_t(df, window_size=24):
    fw = df['value'].iloc[:window_size]
    return fw.argmin()


def generate_windows(df: pd.DataFrame, start_id=0, col_name='value_scale',
                     window_size=24, step_size=24):
    """Generate windows from pd.DataFrame
    Agrs:
        df: Dataset
        start_id: starting index to generate data window seqs.
        col_name: name of column in df.
        window_size: size of generating window seqs
        step_size: step of sliding windows.

    Outputs:
        seqs: a list of windows
    """
    seqs = []
    L = len(df)
    i = start_id
    while i < L - window_size + 1:
        seqs.append(df[col_name][i:i + window_size].to_numpy())
        i += step_size
    seqs = np.asarray(seqs)
    return seqs


def transform_normalize(X, mu, sigma):
    """This is mimic idea of torchvision.transforms.Normalize(mean, std)
    will return X'=(X-mu)/sigma
    Agrs:
        X: is a numpy array
        mu: expected mean value
        sigma: expected std variance
    Output:
        X'=(X-mu)/sigma
    """
    return (X - mu) / sigma


class MyDataset(Dataset):
    def __init__(self, file_name, is_test=False, transform=None):
        self.file_name = file_name
        train_arr, test_arr = self._load_data()
        if is_test:
            self.data = torch.from_numpy(test_arr)
        else:
            self.data = torch.from_numpy(train_arr)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        if self.transform:
            d = self.transform(d)
        return d.float()

    def _load_data(self):
        data_dir = file_paths['data_dir']
        file_name = self.file_name
        file_path = f"{data_dir}/numpy/{file_name}_w{h_parameters['window_size']}_s{h_parameters['step_size']}.npz"
        if os.path.exists(file_path):
            load = np.load(file_path)
            train_arr = load['train_arr']
            test_arr = load['test_arr']
        else:
            # preprocess and save data
            data_file_path = f"{data_dir}/{file_name}.txt"
            f_split = file_name.split('_')
            # dataset_id = f_split[0]
            test_split = int(f_split[4])
            anomaly_start = int(f_split[5])
            anomaly_end = int(f_split[6].strip(''))
            with open(data_file_path) as f:
                values = re.split(r"\s+", f.read())[1:-1]
            df = pd.DataFrame(values, columns=['value'], dtype=float)
            df['timestamp'] = df.index.astype(int)
            df = df[['timestamp', 'value']]
            expected = pd.DataFrame([[anomaly_start, anomaly_end]],
                                    columns=['start', 'end'])
            df['value_scale'] = MinMaxScaler((0, 1)).fit_transform(df['value'].to_numpy().reshape(-1, 1))
            df_train = df.iloc[:].reset_index(drop=True)
            df_test = df.iloc[test_split:].reset_index(drop=True)
            start_id_train = 0  # find_start_t(df_train)
            start_id_test = 0  # find_start_t(df_test)
            step_size = h_parameters['step_size']
            window_size = h_parameters['window_size']
            train_seqs = generate_windows(df_train, start_id=start_id_train,
                                          col_name='value_scale', window_size=window_size,
                                          step_size=step_size)
            test_seqs = generate_windows(df_test, start_id=start_id_test,
                                         col_name='value_scale', window_size=window_size,
                                         step_size=step_size)
            from pyts.image import GramianAngularField
            gaf = GramianAngularField().fit(train_seqs)
            train_X_GAF = gaf.transform(train_seqs)
            test_X_GAF = gaf.transform(test_seqs)

            from pyts.image import RecurrencePlot
            recPlt = RecurrencePlot()
            train_X_RP = recPlt.fit_transform(train_seqs)
            test_X_RP = recPlt.transform(test_seqs)
            train_X_RP_sc = transform_normalize(train_X_RP, 0.5, 0.5)
            test_X_RP_sc = transform_normalize(test_X_RP, 0.5, 0.5)

            # stacking the two array into a single array for training
            train_arr = np.stack((train_X_GAF, train_X_RP_sc), axis=1)
            # stacking the two array into a single array for testing
            test_arr = np.stack((test_X_GAF, test_X_RP_sc), axis=1)
            np.savez_compressed(file_path, train_arr=train_arr, test_arr=test_arr)

        return train_arr, test_arr
