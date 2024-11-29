hist_pig_dict = {
    "P_01_PulHyp": 10418,
    "P_02_PulHyp": 12951,
    "P_03_PulHyp": 8151,
    "P_04_PulHyp": 9990,
    "P_05_PulHyp": 10453,
    "P_06_PulHyp": 6937,
    "P_07_PulHyp": 1448,
    "P_08_PulHyp": 6687,
    "P_09_PulHyp": 8263,
    "P_10_PulHyp": 6507,
}

dap_factor = 180
sap_factor = 180
map_factor = 160

import numpy as np

np.random.seed(42)

import glob
import os
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt

from src.data_util import load_preprocess_examples


def copy_sample(idx, eit, y, pig, aug_path):
    np.savez(
        aug_path + "sample_{0:06d}.npz".format(idx),
        eit=eit,
        y=y,
        pig=pig,
    )


def add_noise_2d(signal, snr_db=20):
    signal_power = np.mean(signal**2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power) * np.random.normal(size=signal.shape)
    return signal + noise


def augment_sample(idx, eit, y, pig, aug_path):
    np.savez(
        aug_path + "sample_{0:06d}.npz".format(idx),
        eit=add_noise_2d(eit),
        y=y,
        pig=pig,
    )


max_samples = 15_000

SELECTED_PIG = 1

l_path = "/mnt/servers/roman/EIT/PulHyp_and_SVV_npz/"
pig_path = ["P_{0:02d}_PulHyp".format(i) for i in [SELECTED_PIG]]

aug_path = "/mnt/servers/roman/EIT/PulHyp_augmented_SNR20/" + pig_path[0] + "/"

X, y, clrs_pig = load_preprocess_examples(
    l_path,
    pig_path,
    sap=True,
    get_pig=True,
    shuffle=False,
    norm_eit="block",
)

y[:, 0] = y[:, 0] / dap_factor  # dap normalization
y[:, 1] = y[:, 1] / sap_factor  # sap normalization
y[:, 2] = y[:, 2] / map_factor  # map normalization

# copy available samples
available = np.arange(0, hist_pig_dict[pig_path[0]])

cpus = int(64)
pool = Pool()
for idx in available:
    pool.apply_async(
        copy_sample, args=(idx, X[idx, :, :, 0], y[idx], clrs_pig[idx], aug_path)
    )
pool.close()

# data augmentation with noise of existing samples
augmented = np.arange(hist_pig_dict[pig_path[0]], max_samples)
n_samples_curr_pig = hist_pig_dict[pig_path[0]]
aug_num = max_samples - n_samples_curr_pig
print(f"There are {n_samples_curr_pig}, {aug_num} will be created.")

sel_from_sample = np.random.randint(0, n_samples_curr_pig, size=aug_num)

cpus = int(64)
pool = Pool()
for i, idx in enumerate(augmented):
    pool.apply_async(
        augment_sample,
        args=(
            idx,
            X[sel_from_sample[i], :, :, 0],
            y[sel_from_sample[i]],
            clrs_pig[sel_from_sample[i]],
            aug_path,
        ),
    )
pool.close()
