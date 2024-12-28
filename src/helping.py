from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt
import numpy as np


def find_pig_indices(pig_list: list, filt_for: str, mode="only"):
    pig_list = pig_list[:, 0]
    if mode == "only":
        idx = np.where(pig_list == filt_for)[0]
    if mode == "exclude":
        idx = np.where(pig_list != filt_for)[0]

    print(f"Found {len(idx)} entries.")
    return idx


def add_noise_2d(signal, snr_db=20):  #  SNR definition
    signal_power = np.mean(signal**2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power) * np.random.normal(size=signal.shape)
    return signal + noise


def copy_sample(idx, eit, y, pig, aug_path):
    np.savez(
        aug_path + "sample_{0:06d}.npz".format(idx),
        eit=eit,
        y=y,
        pig=pig,
    )


def augment_sample(idx, eit, y, pig, aug_path):
    np.savez(
        aug_path + "sample_{0:06d}.npz".format(idx),
        eit=add_noise_2d(eit),
        y=y,
        pig=pig,
    )


def lowpass_filter(data, cutoff=10, fs=1000, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data
