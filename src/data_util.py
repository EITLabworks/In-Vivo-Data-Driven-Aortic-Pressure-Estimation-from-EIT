import numpy as np
from os.path import join
from glob import glob
from scipy import signal
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder


def z_score(X: np.ndarray) -> np.ndarray:
    """
    Computes the z-score normalization of a given array.

    Parameters
    ----------
    X : np.ndarray
        data array

    Returns
    -------
    np.ndarray
        z-score normalized array
    """
    X_mean = np.mean(X)
    X_std = np.std(X)
    return (X - X_mean) / X_std


def normalize_aorta(y: np.ndarray, invert=False):
    """Normalize aorta pressure (z-score normalization)

    invert:  Denormalize if True
    """

    if invert:
        y = y * 20.0 + 85.0
    else:
        y = (y - 85.0) / 20.0
    return y


def normalize_eit(X: np.ndarray, pigs: np.ndarray, norm_eit: str):
    """Normalize EIT signals (z-score normalization)

    norm_eit : str with 'global' or per 'block'

    """

    if norm_eit == "global":
        mx = np.mean(X, axis=(0, 1))
        sx = np.std(X, axis=(0, 1))
        X = (X - mx) / sx

    elif norm_eit == "block":
        le_p = LabelEncoder()
        le_b = LabelEncoder()
        le_p.fit(pigs[:, 0])

        for p in le_p.classes_:
            idx_p = np.where(pigs[:, 0] == p)
            le_b.fit(np.squeeze(pigs[idx_p, 1]))

            for b in le_b.classes_:
                idx = np.where((pigs[:, 0] == p) & (pigs[:, 1] == b))

                mx = np.mean(X[idx, :, :], axis=(0, 1))
                sx = np.std(X[idx, :, :], axis=(0, 1))
                X[idx, :, :] = (X[idx, :, :] - mx) / sx

    return X


def compute_ap(y: list):
    def min_max_mean(aorta):
        return np.array([np.min(aorta), np.max(aorta), np.mean(aorta)])

    y = [min_max_mean(aorta) for aorta in y]
    return y


def resample_eit(X: list, eit_length: int = 64):
    num_cores = 64
    eit_frame_length = X[0].shape[1]

    def worker(eit_arr, length, eit_frame_length):
        return np.array(
            [signal.resample(eit_arr[:, j], length) for j in range(eit_frame_length)]
        ).T

    X = Parallel(n_jobs=num_cores)(
        delayed(worker)(eit_block, eit_length, eit_frame_length) for eit_block in X
    )

    return X


def quality_checks(X: list, y: list, eit_length=64, aorta_length=1024):
    idx = list()

    # quality checks for EIT data
    for n, eit in enumerate(X):
        if eit.shape[0] > eit_length or eit.shape[0] == 0:
            # print(f"dataset excluded: {eit.shape[0]=}>{eit_length}.")
            idx.append(n)

    # quality checks for aorta data
    for n, aorta in enumerate(y):
        if len(aorta) > aorta_length:
            # print(f"dataset excluded: {aorta.shape[0]=}>{aorta_length}.")
            idx.append(n)

    return idx


def load_examples(X: list, y: list, pigs: list, path: str):
    """
    Load aorta pressure (and eit) signals from npz files in a given path

    X, y, pigs:  list to append the loaded data to
    path:        path to directory with npz files
    """

    print(f"Loading data from {path}")
    files = glob(join(path, "*.npz"), recursive=True)
    files = np.sort(files)
    if len(files) == 0:
        raise Exception("No npz files found in directory")

    for filepath in files:
        tmp = np.load(filepath)
        X.append(tmp["eit_v"])
        y.append(tmp["aorta"])
        pigs.append(tmp["data_info"])

    return X, y, pigs


def load_preprocess_examples(
    data_prefix: str,
    examples: list,
    raw=False,
    sap=False,
    get_pig=False,
    zero_padding=False,
    shuffle=True,
    eit_length=64,
    aorta_length=1024,
    norm_aorta=True,
    norm_eit="none",  # or 'global' or 'block'
    quality_check=False,
):
    # initialize data lists
    X = list()
    y = list()
    pigs = list()

    # load raw data from npz files
    for example in examples:
        X, y, pigs = load_examples(X, y, pigs, join(data_prefix, example))

    # if requested return raw data
    if raw:
        return X, y, pigs

    # perform quality checks and clean dataset
    if quality_check:
        rm_idx = quality_checks(X, y, eit_length, aorta_length)

        for idx in sorted(rm_idx, reverse=True):
            del X[idx]
            del y[idx]
            del pigs[idx]

    # create index for shuffling

    N = len(y)
    if shuffle:
        shuffle = np.random.randint(N, size=N)
    else:
        shuffle = range(N)

    # pre-process EIT signals
    if zero_padding:
        # append zeros to examples to equalize lengths
        X = [
            np.concatenate((sample, np.zeros((eit_length - sample.shape[0], 1024))))
            for sample in X
        ]
    else:
        # resample/interpolate signals to equal length
        X = resample_eit(X, eit_length)

    X = np.array(X)

    # if requested normalize EIT signals
    if norm_eit != "none":
        X = normalize_eit(X, np.array(pigs), norm_eit)

    # append empty axis for CNNs
    X = X[:, :, :, np.newaxis]

    # if requested return SAP values and pigs
    if sap:
        y = compute_ap(y)
        return X[shuffle, ...], np.array(y)[shuffle], np.array(pigs)[shuffle, ...]

    # pre-process aorta signals
    if zero_padding:
        # append constant values to examples to equalize lengths
        y = [
            np.concatenate((sample, sample[-1] * np.ones(aorta_length - len(sample))))
            for sample in y
        ]
    else:
        # resample/interpolate signals to equal length aorta_length
        y = [signal.resample(sample, aorta_length) for sample in y]

    y = np.array(y)

    # if requested normalize aorta signals
    if norm_aorta:
        y = normalize_aorta(y)

    if get_pig:
        # return EIT and aorta signals and pigs
        return X[shuffle, ...], y[shuffle, ...], np.array(pigs)[shuffle, ...]
    else:
        # return EIT and aorta signals
        return X[shuffle, ...], y[shuffle, ...]


def load_data(model_path, pig, dataset, valid_colors=False):
    lpath = f"{model_path}/cross_v_{pig}/valid_test_data.npz"
    source_str = "_".join([f"{key}_{value}" for key, value in config.items()])
    tmp = np.load(lpath, allow_pickle=True)
    y_true = tmp[f"y_{dataset}"]
    y_pred = tmp[f"y_{dataset}_preds"]
    if valid_colors and dataset == "valid":
        y_true_clr = tmp["clrs_valid"]
        return y_true, y_pred, source_str, y_true_clr

    print(f"{y_true.shape=}, {y_pred.shape=}, {source_str=}")
    return y_true, y_pred, source_str


def load_aug_sample(path):
    tmp = np.load(path, allow_pickle=True)
    eit = tmp["eit"]
    y = tmp["y"]
    clr = tmp["pig"]
    return eit, y, clr


def load_augmented_example(
    path: str, pigs: list, sample_skip: int = 0, load_samples: str = "upwards", shuffle=False
):
    """
    load_augmented_example

    Parameters
    ----------
    path : str
        data path
    pigs : list
        list of pigs
    sample_skip : int, optional
        limit for sample loading, by default 0
    load_samples : str, optional
        define the upper or lower limit, by default "upwards" | ["upwards", "downwards"]
    shuffle : bool, optional, by default False
        shuffle the data

    Returns
    -------
    X, y, clr_pig
    """
    X = list()
    y = list()
    clr_pig = list()

    for pig in pigs:
        pig_files = glob(path + pig + "/*.npz")
        pig_files = np.sort(pig_files)
        if sample_skip != 0:
            if load_samples == "upwards":
                pig_files = pig_files[sample_skip:]
            elif load_samples == "downwards":
                pig_files = pig_files[:sample_skip]
            print(
                f"Selected {len(pig_files)} from {pig_files[0]} to {pig_files[-1]} from pig {pig}."
            )
        for file in pig_files:
            xs, ys, ps = load_aug_sample(file)
            X.append(xs)
            y.append(ys)
            clr_pig.append(ps)
    X = np.array(X)
    y = np.array(y)
    clr_pig = np.array(clr_pig)

    N = X.shape[0]
    if shuffle:
        shuffle = np.random.randint(N, size=N)
    else:
        shuffle = range(N)

    X = X[shuffle, ...]
    y = y[shuffle, ...]
    clr_pig = clr_pig[shuffle, ...]
    
    X = np.expand_dims(X, axis=3)
    return X, y, clr_pig
