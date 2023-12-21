import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter, butter, filtfilt

from src.utils import resample_non_drop, resample_normalize

def normalize(x):
    return (x - np.mean(x)) / np.std(x)

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def get_differences(x, sigma=8):
    smoothed_x = gaussian_filter1d(x, sigma)
    return np.where((smoothed_x[1:] - smoothed_x[:-1]) >= 0, 1.0, 0.0).tolist()

def difference_process(x, window=1000, sigma=8):
    return get_differences(resample_non_drop(x, window), sigma)

def savgol_normalize(x, window=1000):
    filtered_data = savgol_filter(resample_non_drop(x, window), window_length=20, polyorder=3)
    return (filtered_data - np.mean(filtered_data)).tolist()

def gaussian_normalize(x, window=1000, sigma=8):
    filtered_data = gaussian_filter1d(resample_non_drop(x, window), sigma)
    return (filtered_data - np.mean(filtered_data)).tolist()

def rhyme_func(origin_signal, window=1000):
    origin_signal = resample_normalize(origin_signal, window)
    cutoff = 1/300
    order = 3
    hp_signal = highpass_filter(origin_signal, cutoff=cutoff, fs=1, order=order)
    lp_signal = origin_signal - hp_signal
    sg_hp_signal = savgol_filter(hp_signal, window_length=60, polyorder=5)
    # rhyme_signal = sg_hp_signal*np.abs(sg_hp_signal)*200 + lp_signal
    rhyme_signal = normalize(sg_hp_signal * np.abs(sg_hp_signal)) + normalize(lp_signal) * 0.5
    return rhyme_signal.tolist()