import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

from src.utils import resample_non_drop
  
def get_differences(x, sigma=8):
    smoothed_x = gaussian_filter1d(x, sigma)
    return np.where((smoothed_x[1:] - smoothed_x[:-1]) >= 0, 1.0, 0.0).tolist()

def difference_process(x, window=1000, sigma=8):
    return get_differences(resample_non_drop(x, window), sigma)

def savgol_normalize(x, window=1000):
    filtered_data = savgol_filter(resample_non_drop(x, window), window_length=20, polyorder=3)
    return (filtered_data - np.mean(filtered_data)).tolist()