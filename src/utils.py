import numpy as np
from scipy.signal import resample


def clean_list(x):
    if x[0] == None:
        x[0] = 0
        
    for i in range(1, len(x)):
        if np.isnan(x[i]):
            x[i] = x[i-1]
    return x
    
def resample_non_drop(x, n):
    reflected_signal = np.concatenate((x[::-1], x, x[::-1]))
    resampled_reflected_signal = resample(reflected_signal, 3 * n)
    resampled_signal = resampled_reflected_signal[n:2*n].tolist()
    return resampled_signal
 