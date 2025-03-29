import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import argrelextrema

def extract_graph_features(df, start_time, end_time, mon=0):

    df = df[(df['time'] >= start_time) & (df['time'] <= end_time)].copy()


    x = df["time"].values
    y = df["pressure"].values
    

    mean_y = np.mean(y)
    std_y = np.std(y)
    range_y = np.max(y) - np.min(y)
    cv_y = std_y / mean_y if mean_y != 0 else 0
    skewness_y = skew(y)
    kurtosis_y = kurtosis(y)


    height = np.max(y) - np.min(y)  
    diff_first_last = y[-1] - y[0]  
    

    num_maxima = len(argrelextrema(y, np.greater)[0])
    num_minima = len(argrelextrema(y, np.less)[0])
    num_extrema = num_maxima + num_minima
    
    angles = np.arctan(np.diff(y) / np.diff(x))
    mean_angle = np.mean(angles)

    segment_lengths = np.abs(np.diff(y)) / range_y if range_y != 0 else np.zeros(len(y)-1)
    mean_segment_length = np.mean(segment_lengths)


    fft_values = np.fft.fft(y)
    fft_amplitudes = np.abs(fft_values)
    dominant_freq = np.argmax(fft_amplitudes[1:]) + 1

    energy_y = np.sum(y**2)


    ln_x = np.log(x + 1e-6)  
    mse_ln = np.mean((ln_x - y) ** 2)


    deltas = np.diff(y)
    mean_delta = np.mean(deltas)
    median_delta = np.median(deltas)
    max_delta = np.max(deltas)
    min_delta = np.min(deltas)
    var_delta = np.var(deltas)
    std_delta = np.std(deltas)
    

    sign_changes = np.sum(np.diff(np.sign(deltas)) != 0)

    features_df = pd.DataFrame({
        # "mean": [mean_y],
        "std": [std_y],
        "range": [range_y],
        "cv": [cv_y],
        "skewness": [skewness_y],
        "kurtosis": [kurtosis_y],
        "num_extrema": [num_extrema],
        "mean_angle": [mean_angle],
        "mean_segment_length": [mean_segment_length],
        #"dominant_freq": [dominant_freq],
        "energy": [energy_y],
        #"height": [height],
        #"diff_first_last": [diff_first_last],
        "mse_ln": [mse_ln],
        #"mean_delta": [mean_delta],
        "median_delta": [median_delta],
        "max_delta": [max_delta],
        "min_delta": [min_delta],
        "var_delta": [var_delta],
        "std_delta": [std_delta],
        #"sign_changes": [sign_changes],
        "mon": [mon]
    })
    
    return features_df