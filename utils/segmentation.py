import pandas as pd
import numpy as np
import math

def detect_monotonic_segments_ignore_outliers(
    csv_file_path,
    smoothing_window=27,
    outlier_factor=0,
    noise_tolerance=0,
    min_points=20,
    maxval_proc=80,
    cnt_log_cycles=2.5,
):
    """
    Поиск монотонных сегментов (возрастание/убывание) с исключением выбросов.
    Возвращает словарь: {1: [(start_time, end_time), ...], -1: [...]}.
    """
    df = pd.read_csv(csv_file_path).sort_values(by="time").reset_index(drop=True)
    max_pressure = df["pressure"].max()
    threshold_pressure = (maxval_proc / 100.0) * max_pressure
    df["pressure_smooth"] = (
        df["pressure"]
        .rolling(window=smoothing_window, center=True, min_periods=1)
        .median()
    )
    mad = np.median(np.abs(df["pressure"] - df["pressure_smooth"]))
    threshold = outlier_factor * mad
    df["is_outlier"] = np.abs(df["pressure"] - df["pressure_smooth"]) > threshold
    df_clean = df[~df["is_outlier"]].reset_index(drop=True)
    if len(df_clean) < 2:
        return {1: [], -1: []}

    segments = {1: [], -1: []}
    current_trend = None
    current_start_idx = None
    last_valid_index = None
    n = len(df_clean)
    log_threshold = cnt_log_cycles * math.log(10)

    def close_segment(end_idx, trend):
        seg_length = end_idx - current_start_idx + 1
        if seg_length >= min_points:
            segment_data = df_clean.iloc[current_start_idx:end_idx + 1]
            if (segment_data["pressure"] >= threshold_pressure).any():
                start_time = segment_data.iloc[0]["time"]
                end_time = segment_data.iloc[-1]["time"]
                segments[trend].append((start_time, end_time))

    def get_sign_and_change(old_trend, diff, noise_tolerance):
        if diff > 0:
            raw_sign = 1
        elif diff < 0:
            raw_sign = -1
        else:
            raw_sign = 0
        changed = False
        if old_trend is None:
            return raw_sign, False
        if old_trend != 0 and raw_sign != 0:
            if (raw_sign != old_trend) and (abs(diff) > noise_tolerance):
                changed = True
        return raw_sign, changed

    for i in range(1, n):
        time_prev = df_clean.loc[i - 1, "time"]
        time_curr = df_clean.loc[i, "time"]
        if time_prev <= 0 or time_curr <= 0:
            log_diff = 0
        else:
            log_diff = math.log(time_curr) - math.log(time_prev)
        diff = df_clean.loc[i, "pressure_smooth"] - df_clean.loc[i - 1, "pressure_smooth"]
        new_sign, changed = get_sign_and_change(current_trend, diff, noise_tolerance)
        if current_trend is None:
            if new_sign != 0:
                current_trend = new_sign
                current_start_idx = i - 1
                last_valid_index = i - 1
            continue
        log_exceeded = (log_diff > log_threshold)
        if changed or log_exceeded:
            close_segment(last_valid_index, current_trend)
            if log_exceeded:
                current_trend = new_sign if new_sign != 0 else None
                current_start_idx = i
                last_valid_index = i
            else:
                if new_sign != 0:
                    current_trend = new_sign
                    current_start_idx = i - 1
                    last_valid_index = i - 1
                else:
                    current_trend = None
                    current_start_idx = None
                    last_valid_index = None
        else:
            last_valid_index = i
            if new_sign != 0:
                current_trend = new_sign

    if current_trend is not None and last_valid_index is not None and current_start_idx is not None:
        close_segment(last_valid_index, current_trend)

    return segments
