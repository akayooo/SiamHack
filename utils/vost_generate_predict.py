import os
import math
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import argrelextrema
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

def detect_monotonic_segments_ignore_outliers(
    csv_file_path,
    smoothing_window=27,
    outlier_factor=0,
    noise_tolerance=0,
    min_points=35,
    maxval_proc=80,
    cnt_log_cycles=2,
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

def create_submission(
    submission_file: str,
    recovery_file: str,
    drop_file: str,
    output_file: str = "SABMISHEN.csv"
):
    """
    Читает submission-файл, обновляет в нём столбцы 'drop' и 'recovery'
    из файлов drop_ans.csv и recovery_ans.csv и сохраняет результат.
    """
    df_sub = pd.read_csv(submission_file)
    df_recovery = pd.read_csv(recovery_file)
    df_drop = pd.read_csv(drop_file)

    drop_dict = {}
    for _, row in df_drop.iterrows():
        fname = row["filename"]
        drop_dict.setdefault(fname, []).append([row["time_start"], row["time_stop"]])

    recovery_dict = {}
    for _, row in df_recovery.iterrows():
        fname = row["filename"]
        recovery_dict.setdefault(fname, []).append([row["time_start"], row["time_stop"]])

    if "drop" not in df_sub.columns:
        df_sub["drop"] = "[]"
    if "recovery" not in df_sub.columns:
        df_sub["recovery"] = "[]"

    for idx, row in df_sub.iterrows():
        filename_with_ext = row["file"] + ".csv"
        d_list = drop_dict.get(filename_with_ext, [])
        r_list = recovery_dict.get(filename_with_ext, [])
        df_sub.at[idx, "drop"] = str(d_list) if d_list else "[]"
        df_sub.at[idx, "recovery"] = str(r_list) if r_list else "[]"

    df_sub.to_csv(output_file, index=False)
    print(f"Обновлённый submission сохранён в {output_file}")

def extract_graph_features(df, start_time, end_time, mon=0):
    """
    Считает набор статистических, формообразующих и прочих признаков
    для фрагмента ряда [start_time, end_time].
    """
    df = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
    
    x = df["time"].values - df["time"].values[0]
    y = df["pressure"].values - df["pressure"].values[0]
    
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
    angles = np.arctan(np.diff(y) / np.diff(x)) if len(x) > 1 else np.array([0.0])
    mean_angle = np.mean(angles) if angles.size > 0 else 0.0
    segment_lengths = np.abs(np.diff(y)) / range_y if range_y != 0 else np.zeros(len(y)-1)
    mean_segment_length = np.mean(segment_lengths) if segment_lengths.size > 0 else 0.0
    fft_values = np.fft.fft(y)
    fft_amplitudes = np.abs(fft_values)
    dominant_freq = np.argmax(fft_amplitudes[1:]) + 1 if len(fft_amplitudes) > 1 else 0
    energy_y = np.sum(y**2)
    ln_x = np.log(x + 1e-6)
    mse_ln = np.mean((ln_x - y)**2)
    deltas = np.diff(y)
    mean_delta = np.mean(deltas) if deltas.size > 0 else 0.0
    median_delta = np.median(deltas) if deltas.size > 0 else 0.0
    max_delta = np.max(deltas) if deltas.size > 0 else 0.0
    min_delta = np.min(deltas) if deltas.size > 0 else 0.0
    var_delta = np.var(deltas) if deltas.size > 0 else 0.0
    std_delta = np.std(deltas) if deltas.size > 0 else 0.0
    sign_changes = np.sum(np.diff(np.sign(deltas)) != 0) if deltas.size > 1 else 0

    features_df = pd.DataFrame({
        "mean": [mean_y],
        "std": [std_y],
        "range": [range_y],
        "cv": [cv_y],
        "skewness": [skewness_y],
        "kurtosis": [kurtosis_y],
        "num_extrema": [num_extrema],
        "mean_angle": [mean_angle],
        "mean_segment_length": [mean_segment_length],
        "dominant_freq": [dominant_freq],
        "energy": [energy_y],
        "height": [height],
        "diff_first_last": [diff_first_last],
        "mse_ln": [mse_ln],
        "mean_delta": [mean_delta],
        "median_delta": [median_delta],
        "max_delta": [max_delta],
        "min_delta": [min_delta],
        "var_delta": [var_delta],
        "std_delta": [std_delta],
        "sign_changes": [sign_changes],
        "mon": [mon]
    })
    return features_df

def main():
    # Загружаем обучающие данные
    data_train_all = pd.read_csv('vost_razmetka_with_features.csv')
    data_train_all = data_train_all[data_train_all['target'] != 3]
    data_train_all = data_train_all.drop(['time_start','time_stop','mon'], axis=1)


    # АНСАМБЛЬ ДЛЯ RECOVERY (target 1 и 0)
    # Фильтруем данные для восстановления
    data_train_recovery = data_train_all[(data_train_all['target'] == 1) | (data_train_all['target'] == 0)]
    data_train_recovery = data_train_recovery[data_train_recovery['mean_angle'] >= 0]
    y_train_recovery = data_train_recovery['target']
    x_train_recovery = data_train_recovery.drop('target', axis=1)
    
    # Модель CatBoost для recovery
    model_recovery_cb = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.01,
        depth=8,
        bootstrap_type='Bernoulli',
        subsample=0.8,
        rsm=0.8,
        l2_leaf_reg=1.0,
        random_seed=42,
        verbose=False
    )
    # Модель XGBoost для recovery
    model_recovery_xgb = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.01,
        reg_lambda=1.0,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # Обучение моделей
    model_recovery_cb.fit(x_train_recovery, y_train_recovery)
    model_recovery_xgb.fit(x_train_recovery, y_train_recovery)


    # АНСАМБЛЬ ДЛЯ DROP (target 2 и 0)
    data_train_drop = data_train_all[(data_train_all['target'] == 2) | (data_train_all['target'] == 0)]
    data_train_drop['target'] = (data_train_drop['target'] == 2).astype(int)
    data_train_drop = data_train_drop[data_train_drop['mean_angle'] < 0]
    y_train_drop = data_train_drop['target']
    x_train_drop = data_train_drop.drop('target', axis=1)
    
    # Модель CatBoost для drop
    model_drop_cb = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.01,
        depth=8,
        bootstrap_type='Bernoulli',
        subsample=0.8,
        rsm=0.8,
        l2_leaf_reg=1.0,
        random_seed=42,
        verbose=False
    )
    # Модель XGBoost для drop
    model_drop_xgb = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.01,
        reg_lambda=1.0,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # Обучение моделей
    model_drop_cb.fit(x_train_drop, y_train_drop)
    model_drop_xgb.fit(x_train_drop, y_train_drop)


    # Обработка тестовых файлов
    folder_path = '/Users/savinovsvatoslav/Code/skvazhina_hack/SiamHack/Sabmishalka/test_files'  # <-- укажите путь к папке с тестовыми файлами
    df_test_segments = pd.DataFrame()

    for filename in os.listdir(folder_path):
        if not filename.endswith('.csv'):
            continue
        full_path = os.path.join(folder_path, filename)
        seg = detect_monotonic_segments_ignore_outliers(full_path)
        mini_data = pd.read_csv(full_path)
        for interval in seg[1]:
            feats = extract_graph_features(mini_data, interval[0], interval[1], mon=1)
            feats['time_start'] = interval[0]
            feats['time_stop'] = interval[1]
            feats['filename'] = filename
            df_test_segments = pd.concat((df_test_segments, feats))
        for interval in seg[-1]:
            feats = extract_graph_features(mini_data, interval[0], interval[1], mon=-1)
            feats['time_start'] = interval[0]
            feats['time_stop'] = interval[1]
            feats['filename'] = filename
            df_test_segments = pd.concat((df_test_segments, feats))


    # Предсказание для DROP (анализ сегментов с mon == -1)
    drop_data = df_test_segments[df_test_segments['mon'] == -1].reset_index(drop=True)
    x_drop = drop_data.drop(['filename','time_start','time_stop','mon'], axis=1)
    # Получаем вероятности от обеих моделей
    drop_probs_cb = model_drop_cb.predict_proba(x_drop)[:, 1]
    drop_probs_xgb = model_drop_xgb.predict_proba(x_drop)[:, 1]
    # Усредняем вероятности и порог 0.5 для получения финального решения
    drop_avg_probs = (drop_probs_cb + drop_probs_xgb) / 2
    drop_pred = (drop_avg_probs > 0.5).astype(int)
    drop_data['target'] = drop_pred
    drop_data = drop_data[drop_data['target'] == 1]
    drop_data.to_csv('drop_ans.csv', index=False)

    # Предсказание для RECOVERY (анализ сегментов с mon == 1)
    recovery_data = df_test_segments[df_test_segments['mon'] == 1].reset_index(drop=True)
    x_recovery = recovery_data.drop(['filename','time_start','time_stop','mon'], axis=1)
    recovery_probs_cb = model_recovery_cb.predict_proba(x_recovery)[:, 1]
    recovery_probs_xgb = model_recovery_xgb.predict_proba(x_recovery)[:, 1]
    recovery_avg_probs = (recovery_probs_cb + recovery_probs_xgb) / 2
    recovery_pred = (recovery_avg_probs > 0.5).astype(int)
    recovery_data['target'] = recovery_pred
    recovery_data = recovery_data[recovery_data['target'] == 1]
    recovery_data.to_csv('recovery_ans.csv', index=False)

    # Создание файла submission
    submission_file = 'submission_clear.csv'
    recovery_file = 'recovery_ans.csv'
    drop_file = 'drop_ans.csv'
    create_submission(submission_file, recovery_file, drop_file, "VOST_SUBMISSEN.csv")

if __name__ == "__main__":
    main()
