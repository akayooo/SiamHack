import os
import pandas as pd
import numpy as np
import math

import xgboost
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
import matplotlib.pyplot as plt

# Импорт функций из utils
from utils.submission import submittion_file, create_submission
from utils.segmentation import detect_monotonic_segments_ignore_outliers
from utils.features import extract_graph_features

def main():
    # Загружаем обучающие данные
    razmetka_filename = 'vost_clear_razmetka.csv'
    data_train_all = submittion_file('razmetka_files/' + razmetka_filename)
    data_train_all = data_train_all[data_train_all['target'] != 3]
    data_train_all = data_train_all.drop(['time_start','time_stop','mon'], axis=1)

    # ----------------------------
    # АНСАМБЛЬ ДЛЯ RECOVERY (target 1 и 0)
    data_train_recovery = data_train_all[(data_train_all['target'] == 1) | (data_train_all['target'] == 0)]
    data_train_recovery = data_train_recovery[data_train_recovery['mean_angle'] >= 0]
    y_train_recovery = data_train_recovery['target']
    x_train_recovery = data_train_recovery.drop('target', axis=1)
    
    model_recovery_cb = CatBoostClassifier(
        verbose=False
    )
    model_recovery_xgb = XGBClassifier(
        use_label_encoder=False,
    )
    model_recovery_light = lgb.LGBMClassifier(
        verbose=-1
    )
    
    model_recovery_cb.fit(x_train_recovery, y_train_recovery)
    model_recovery_xgb.fit(x_train_recovery, y_train_recovery)
    model_recovery_light.fit(x_train_recovery, y_train_recovery)

    # ----------------------------
    # АНСАМБЛЬ ДЛЯ DROP (target 2 и 0)
    data_train_drop = data_train_all[(data_train_all['target'] == 2) | (data_train_all['target'] == 0)]
    data_train_drop['target'] = (data_train_drop['target'] == 2).astype(int)
    data_train_drop = data_train_drop[data_train_drop['mean_angle'] < 0]
    y_train_drop = data_train_drop['target']
    x_train_drop = data_train_drop.drop('target', axis=1)
    
    model_drop_cb = CatBoostClassifier(
        verbose=False
    )
    model_drop_xgb = XGBClassifier(
        use_label_encoder=False,
    )
    model_drop_light = lgb.LGBMClassifier(
        verbose=-1
    )
    
    model_drop_cb.fit(x_train_drop, y_train_drop)
    model_drop_xgb.fit(x_train_drop, y_train_drop)
    model_drop_light.fit(x_train_drop, y_train_drop)

    # ----------------------------
    # Обработка тестовых файлов
    folder_path = 'test_files'  # Укажите путь к папке с тестовыми файлами
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

    # ----------------------------
    # Предсказание для DROP (анализ сегментов с mon == -1)
    drop_data = df_test_segments[df_test_segments['mon'] == -1].reset_index(drop=True)
    x_drop = drop_data.drop(['filename','time_start','time_stop','mon'], axis=1)
    drop_probs_cb = model_drop_cb.predict_proba(x_drop)[:, 1]
    drop_probs_xgb = model_drop_xgb.predict_proba(x_drop)[:, 1]
    drop_probs_light = model_drop_light.predict_proba(x_drop)[:, 1]
    drop_avg_probs = (drop_probs_cb + drop_probs_xgb + drop_probs_light) / 3
    drop_pred = (drop_avg_probs > 0.5).astype(int)
    drop_data['target'] = drop_pred
    drop_data = drop_data[drop_data['target'] == 1]
    drop_data.to_csv('razmetka_files/drop_ans.csv', index=False)

    # Предсказание для RECOVERY (анализ сегментов с mon == 1)
    recovery_data = df_test_segments[df_test_segments['mon'] == 1].reset_index(drop=True)
    x_recovery = recovery_data.drop(['filename','time_start','time_stop','mon'], axis=1)
    recovery_probs_cb = model_recovery_cb.predict_proba(x_recovery)[:, 1]
    recovery_probs_xgb = model_recovery_xgb.predict_proba(x_recovery)[:, 1]
    recovery_probs_light = model_recovery_light.predict_proba(x_recovery)[:, 1]
    recovery_avg_probs = (recovery_probs_cb + recovery_probs_xgb + recovery_probs_light) / 3
    recovery_pred = (recovery_avg_probs > 0.5).astype(int)
    recovery_data['target'] = recovery_pred
    recovery_data = recovery_data[recovery_data['target'] == 1]
    recovery_data.to_csv('razmetka_files/recovery_ans.csv', index=False)

    # Создание файла submission
    submission_file = 'razmetka_files/submission_clear.csv'
    recovery_file = 'razmetka_files/recovery_ans.csv'
    drop_file = 'razmetka_files/drop_ans.csv'
    create_submission(submission_file, recovery_file, drop_file, "VOST_SUBMISSEN_ultim_" + razmetka_filename)

if __name__ == "__main__":
    main()
