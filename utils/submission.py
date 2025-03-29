import os
import pandas as pd
from .features import extract_graph_features

def submittion_file(razmetka_path):
    df = pd.read_csv(razmetka_path)
    df_ans = pd.DataFrame()
    for i in range(df.shape[0]): 
        mezh_path = df.loc[i, 'filename']
        path = os.path.join('marked_data', mezh_path)
        mini_data = pd.read_csv(path)
        mezh_df = extract_graph_features(mini_data, df.loc[i, 'time_start'], df.loc[i, 'time_stop'])
        mezh_df['target'] = df.loc[i, 'class']
        mezh_df['time_start'] = df.loc[i, 'time_start']
        mezh_df['time_stop'] = df.loc[i, 'time_stop']
        df_ans = pd.concat((df_ans, mezh_df), axis=0, ignore_index=True)
    return df_ans

def create_submission(submission_file: str, recovery_file: str, drop_file: str, output_file: str = "SABMISHEN.csv"):
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