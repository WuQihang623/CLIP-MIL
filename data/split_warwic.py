import os

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


def split_data(patient_ids, labels, seed):
    k_fold = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(k_fold.split(patient_ids, labels)):
        folds.append(val_idx)

    train_ids = []
    for i in range(3):
        train_ids.extend(folds[i])
    val_ids = folds[-1]
    return train_ids, val_ids


if __name__ == '__main__':
    df = pd.DataFrame(columns=["train", "train_label", "val", "val_label", "test", "test_label"])

    train_df = pd.read_csv("/home/auwqh/dataset/WarwickHER2/HER2_GT/Training_data.csv")
    patient_ids = train_df['CaseNo'].dropna().tolist()
    labels = train_df["HeR2 SCORE"].dropna().tolist()
    train_ids, val_ids = split_data(patient_ids, labels, 42)
    for j, id in enumerate(train_ids):
        df.loc[j, "train"] = str(patient_ids[id]).zfill(2) + "_HER2"
        df.loc[j, "train_label"] = int(labels[id])
    for j, id in enumerate(val_ids):
        df.loc[j, "val"] = str(patient_ids[id]).zfill(2) + "_HER2"
        df.loc[j, "val_label"] = int(labels[id])

    test_df = pd.read_csv("/home/auwqh/dataset/WarwickHER2/HER2_GT/Testing_data.csv")

    test_patient_ids = test_df["CaseNo"].dropna().tolist()
    test_labels = test_df["HeR2 SCORE"].dropna().tolist()
    for j in range(len(test_patient_ids)):
        df.loc[j, "test"] = str(int(test_patient_ids[j])).zfill(2) + "_HER2"
        df.loc[j, "test_label"] = int(test_labels[j])

    df.to_csv("/home/auwqh/code/CLIP-MIL/data/Warwick/fold_0.csv")
