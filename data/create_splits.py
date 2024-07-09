"""
    Split the PD-L1 dataset into 5 fold
"""

import os

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


def split_data(patient_ids, labels, seed, save_dir):
    """
    Split the PD-L1 dataset into 5 folds.
    Args:
        :param patient_ids:
        :param labels:
        :param seed:
        :param save_dir:
        :return:
    """
    os.makedirs(save_dir, exist_ok=True)
    k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(k_fold.split(patient_ids, labels)):
        folds.append(val_idx)

    for i in range(5):
        train_ids = []
        val_ids = []
        test_ids = []
        train_ids.extend(folds[i])
        train_ids.extend(folds[(i + 1) % 5].tolist())
        train_ids.extend(folds[(i + 2) % 5])
        val_ids.extend(folds[(i + 3) % 5])
        test_ids.extend(folds[(i + 4) % 5])
        df = pd.DataFrame(columns=["train", "train_label", "val", "val_label", "test", "test_label"])

        for j, id in enumerate(train_ids):
            df.loc[j, "train"] = patient_ids[id]
            df.loc[j, "train_label"] = labels[id]
        for j, id in enumerate(val_ids):
            df.loc[j, "val"] = patient_ids[id]
            df.loc[j, "val_label"] = labels[id]
        for j, id in enumerate(test_ids):
            df.loc[j, "test"] = patient_ids[id]
            df.loc[j, "test_label"] = labels[id]

        print(f"Fold {i}:")
        print("train label:", df["train_label"].value_counts())
        print("val label:", df["val_label"].value_counts())
        print("test_label:", df["test_label"].value_counts())

        df.to_csv(os.path.join(save_dir, f"fold_{i}.csv"))


def convert_score_to_label(score: np.ndarray):
    label = np.zeros(score.shape, dtype=np.uint8)
    label[score==0] = 0
    label[np.logical_and(score>=1, score<50)] = 1
    label[score>=50] = 2
    return label

### PDL1
# if __name__ == '__main__':
#     csv_path = "/home/auwqh/dataset/PDL1/meta_data/excel_data/PDL1_score_clinical.xlsx"
#     save_dir = "/home/auwqh/code/CLIP-MIL/data/PDL1_fold"
#     df = pd.read_excel(csv_path)
#     patient_ids = df["检查号"].tolist()
#     scores = df["诊断结果"].to_numpy(np.uint8)
#     labels = convert_score_to_label(scores)
#     split_data(patient_ids, labels, 2024, save_dir)


### HER2
if __name__ == '__main__':
    csv_path = "/home/auwqh/code/CLIP-MIL/data/HER2_score_clinical.xlsx"
    save_dir = "/home/auwqh/code/CLIP-MIL/data/HER2_fold"
    df = pd.read_excel(csv_path)
    patient_ids = df["检查号"].tolist()
    labels = df["average"].tolist()
    split_data(patient_ids, labels, 2024, save_dir)

