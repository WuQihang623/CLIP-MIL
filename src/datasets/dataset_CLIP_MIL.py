import os
import re

import torch
import pandas as pd
from torch.utils.data import Dataset

def extract_and_convert_score(item):
    match = re.search(r'\b\d+%?\b', item)
    if match:
        score_str = match.group()
        if '%' in score_str:  # 如果是百分比，去除符号并转换为整数
            return int(score_str.replace('%', '')) * 1  # 转换为整数，这里乘以1是为了确保结果是整数
        else:
            return int(score_str)  # 直接转换为整数
    return None

def find_nearest_index(lst, target):
    return min(range(len(lst)), key=lambda i: abs(lst[i] - target))

class dataset_CLIP_MIL(Dataset):
    def __init__(self, csv_path, clinical_path, bag_prompts, feat_dir, mode):
        self.patient_ids, self.label, self.scores = self.return_splits(csv_path, clinical_path, bag_prompts, mode)
        self.feat_dir = feat_dir

    def pdl1_score(self, patient_id, clinical):
        score = int(clinical.loc[clinical['检查号'] == patient_id, '诊断结果'].values[0])
        return score

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        feat_path = os.path.join(self.feat_dir, patient_id + ".pt")
        feat = torch.load(feat_path, map_location="cpu")
        label = self.label[idx]
        score_id = self.scores[idx]

        return feat, label, score_id

    def __len__(self):
        return len(self.patient_ids)

    def return_splits(self, csv_path, clinical_path, bag_prompts, mode):
        df = pd.read_csv(csv_path)
        if mode == "train":
            patient_ids = df['train'].dropna().tolist()
            label = df['train_label'].dropna().tolist()
        elif mode == "val":
            patient_ids = df['val'].dropna().tolist()
            label = df['val_label'].dropna().tolist()
        else:
            patient_ids = df['test'].dropna().tolist()
            label = df['test_label'].dropna().tolist()

        scores = []
        clinical = pd.read_excel(clinical_path)
        bag_prompts = list(bag_prompts.keys())
        bag_prompts_level = [extract_and_convert_score(prompt) for prompt in bag_prompts]
        for patient_id in patient_ids:
            score = self.pdl1_score(patient_id, clinical)
            score_ids = find_nearest_index(bag_prompts_level, score)
            scores.append(score_ids)

        print("-----label and score----")
        for l, s in zip(label, scores):
            print(l, s)
        return patient_ids, label, scores

if __name__ == '__main__':
    csv_path = "/home/auwqh/code/CLIP-MIL/data/PDL1_fold/fold_0.csv"
    clinical_path = "/home/auwqh/dataset/PDL1/meta_data/excel_data/PDL1_score_clinical.xlsx"
    bag_prompts = {
        "TPS score is 0": 0,
        "TPS score is 1%": 1,
        "TPS score is 5%": 5,
        "TPS score is 10%": 10,
        "TPS score is 20%": 20,
        "TPS score is 30%": 30,
        "TPS score is 40%": 40,
        "TPS score is 50%": 50,
        "TPS score is 60%": 60,
        "TPS score is 70%": 70,
        "TPS score is 80%": 80,
        "TPS score is 90%": 90,
    }
    feat_dir = "/home/auwqh/dataset/PDL1/meta_data/Testing/patch/clip_ViTB32/pt_files/"
    mode = "train"
    dataset = dataset_CLIP_MIL(csv_path, clinical_path, bag_prompts, feat_dir, mode)
    for feat, label, score_ids in dataset:
        print(feat.shape, label, score_ids)
