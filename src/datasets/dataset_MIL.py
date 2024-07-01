import os

import torch
import pandas as pd
from torch.utils.data import Dataset


class dataset_MIL(Dataset):
    def __init__(self, csv_path, feat_dir, mode):
        """
        :param csv_path:
        :param feat_dir:
        :param mode:
        """
        self.patient_ids, self.label = self.return_splits(csv_path, mode)
        self.feat_dir = feat_dir

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        feat_path = os.path.join(self.feat_dir, patient_id + ".pt")
        feat = torch.load(feat_path, map_location="cpu")
        label = self.label[idx]
        return feat, label


    def __len__(self):
        return len(self.patient_ids)

    def return_splits(self, csv_path, mode):
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
        return patient_ids, label
