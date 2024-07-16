import os
import pickle

import cv2
import yaml
import argparse

import h5py
import torch
import openslide
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from src.models.CLIP_MIL import CLIP_MIL

def to_percentiles(scores):
    from scipy.stats import rankdata
    result = rankdata(scores.cpu().numpy(), 'average')/len(scores)
    return result

@torch.no_grad()
def creat_heatmap(model, cls_names, wsi_names, wsi_dir, h5_dir, save_dir, device, descriptions="instance", patch_size=512):
    os.makedirs(save_dir, exist_ok=True)
    cmap = plt.get_cmap("jet")
    for wsi_name in wsi_names:
        h5_path = os.path.join(h5_dir, wsi_name + '.h5')
        wsi_path = os.path.join(wsi_dir, wsi_name + '.tiff')

        h5_file = h5py.File(h5_path, 'r')
        coords = np.array(h5_file["coords"], dtype=np.int64)
        features = np.array(h5_file["features"], dtype=np.float32)
        features = torch.from_numpy(features).to(device)

        slide = openslide.open_slide(wsi_path)
        level = slide.get_best_level_for_downsample(64)
        dimension = slide.level_dimensions[level]
        heatmap_downsample = int(slide.level_downsamples[level])

        data = {
            "properties": {
                "heatmap_info": {"heatmap_info": [], "downsample": heatmap_downsample}
            },
            "heatmap_info": {}
        }

        output = model(features.unsqueeze(0))
        if descriptions == "instance":
            attn_raw = output["inst_attn"].cpu()
        elif descriptions == "stain":
            attn_raw = output["stain_attn"].cpu()
        else:
            raise ValueError("descriptions must be instance or stain")
        assert attn_raw.dim() == 4
        attn_raw = torch.softmax(attn_raw, dim=2)
        attn_raw = torch.mean(attn_raw, dim=1).squeeze(0)

        for cls_idx, cls_name in enumerate(cls_names):
            attn = attn_raw[cls_idx, :]
            attn = to_percentiles(attn)
            heatmap = np.zeros(dimension, dtype=np.uint8)
            ps = int(patch_size / heatmap_downsample)
            for i, prob in enumerate(attn):
                coord = (coords[i] / heatmap_downsample).astype("int32")
                heatmap[coord[1]:coord[1] + ps, coord[0]:coord[0] + ps] = int(prob * 255)
            heatmap = (cmap(heatmap) * 255)[:, :, :3].astype(np.uint8)

            heatmap = cv2.GaussianBlur(heatmap, (7, 7), 0)

            data["properties"]["heatmap_info"]["heatmap_info"].append(cls_name)
            data["heatmap_info"][cls_name] = heatmap

        save_path = os.path.join(save_dir, wsi_name + '.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
            f.close()

        print(f"finish {wsi_name}")



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="/home/auwqh/code/CLIP-MIL/examples/config_HER2/clip_instanceStainPooling_ensemble.yaml")
    parser.add_argument('--checkpoint_dir', type=str, default="/home/auwqh/code/CLIP-MIL/save_weights/HER2/clip_instanceStainPooling_ensemble")
    parser.add_argument('--description', type=str, default="instance", choices=["instance", "stain"])
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--csv_dir', type=str, default="/home/auwqh/code/CLIP-MIL/data/HER2_fold")
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--wsi_dir', type=str, default="/home/auwqh/dataset/HER2/WSI/Testing/WSI/")
    parser.add_argument('--h5_dir', type=str, default="/home/auwqh/dataset/HER2/patch/clip_ViTB32/h5_files/")
    parser.add_argument('--save_dir', type=str, default="/home/auwqh/code/CLIP-MIL/heatmap/clip_mil/clip_mil_instanceStainPooling_ensemble")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args =get_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

    model = CLIP_MIL(**config["model"]).to(args.device)
    model.eval()
    checkpoint_path = os.path.join(args.checkpoint_dir, f"model_fold{args.fold}.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    csv_path = os.path.join(args.csv_dir, f"fold_{args.fold}.csv")
    df = pd.read_csv(csv_path)
    wsi_names = sorted(df["test"].dropna().tolist())

    if args.description == "instance":
        cls_names = config["model"]["instance_descriptions"]
    elif args.description == "stain":
        cls_names = config["model"]["stain_descriptions"]
    else:
        raise ValueError("description must be instance or stain")
    creat_heatmap(model, cls_names, wsi_names, args.wsi_dir, args.h5_dir, args.save_dir, args.device, args.description)