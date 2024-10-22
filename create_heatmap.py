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
from src.models.ABMIL import ABMIL
from src.models.CLAM import CLAM_MB
from src.models.TransMIL import TransMIL
from src.models.MPL_MIL import MPL_MIL

def to_percentiles(scores):
    from scipy.stats import rankdata
    result = rankdata(scores.cpu().numpy(), 'average')/len(scores)
    return result

@torch.no_grad()
def creat_heatmap(model, cls_names, wsi_names, wsi_dir, h5_dir, save_dir, device, descriptions="instance", patch_size=512, ext=".tiff"):
    os.makedirs(save_dir, exist_ok=True)
    cmap = plt.get_cmap("jet")
    for wsi_name in wsi_names:
        h5_path = os.path.join(h5_dir, wsi_name + '.h5')
        wsi_path = os.path.join(wsi_dir, wsi_name + ext)

        h5_file = h5py.File(h5_path, 'r')
        coords = np.array(h5_file["coords"], dtype=np.int64)
        features = np.array(h5_file["features"], dtype=np.float32)
        features = torch.from_numpy(features).to(device)

        slide = openslide.open_slide(wsi_path)
        level = slide.get_best_level_for_downsample(64)
        dimension = slide.level_dimensions[level]
        heatmap_downsample = int(slide.level_downsamples[level])

        thumb = slide.read_region((0, 0), level, dimension).convert("RGB")
        thumb = np.array(thumb, dtype=np.uint8)

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

        print(attn_raw.shape)

        ### 绘制类似CLIP的那种 attention map

        attn_raw = torch.softmax(attn_raw, dim=2)
        if args.head == -1:
            attn_raw = torch.mean(attn_raw, dim=1).squeeze(0)
        else:
            attn_raw = attn_raw[:, args.head].squeeze(0)

        print(attn_raw.shape)

        for cls_idx, cls_name in enumerate(cls_names):
            attn = attn_raw[cls_idx, :]
            attn = to_percentiles(attn)

            heatmap = np.zeros((dimension[1], dimension[0]), dtype=np.uint8)
            ps = int(patch_size / heatmap_downsample)
            for i, prob in enumerate(attn):
                coord = (coords[i] / heatmap_downsample).astype("int32")
                heatmap[coord[1]:coord[1] + ps, coord[0]:coord[0] + ps] = int(prob * 255)

            zero_mask = heatmap == 0
            heatmap = (cmap(heatmap) * 255)[:, :, :3].astype(np.uint8)
            heatmap[zero_mask] = [255, 255, 255]

            heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)

            data["properties"]["heatmap_info"]["heatmap_info"].append(cls_name)
            data["heatmap_info"][cls_name] = heatmap

            heatmap = (0.7 * thumb + 0.3 * heatmap).astype("uint8")
            image_dir = os.path.join(save_dir, wsi_name)
            os.makedirs(image_dir, exist_ok=True)
            Image.fromarray(heatmap).save(os.path.join(image_dir, cls_name + ".png"))

        image_dir = os.path.join(save_dir, wsi_name)
        Image.fromarray(thumb).save(os.path.join(image_dir, "thumb.png"))

        save_path = os.path.join(save_dir, wsi_name + '.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
            f.close()

        print(f"finish {wsi_name}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2/clip_numhead(4)/clip_group_ensemble/config.yaml")
    parser.add_argument('--checkpoint_dir', type=str, default="/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2/clip_numhead(4)/clip_group_ensemble")
    parser.add_argument('--description', type=str, default="instance", choices=["instance", "stain", "None"])
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--csv_dir', type=str, default="/home/auwqh/code/CLIP-MIL/data/HER2_fold")
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--wsi_dir', type=str, default="/home/auwqh/dataset/HER2/WSI/Testing/WSI/")
    parser.add_argument('--h5_dir', type=str, default="/home/auwqh/dataset/HER2/patches_features_20x/clip_ViTB32/h5_files/")
    parser.add_argument('--save_dir', type=str, default="/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2/clip_numhead(4)/clip_group_ensemble/heatmap_stn")
    parser.add_argument('--head', type=int, default=0, help="要可视化那个head的注意力热图，-1表示采用平均的结果")
    parser.add_argument('--ext', type=str, default=".tiff")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args =get_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

    model = MPL_MIL(**config["model"]).to(args.device)
    model.eval()

    for fold in range(args.start, args.fold):
        checkpoint_path = os.path.join(args.checkpoint_dir, f"model_fold{fold}.pth")
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

        csv_path = os.path.join(args.csv_dir, f"fold_{fold}.csv")
        df = pd.read_csv(csv_path)
        wsi_names = sorted(df["test"].dropna().tolist())

        if args.description == "instance":
            cls_names = config["model"]["instance_descriptions"]
        elif args.description == "stain":
            cls_names = config["model"]["stain_descriptions"]
        else:
            raise ValueError("description must be instance or stain")
        creat_heatmap(model, cls_names, wsi_names, args.wsi_dir, args.h5_dir, args.save_dir, args.device, args.description, ext=args.ext)