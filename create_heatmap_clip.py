import os
import pickle

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
from src import clip
from torch import nn

def creat_heatmap(text_embedding, wsi_names, wsi_dir, h5_dir, save_dir, device, patch_size=512):
    os.makedirs(save_dir, exist_ok=True)
    cmap = plt.get_cmap("jet")
    for wsi_name in wsi_names:
        h5_path = os.path.join(h5_dir, wsi_name + '.h5')
        wsi_path = os.path.join(wsi_dir, wsi_name + '.tiff')

        h5_file = h5py.File(h5_path, 'r')
        coords = np.array(h5_file["coords"], dtype=np.int64)
        features = np.array(h5_file["features"], dtype=np.float32)
        features = torch.from_numpy(features).to(device)

        fc = nn.Linear(512, 512)
        features = fc(features)

        slide = openslide.open_slide(wsi_path)
        level = slide.get_best_level_for_downsample(64)
        dimension = slide.level_dimensions[level]
        heatmap_downsample = int(slide.level_downsamples[level])

        heatmap = np.zeros(dimension, dtype=np.uint8)

        features = features / features.norm(dim=1, keepdim=True)
        text_embedding = text_embedding / text_embedding.norm(dim=1, keepdim=True)

        logits_per_image = 100 * features @ text_embedding.t()
        attention = logits_per_image.softmax(dim=1).detach().numpy()
        # attention = (attention == 5).astype("uint8")
        attention = attention[:, 5]

        ps = int(patch_size / heatmap_downsample)
        for i, prob in enumerate(attention):
            coord = (coords[i] / heatmap_downsample).astype("int32")
            heatmap[coord[1]:coord[1]+ps, coord[0]:coord[0]+ps] = int(prob * 255)
        heatmap = (cmap(heatmap) * 255)[:, :, :3].astype(np.uint8)

        data = {
            "properties": {
                "heatmap_info": {"heatmap_info": ["肿瘤区域"], "downsample": heatmap_downsample}
            },
            "heatmap_info": {"肿瘤区域": heatmap}
        }
        save_path = os.path.join(save_dir, wsi_name + '.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
            f.close()

        print(f"finish {wsi_name}")



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="/home/auwqh/code/CLIP-MIL/save_weights/PDL1/clip_description_group_firstoken/config.yaml")
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--csv_dir', type=str, default="/home/auwqh/code/CLIP-MIL/data/PDL1_fold")
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--wsi_dir', type=str, default="/home/auwqh/dataset/PDL1/meta_data/Testing/WSI/")
    parser.add_argument('--h5_dir', type=str, default="/home/auwqh/dataset/PDL1/meta_data/Testing/patch/clip_ViTB32/h5_files/")
    parser.add_argument('--save_dir', type=str, default="/home/auwqh/code/CLIP-MIL/heatmap_clip_softmax")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args =get_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

    model, preprocess = clip.load("ViT-B/32", device=args.device)
    model.to(args.device)

    template = config["model"]["instance_template"]
    classes = config["model"]["instance_texts"]
    prompts = []
    for cls_name, _ in classes.items():
        prompts.append(template + " " + cls_name)
    prompt = clip.tokenize(prompts).to(args.device)

    text_embedding = model.encode_text(prompt).detach().to(args.device)

    csv_path = os.path.join(args.csv_dir, f"fold_{args.fold}.csv")
    df = pd.read_csv(csv_path)
    wsi_names = sorted(df["test"].dropna().tolist())

    creat_heatmap(text_embedding, wsi_names, args.wsi_dir, args.h5_dir, args.save_dir, args.device)