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

def creat_heatmap(text_embedding, cls_names, wsi_names, wsi_dir, h5_dir, save_dir, device, patch_size=512):
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


        features = features / features.norm(dim=1, keepdim=True)
        text_embedding = text_embedding / text_embedding.norm(dim=1, keepdim=True)

        logits_per_image = 100 * features @ text_embedding.t()
        attn_raw = logits_per_image.softmax(dim=1).detach()

        attn = torch.zeros_like(attn_raw)
        max_values, _ = torch.max(attn_raw, dim=1)
        attn[attn_raw == max_values.unsqueeze(1)] = 1


        for cls_idx, cls_name in enumerate(cls_names):
            heatmap = np.zeros(dimension, dtype=np.uint8)
            ps = int(patch_size / heatmap_downsample)
            for i, prob in enumerate(attn[:, cls_idx]):
                coord = (coords[i] / heatmap_downsample).astype("int32")
                heatmap[coord[1]:coord[1]+ps, coord[0]:coord[0]+ps] = int(prob * 255)
            heatmap = (cmap(heatmap) * 255)[:, :, :3].astype(np.uint8)

            data["properties"]["heatmap_info"]["heatmap_info"].append(cls_name)
            data["heatmap_info"][cls_name] = heatmap
        save_path = os.path.join(save_dir, wsi_name + '.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
            f.close()

        print(f"finish {wsi_name}")



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="/home/auwqh/code/CLIP-MIL/examples/config_HER2/clip_instancepooling.yaml")
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--csv_dir', type=str, default="/home/auwqh/code/CLIP-MIL/data/HER2_fold")
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--wsi_dir', type=str, default="/home/auwqh/dataset/HER2/WSI/Testing/WSI/")
    parser.add_argument('--h5_dir', type=str, default="/home/auwqh/dataset/HER2/patch/clip_ViTB32/h5_files/")
    parser.add_argument('--save_dir', type=str, default="/home/auwqh/code/CLIP-MIL/heatmap/clip/clip_stain_argmax")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args =get_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

    model, preprocess = clip.load("ViT-B/32", device=args.device)
    model.to(args.device)

    template = config["model"]["template"]
    # instance_descriptions = config["model"]["instance_descriptions"]
    instance_descriptions = config["model"]["stain_descriptions"]
    embeddings = []
    for cls_name, cls_descriptions in instance_descriptions.items():
        prompts = []
        for description in cls_descriptions:
            prompts.append(f"{template} {cls_name}, which is {description}")

        prompts = clip.tokenize(prompts).to(args.device)
        embedding = model.encode_text(prompts).detach().to(args.device)
        embedding = torch.mean(embedding, dim=0, keepdim=True)
        embeddings.append(embedding)
    embeddings = torch.cat(embeddings, dim=0)


    csv_path = os.path.join(args.csv_dir, f"fold_{args.fold}.csv")
    df = pd.read_csv(csv_path)
    wsi_names = sorted(df["test"].dropna().tolist())

    creat_heatmap(embeddings, list(instance_descriptions.keys()), wsi_names, args.wsi_dir, args.h5_dir, args.save_dir, args.device)