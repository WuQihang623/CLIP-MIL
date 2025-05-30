import os
import json
import sys

import numpy as np
import tqdm
import argparse

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from src import clip
from src.datasets.custom_dataset import CustomDataset

def main(args):
    device = torch.device(args.device)
    model, preprocess = clip.load("ViT-B/32", device=device)

    model.to(device)
    with open(args.template_path, 'r') as f:
        template = json.load(f)
        f.close()

    with open(args.data, "r") as f:
        data = json.load(f)
        f.close()

    text_prompt = clip.tokenize(template).to(device)

    dataset = CustomDataset(data["image_list"], data["label_list"], preprocess)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    dataloader = tqdm.tqdm(dataloader, file=sys.stdout)

    pred_list = []
    true_list = []
    with torch.no_grad():
        for batch_data in dataloader:
            inputs = batch_data["input"].to(device)
            label = batch_data["label"].to(device)
            logits_per_image, logits_per_text = model(inputs, text_prompt)
            best_matches = torch.argmax(logits_per_image, dim=1)

            # 添加到 pred_list 和 true_list 中
            pred_list.extend(best_matches.cpu().numpy())
            true_list.extend(label.cpu().numpy())

    # 计算准确率和 F1 分数
    pred_list = np.array(pred_list)
    true_list = np.array(true_list)
    print("Predictions:", pred_list.shape)
    print("Labels:", true_list.shape)

    accuracy = accuracy_score(true_list, pred_list)
    f1 = f1_score(true_list, pred_list)

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"{args.save_name}.txt")
    with open(save_path, 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\nF1 Score: {f1:.4f}")
        f.write("\nTemplates:\n")
        for text in template:
            f.write(f"{text}\n")
        for pred, label in zip(pred_list, true_list):
            f.write(f"{pred}, {label}\n")
        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CLIP Model Testing Script")
    parser.add_argument("--model", type=str, default="ViT-B/32")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument("--data", type=str, default="/home/auwqh/code/CLIP-MIL/data/tumor_dataset.json", help="Root directory of the dataset")
    parser.add_argument("--template_path", type=str, default="/home/auwqh/code/CLIP-MIL/src/templates/is_tumor_v2.json", help="Template for text descriptions")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for DataLoader")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument("--save_dir", type=str, default="/home/auwqh/code/CLIP-MIL/src/performers", help="Directory to save the model")
    parser.add_argument('--save_name', type=str, default="tumor_classification_v2", help="Name of the saved file")
    args = parser.parse_args()
    main(args)
