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
from sklearn.manifold import TSNE
from src.models.MPL_MIL import MPL_MIL
from src.models.ABMIL import ABMIL
from src.models.CLAM import CLAM_MB
from src.models.TransMIL import TransMIL
from src.models.TOP import PromptLearner, TOP
from src.models.ViLa_MIL import ViLa_MIL_Model

@torch.no_grad()
def infer_wsi_feature(model, wsi_names, h5_dir, device):
    wsi_features = []
    for wsi_name in wsi_names:
        print(wsi_name)
        h5_path = os.path.join(h5_dir, wsi_name + '.h5')

        h5_file = h5py.File(h5_path, 'r')
        features = np.array(h5_file["features"], dtype=np.float32)
        features = torch.from_numpy(features).to(device)

        if not isinstance(model, ViLa_MIL_Model):
            output = model(features.unsqueeze(0))
        else:
            output = model.high_slide_features(features.unsqueeze(0))
        wsi_feature = output["features"].squeeze(0).cpu().numpy()
        print(wsi_feature.shape)
        wsi_features.append(wsi_feature)
    return wsi_features


def plot_tsne(feature, labels, save_path):
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)

    # 转换为NumPy数组
    feature = np.array(feature)
    labels = np.array(labels, dtype=np.uint8)

    print(feature.shape)
    print(labels.shape)

    # t-SNE降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, labels.shape[0] / 1.2))
    tsne_results = tsne.fit_transform(feature)

    print(tsne_results.shape)

    # 绘制t-SNE图
    plt.figure(figsize=(6, 6))

    unique_labels = np.unique(labels)
    for label in unique_labels:
        indices = labels == label

        if label == 0:
            label = "HER2 0"
        else:
            label = "HER2 {}+".format(int(label))

        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=label, alpha=0.5, s=40)

    plt.legend(fontsize=16, loc="lower right")
    plt.xticks([])  # 隐藏x轴的数值
    plt.yticks([])  # 隐藏y轴的数值
    plt.show()
    plt.savefig(save_path)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="TransMIL", choices=["ABMIL", "CLAM", "TransMIL", "CLIPMIL", "TOP", "ViLa_MIL"])
    parser.add_argument('--config', type=str, default="/home/auwqh/code/CLIP-MIL/examples/config_PD_L1/clip_instancepooling_ensemble.yaml")
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument("--feat_dim", type=int, default=512)
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--checkpoint_dir', type=str, default="/home/auwqh/code/CLIP-MIL/save_weights/PDL1/TransMIL_clip_ViTB32_weights")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--csv_dir', type=str, default="/home/auwqh/code/CLIP-MIL/data/PDL1_fold")
    parser.add_argument('--h5_dir', type=str, default="/home/auwqh/dataset/PDL1/meta_data/Testing/patch/clip_ViTB32/h5_files/")
    parser.add_argument('--save_path', type=str, default="/home/auwqh/code/CLIP-MIL/save_weights/PDL1/TransMIL_clip_ViTB32_weights/tsne.png")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args =get_args()

    if args.model == "CLIPMIL":
        with open(args.config, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            f.close()
        model = MPL_MIL(**config["model"]).to(args.device)
    elif args.model == "ABMIL":
        model = ABMIL(num_classes=args.num_classes, feature_dim=args.feat_dim).to(args.device)
    elif args.model == "CLAM":
        model = CLAM_MB(n_classes=args.num_classes, feature_dim=args.feat_dim).to(args.device)
    elif args.model == "TransMIL":
        model = TransMIL(n_classes=args.num_classes, feature_dim=args.feat_dim).to(args.device)
    elif args.model == "TOP":
        with open(args.config, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            f.close()
        model_config = config["model"]
        bag_prompt_learner = PromptLearner(n_ctx=model_config["bagLevel_n_ctx"],
                                           ctx_init=model_config["bagPrompt_ctx_init"],
                                           all_ctx_trainable=model_config["all_ctx_trainable"],
                                           csc=model_config["csc"],
                                           classnames=["HER2 0", "HER2 1+", "HER2 2+", "HER2 3+"],
                                           clip_model='ViT-B/32', p_drop_out=model_config["p_bag_drop_out"])
        prompts_pathology_template_withDescription = [
            model_config["pathology_templates_t"].format(tissue_type).replace(".", ", which is {}".format(
                tissue_description)) for
            tissue_type, tissue_description in model_config["knowledge_from_chatGPT"].items()]
        instancePrompt_ctx_init = [i + '* * * * * * * * * *' for i in prompts_pathology_template_withDescription]
        instance_prompt_learner = PromptLearner(n_ctx=model_config["instanceLevel_n_ctx"],
                                                ctx_init=instancePrompt_ctx_init,
                                                all_ctx_trainable=model_config["all_ctx_trainable"],
                                                csc=model_config["csc"],
                                                classnames=["Prototype {}".format(i) for i in
                                                            range(len(instancePrompt_ctx_init))],
                                                clip_model='ViT-B/32', p_drop_out=model_config["p_drop_out"])

        model = TOP(bag_prompt_learner, instance_prompt_learner, clip_model="ViT-B/32", pooling_strategy=model_config["pooling_strategy"]).to(args.device)
    elif args.model == "ViLa_MIL":
        import ml_collections
        args.text_prompt = np.array(pd.read_csv("/home/auwqh/code/CLIP-MIL/save_weights/ZJY_HER2/ViLa_MIL/HER2_two_scale_text_prompt.csv", header=None)).squeeze()

        config = ml_collections.ConfigDict()
        config.input_size = 512
        config.hidden_size = 192
        config.text_prompt = args.text_prompt
        config.prototype_number = 16
        model_dict = {'config': config, 'num_classes': 4}
        model = ViLa_MIL_Model(**model_dict).to(args.device)
    else:
        raise ValueError("Invalid model name")

    model.eval()

    features = []
    labels = []
    for fold in range(args.fold):
        checkpoint_path = os.path.join(args.checkpoint_dir, f"model_fold{fold}.pth")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(args.checkpoint_dir, f"model_fold{fold}.pt")
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

        csv_path = os.path.join(args.csv_dir, f"fold_{fold}.csv")
        df = pd.read_csv(csv_path)

        wsi_names = df["test"].dropna().tolist()

        test_labels = df['test_label'].dropna().tolist()

        wsi_features = infer_wsi_feature(model, wsi_names, args.h5_dir, args.device)
        features.extend(wsi_features)
        labels.extend(test_labels)

    plot_tsne(features, labels, args.save_path)
