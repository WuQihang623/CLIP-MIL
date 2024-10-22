# Load the corresponding folded model and test it on the corresponding test set
# test the ensemble model's performance
import os
import yaml
import json
import argparse

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from src.models.ABMIL import ABMIL
from src.models.CLAM import CLAM_MB
from src.models.TransMIL import TransMIL
from src.models.MPL_MIL import MPL_MIL
from src.train_utils.loss import MIL_Loss
from src.models.TOP import PromptLearner, TOP
from src.datasets.dataset_CLIP_MIL import dataset_CLIP_MIL
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def append_to_json_file(file_path, new_data):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            try:
                existing_data = json.load(file)
            except json.JSONDecodeError:
                existing_data = {}
    else:
        existing_data = {}

    if not isinstance(existing_data, dict):
        existing_data = {}

    existing_data.update(new_data)

    # 将合并后的数据写回JSON文件
    with open(file_path, 'w') as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=2)

def plot_confusion_matrix(conf_matrix, labels, save_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar()

    # 添加标签
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)

    # 在图像中添加数字
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(save_path)

def test_one_fold(model, fold, loss_fn, device, args):
    csv_path = os.path.join(args.csv_dir, f"fold_{fold}.csv")
    test_set = dataset_CLIP_MIL(csv_path, args.pt_dir, "test")
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    if args.model != "TOP":
        from src.MPL_trainer import test_one_epoch
        test_loss, test_acc, test_f1, test_precision, test_recall, test_preds_list, test_targets_list = test_one_epoch(
            model, test_loader, loss_fn, device=device
        )
    else:
        from src.TOP_trainer import test_one_epoch
        test_loss, test_acc, test_f1, test_precision, test_recall, test_preds_list, test_targets_list = test_one_epoch(
            model, test_loader, 0, device
        )
    return test_preds_list, test_targets_list

def main(args):
    device = torch.device(args.device)

    log_path = os.path.join(args.checkpoint_dir, f"{args.save_name}.json")
    if os.path.exists(log_path):
        os.remove(log_path)

    if args.model == "CLIPMIL":
        with open(args.config, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            f.close()
        model = MPL_MIL(**config["model"])
    elif args.model == "ABMIL":
        model = ABMIL(num_classes=args.num_classes, feature_dim=args.feat_dim)
    elif args.model == "CLAM":
        model = CLAM_MB(n_classes=args.num_classes, feature_dim=args.feat_dim)
    elif args.model == "TransMIL":
        model = TransMIL(n_classes=args.num_classes, feature_dim=args.feat_dim)
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

        model = TOP(bag_prompt_learner, instance_prompt_learner, clip_model="ViT-B/32", pooling_strategy=model_config["pooling_strategy"])
    else:
        raise ValueError(f"model name error!")
    model.to(device)
    model.eval()

    loss_fn = MIL_Loss()

    acc_list = []
    prec_list = []
    recall_list = []
    f1_list = []

    for fold in range(args.train_fold):
        checkpoint_path = os.path.join(args.checkpoint_dir, f"model_fold{fold}.pth")
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

        if args.test_fold == -1:
            preds_list, target_list = test_one_fold(model, fold, loss_fn, device, args)
        else:
            preds_list = np.array([])
            target_list = np.array([])
            for data_fold in range(args.test_fold):
                preds_fold, target_fold = test_one_fold(model, data_fold, loss_fn, device, args)
                preds_list = np.append(preds_list, preds_fold)
                target_list = np.append(target_list, target_fold)

        acc = accuracy_score(target_list, preds_list)
        f1 = f1_score(target_list, preds_list, average="macro")
        prec = precision_score(target_list, preds_list, average="macro")
        recall = recall_score(target_list, preds_list, average="macro")
        acc_list.append(acc)
        prec_list.append(prec)
        recall_list.append(recall)
        f1_list.append(f1)
        append_to_json_file(log_path, {f"fold_{fold}": {"predtions": preds_list.tolist(), "targets": target_list.tolist()}})

    acc_mean = np.mean(acc_list)
    acc_std = np.std(acc_list)
    prec_mean = np.mean(prec_list)
    prec_std = np.std(prec_list)
    recall_mean = np.mean(recall_list)
    recall_std = np.std(recall_list)
    f1_mean = np.mean(f1_list)
    f1_std = np.std(f1_list)

    append_to_json_file(log_path, {"acc_mean": f"{acc_mean:.3f}", "acc_std": f"{acc_std:.3f}",
                                   "prec_mean": f"{prec_mean:.3f}", "prec_std": f"{prec_std:.3f}",
                                   "recall_mean": f"{recall_mean:.3f}", "recall_std": f"{recall_std:.3f}",
                                   "f1_mean": f"{f1_mean:.3f}", "f1_std": f"{f1_std:.3f}"})

    with open(log_path, 'r') as f:
        data = json.load(f)
        f.close()
    preds_list = []
    target_list = []
    for fold in range(args.train_fold):
        preds_list.extend(data[f"fold_{fold}"]["predtions"])
        target_list.extend(data[f"fold_{fold}"]["targets"])
    conf_matrix = confusion_matrix(target_list, preds_list)
    plot_confusion_matrix(conf_matrix, ["HER2 0", "HER2 1+", "HER2 2+", "HER2 3+"], os.path.join(args.checkpoint_dir, f"confusion_matrix{args.save_name}.png"))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="TransMIL", choices=["ABMIL", "CLAM", "TransMIL", "CLIPMIL", "TOP"])
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--feat_dim", type=int, default=512)
    parser.add_argument("--checkpoint_dir", type=str, default="")
    parser.add_argument("--train_fold", type=int, default=5)
    parser.add_argument("--test_fold", type=int, default=-1, help="When the model is tested on the same dataset, it defaults to 1, and when tested on different datasets, it is set to the fold of that dataset.")
    parser.add_argument("--save_name", type=str, default="")
    parser.add_argument("--csv_dir", type=str, default="")
    parser.add_argument("--pt_dir", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    main(args)
