import json
import os
import yaml
import time
import yaml
import random
import argparse

import torch
import numpy as np
from torch.optim import Adam, SGD
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.models.TOP import TOP, PromptLearner
from src.train_utils.logger import get_logger
from src.datasets.dataset_CLIP_MIL import dataset_CLIP_MIL
from src.train_utils.loss import MIL_Loss
from src.train_utils.utils import save_yaml_ordered
from src.TOP_trainer import train_one_epoch, test_one_epoch
from src.train_utils.plot_confusion import plot_confusion_matrix


def main(config):
    device = config["device"]

    def warmup_lr_lambda(current_step, warmup_steps, base_lr, max_lr):
        if current_step < warmup_steps:
            return base_lr + (max_lr - base_lr) * (current_step / warmup_steps)
        else:
            return max_lr

    start = time.time()
    prediction_list = []  # 测试集的模型预测结果
    target_list = []  # 测试集的真实标签
    logger = get_logger(config["log_name"], os.path.join(config["save_dir"], "result.log"))
    for k in range(config["fold"]):
        logger.info(f"-----------Fold {k}-----------")
        csv_path = os.path.join(config["fold_dir"], f"fold_{k}.csv")
        train_set = dataset_CLIP_MIL(csv_path, config["feat_dir"], "train")
        val_set = dataset_CLIP_MIL(csv_path, config["feat_dir"], "val")
        test_set = dataset_CLIP_MIL(csv_path, config["feat_dir"], "test")

        class_sample_count = [len([label for label in train_set.label if label == cls]) for cls in set(train_set.label)]
        print("class_sample_count", class_sample_count)
        class_weights = [1 / count for count in class_sample_count]
        weights = [class_weights[label] for label in train_set.label]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

        train_loader = DataLoader(
            train_set,
            batch_size=config["batch_size"],
            sampler=sampler,
            num_workers=config["num_workers"],
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=True,
        )

        print("train loader", len(train_loader))
        print("val loader", len(val_loader))
        print("test loader", len(test_loader))

        model_config = config["model"]
        bag_prompt_learner = PromptLearner(n_ctx=model_config["bagLevel_n_ctx"],
                                           ctx_init=model_config["bagPrompt_ctx_init"],
                                           all_ctx_trainable=model_config["all_ctx_trainable"],
                                           csc=model_config["csc"],
                                           classnames=["HER2 0", "HER2 1+", "HER2 2+", "HER2 3+"],
                                           clip_model='ViT-B/32', p_drop_out=model_config["p_bag_drop_out"])
        prompts_pathology_template_withDescription = [
            model_config["pathology_templates_t"].format(tissue_type).replace(".", ", which is {}".format(tissue_description)) for
            tissue_type, tissue_description in model_config["knowledge_from_chatGPT"].items()]
        instancePrompt_ctx_init = [i + '* * * * * * * * * *' for i in prompts_pathology_template_withDescription]
        instance_prompt_learner = PromptLearner(n_ctx=model_config["instanceLevel_n_ctx"],
                                                ctx_init=instancePrompt_ctx_init,
                                                all_ctx_trainable=model_config["all_ctx_trainable"],
                                                csc=model_config["csc"],
                                                classnames=["Prototype {}".format(i) for i in range(len(instancePrompt_ctx_init))],
                                                clip_model='ViT-B/32', p_drop_out=model_config["p_drop_out"])

        model = TOP(bag_prompt_learner, instance_prompt_learner, clip_model="ViT-B/32", pooling_strategy=model_config["pooling_strategy"]).to(device)

        for param in model.text_encoder.parameters():
            param.requires_grad = False

        pg = [p for p in model.parameters() if p.requires_grad]
        optimizer = Adam(pg, lr=config['lr'])

        sched_cfg = config["scheduler"]
        lr_lambda = lambda step: warmup_lr_lambda(
            step, sched_cfg["warmup_steps"], sched_cfg["base_lr"], sched_cfg["max_lr"]
        ) / sched_cfg["max_lr"]

        warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        cosine_anneal_scheduler = CosineAnnealingLR(
            optimizer, T_max=sched_cfg["T_max"], eta_min=sched_cfg["eta_min"]
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_anneal_scheduler],
            milestones=[sched_cfg["warmup_steps"]]  # 热身结束的epoch作为切换点
        )

        checkpoint_path = os.path.join(config["save_dir"], f'model_fold{k}.pth')
        best_f1 = 0
        best_loss = 100
        patience = config["early_stop"]
        now_patience = 0

        weight_lossA = config["weight_lossA"]
        for epoch in range(config["n_epochs"]):
            train_loss, train_err = train_one_epoch(
                model, train_loader, optimizer, scheduler, epoch, device, weight_lossA, logger
            )
            val_loss, val_acc, val_f1, val_precision, val_recall, val_preds_list, val_targets_list = test_one_epoch(
                model, val_loader, weight_lossA, device
            )
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(
                f"Epoch: {epoch} | train_loss: {train_loss:.3f} | train_err: {train_err:.3f} | val_loss: {val_loss:.3f} | val_acc: {val_acc:.3f} |\nEpoch: {epoch} | val_f1: {val_f1:.3f} | val_precision: {val_precision:.3f} | val_recall: {val_recall:.3f} | lr: {current_lr:.9f}")

            if best_f1 < val_f1 or (best_f1 == val_f1 and best_loss > val_loss):
                now_patience = 0
                best_f1 = val_f1
                best_loss = val_loss if best_loss > val_loss else best_loss
                torch.save(model.state_dict(), checkpoint_path)
                logger.info("saving model")
            else:
                now_patience += 1
            logger.info('\r')

            if (now_patience >= patience):
                break

        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'), strict=True)
        test_loss, test_acc, test_f1, test_precision, test_recall, test_preds_list, test_targets_list = test_one_epoch(
            model, test_loader, weight_lossA, device
        )
        prediction_list.extend(test_preds_list.tolist())
        target_list.extend(test_targets_list.tolist())

    with open(os.path.join(config["save_dir"], "preds.json"), 'w') as f:
        f.write(json.dumps({"pred": prediction_list, "target": target_list}))
        f.close()

    acc = accuracy_score(target_list, prediction_list)
    F1 = f1_score(target_list, prediction_list, average='macro')
    precision = precision_score(target_list, prediction_list, average='macro')
    recall = recall_score(target_list, prediction_list, average='macro')

    end = time.time()
    logger.info(f"time: {end - start:.3f}")

    logger.info(f"acc: {acc:.3f}, F1: {F1:.3f}, precision: {precision:.3f}, recall: {recall:.3f}")

    conf_matrix = confusion_matrix(target_list, prediction_list)
    plot_confusion_matrix(conf_matrix, ["HER2 0", "HER2 1+", "HER2 2+", "HER2 3+"], save_path=os.path.join(args.save_dir, "confusion.png"))


def get_args():
    parser = argparse.ArgumentParser(description="Train CLIP MIL model")
    parser.add_argument('--config', type=str, default='/home/auwqh/code/CLIP-MIL/examples/config/clip_mil_vit_b32_no_descrip.yaml', help='Path to config file')
    parser.add_argument('--fold', type=int, default=5)
    parser.add_argument('--save_dir', type=str, default='/home/auwqh/code/CLIP-MIL/save_weights/PDL1', help='Path to save results')
    parser.add_argument('--log_name', type=str, default='TOP', help='Name of log file')
    parser.add_argument('--feat_dir', type=str, default="/home/auwqh/dataset/PDL1/meta_data/Testing/patch/clip_ViTB32/pt_files/")
    parser.add_argument('--fold_dir', type=str, default="/home/auwqh/code/CLIP-MIL/data/PDL1_fold")

    args = parser.parse_args()
    return args

def seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.cuda.manual_seed_all(s)

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    args = get_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

    seed(config['seed'])

    args.save_dir = os.path.join(args.save_dir, args.log_name)
    os.makedirs(args.save_dir, exist_ok=True)
    config.update(vars(args))
    save_yaml_ordered(config, filename=os.path.join(args.save_dir, "config.yaml"))

    main(config)

