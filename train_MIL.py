import json
import os
import yaml
import time
import random
import argparse

import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.models.ABMIL import ABMIL
from src.models.CLAM import CLAM_MB
from src.models.TransMIL import TransMIL
from src.train_utils.logger import get_logger
from src.datasets.dataset_MIL import dataset_MIL
from src.custom_trainer import train_one_epoch, test_one_epoch

def main(args):
    acc_list = []
    auc_list = []
    precision_list = []
    recall_list = []
    prediction_list = [] # 测试集的模型预测结果
    target_list = [] # 测试集的真实标签
    start = time.time()
    logger = get_logger(args.log_name, os.path.join(args.save_dir, "result.log"))
    for k in range(5):
        logger.info(f"-----------第{k}fold开始训练------------")
        csv_path = os.path.join(args.fold_dir, f"fold_{k}.csv")
        train_set = dataset_MIL(csv_path=csv_path, feat_dir=args.feat_dir, mode="train")
        val_set = dataset_MIL(csv_path=csv_path, feat_dir=args.feat_dir, mode="val")
        test_set = dataset_MIL(csv_path=csv_path, feat_dir=args.feat_dir, mode="test")

        class_sample_count = [len([label for label in train_set.label if label == cls]) for cls in set(train_set.label)]
        print("class_sample_count", class_sample_count)
        class_weights = [1 / count for count in class_sample_count]
        weights = [class_weights[label] for label in train_set.label]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.workers, sampler=sampler)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

        print("train loader", len(train_loader))
        print("val loader", len(val_loader))
        print("test loader", len(test_loader))

        if args.model_name == "TransMIL":
            model = TransMIL(n_classes=args.n_classes, feature_dim=args.feature_dim).to(args.device)
        elif args.model_name == "ABMIL":
            model = ABMIL(args.n_classes, feature_dim=args.feature_dim).to(args.device)
        elif args.model_name == "CLAM_MB":
            model = CLAM_MB(n_classes=args.n_classes, feature_dim=args.feature_dim).to(args.device)
        else:
            raise NotImplementedError

        criterion = nn.CrossEntropyLoss().to(args.device)

        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

        checkpoint_path = os.path.join(args.save_dir, f'model_fold{k}.pth')
        best_acc = 0
        patience = 10
        now_patience = 0
        for epoch in range(args.n_epochs):
            train_loss, train_err = train_one_epoch(model, train_loader, criterion, optimizer, epoch, args.device, logger, args)
            val_loss, val_acc, val_auc_list, val_precision, val_recall, val_preds_list, val_targets_list = test_one_epoch(model, val_loader, criterion, args.device)
            logger.info(f"Epoch: {epoch}, train_loss: {train_loss:.3f}, train_err: {train_err:.3f}, val_loss: {val_loss:.3f}, val_acc: {val_acc:.3f}, val_auc: {np.mean(val_auc_list):.3f}, val_precision: {val_precision:.3f}, val_recall: {val_recall:.3f}")
            scheduler.step()

            if best_acc < val_acc:
                now_patience = 0
                best_acc = val_acc
                torch.save(model.state_dict(), checkpoint_path)
                logger.info("saving model")
            else:
                now_patience += 1
            logger.info('\r')

            if (now_patience >= patience):
                break

        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'), strict=True)
        test_loss, test_acc, test_auc_list, test_precision, test_recall, test_preds_list, test_targets_list = test_one_epoch(
            model, val_loader, criterion, args.device)
        acc_list.append(test_acc)
        auc_list.append(np.mean(test_auc_list))
        precision_list.append(test_precision)
        recall_list.append(test_recall)
        prediction_list.extend(test_preds_list.reshape(-1).tolist())
        target_list.extend(test_targets_list.reshape(-1).tolist())


    with open(os.path.join(args.save_dir, "preds.json"), "w") as f:
        f.write(json.dumps({"pred": prediction_list, "target": target_list}))
        f.close()

    acc = accuracy_score(target_list, prediction_list)
    F1 = f1_score(target_list, prediction_list, average='macro')
    precision = precision_score(target_list, prediction_list, average='macro')
    recall = recall_score(target_list, prediction_list, average='macro')

    end = time.time()
    logger.info(f"time: {end-start:.3f}")

    logger.info(f"acc: {acc:.3f}, F1: {F1:.3f}, precision: {precision:.3f}, recall: {recall:.3f}")


def get_args():
    parser = argparse.ArgumentParser(description='Train MIL model')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--model_name', type=str, default='CLAM_SB', choices=['CLAM_SB', 'CLAM_MB', 'ABMIL', 'TransMIL'])
    parser.add_argument('--n_classes', type=int, default=3)
    parser.add_argument('--feature_dim', type=int, default=1024, help='patch image feature dimension')
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--feat_dir', type=str, default="")
    parser.add_argument('--fold_dir', type=str, default="")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--save_dir', type=str, default='/home/auwqh/code/CLIP-MIL/save_weight',
                        help='path to trained model checkpoint')
    parser.add_argument("--log_name", type=str, default="CLAM")
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.1)
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
    warnings.filterwarnings("ignore", category=UserWarning)

    args = get_args()
    seed(args.seed)

    args.save_dir = os.path.join(args.save_dir, args.log_name)
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "config.yaml"), "w") as f:
        save_cfg = {}
        save_cfg.update(vars(args))
        yaml.dump(save_cfg, f, default_flow_style=False)
        f.close()

    main(args)
