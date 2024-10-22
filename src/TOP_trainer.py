import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_metrics(predictions, targets):
    _, pred = predictions.topk(1, 1, True, True)
    pred = pred.squeeze().cpu().numpy()
    real = targets.numpy()
    neq = pred != real
    err = float(neq.sum())
    fps = float(np.logical_and(pred == 1, neq).sum())
    fns = float(np.logical_and(pred == 0, neq).sum())
    return err, fps, fns

def train_one_epoch(model, loader, optimizer, scheduler, epoch, device, weight_lossA, logger):
    model.train()
    total_loss = 0.
    total_err = 0.

    for step, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device).type(torch.float32)
        targets = targets.to(device).type(torch.long)

        predictions = model(inputs.squeeze(0))
        bag_prediction = predictions["logits"]
        instance_attn_score = predictions["attn"]

        bag_prediction = torch.softmax(bag_prediction, 1)
        loss_D = F.cross_entropy(bag_prediction, targets)
        instance_attn_score_normed = torch.softmax(instance_attn_score, 0)
        loss_A = torch.triu(instance_attn_score_normed.T @ instance_attn_score_normed, diagonal=1).mean()
        loss = loss_D + weight_lossA * loss_A

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        err, fps, fns = calculate_metrics(predictions["logits"].detach(), targets.cpu())
        total_err += err

        total_loss += loss.detach()

    total_loss = total_loss / len(loader.dataset)
    total_err = total_err / len(loader.dataset)

    string = f"Epoch {epoch} | Train | Acc: {1 - total_err:.4f} | Train Loss: {total_loss:.4f}"
    logger.info(string)
    return total_loss, 1 - total_err

@torch.no_grad()
def test_one_epoch(model, loader, weight_lossA, device):
    model.eval()
    total_loss = 0
    preds_list = []
    targets_list = []

    for step, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device).type(torch.float32)
        targets = targets.to(device).type(torch.long)

        predictions = model(inputs)
        preds = predictions["logits"].detach().argmax(dim=1)

        bag_prediction = predictions["logits"]
        instance_attn_score = predictions["attn"]

        bag_prediction = torch.softmax(bag_prediction, 1)
        loss_D = F.cross_entropy(bag_prediction, targets)
        instance_attn_score_normed = torch.softmax(instance_attn_score, 0)
        loss_A = torch.triu(instance_attn_score_normed.T @ instance_attn_score_normed, diagonal=1).mean()
        loss = loss_D + weight_lossA * loss_A

        total_loss += loss.detach()

        preds_list.extend(preds.cpu().numpy().tolist())
        targets_list.extend(targets.cpu().numpy().tolist())

    total_loss = total_loss / len(loader.dataset)

    targets_list = np.array(targets_list, dtype=np.int32)
    preds_list = np.array(preds_list, dtype=np.int32)
    acc = accuracy_score(targets_list, preds_list)
    f1 = f1_score(targets_list, preds_list, average="macro")
    precision = precision_score(targets_list, preds_list, average="macro")
    recall = recall_score(targets_list, preds_list, average="macro")

    return total_loss, acc, f1, precision, recall, preds_list, targets_list

