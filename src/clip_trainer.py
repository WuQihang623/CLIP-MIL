import torch
import numpy as np
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


def train_one_epoch(model, loader, loss_fn, optimizer, scheduler, epoch, device, logger):
    model.train()
    total_loss = {}
    total_err = 0.

    for step, (inputs, targets, clinical_scores) in enumerate(loader):
        inputs = inputs.to(device).type(torch.float32)
        targets = targets.to(device).type(torch.long)
        clinical_scores = clinical_scores.to(device).type(torch.long)

        predictions = model(inputs)
        loss_dict = loss_fn(
            preds_cls=predictions["cls_logits"],
            targets_cls=targets,
            preds_bag=predictions.get("bag_prompt_logits"),
            targets_bag=clinical_scores)
        loss = loss_dict["loss"]
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        for k, v in loss_dict.items():
            if k not in total_loss:
                total_loss[k] = 0.
            total_loss[k] += v.item() * targets.shape[0]

        err, fps, fns = calculate_metrics(predictions["cls_logits"].detach(), targets.cpu())
        total_err += err

    for k, v in total_loss.items():
        total_loss[k] = total_loss[k] / len(loader.dataset)
    total_err = total_err / len(loader.dataset)
    string = f"Epoch {epoch} | Train | Acc: {1 - total_err:.4f} "
    for k, v in total_loss.items():
        string += f"| {k}: {v:.3f}"
    logger.info(string)
    return total_loss["loss"], 1 - total_err


@torch.no_grad()
def test_one_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = {}
    preds_list = []
    targets_list = []

    for step, (inputs, targets, clinical_scores) in enumerate(loader):
        inputs = inputs.to(device).type(torch.float32)
        targets = targets.to(device).type(torch.long)
        clinical_scores = clinical_scores.to(device).type(torch.long)

        predictions = model(inputs)
        preds = predictions["cls_logits"].detach().argmax(dim=1)
        loss_dict = loss_fn(
            preds_cls=predictions["cls_logits"],
            targets_cls=targets,
            preds_bag=predictions.get("bag_prompt_logits"),
            targets_bag=clinical_scores)

        for k, v in loss_dict.items():
            if k not in total_loss:
                total_loss[k] = 0.
            total_loss[k] += v.item() * targets.shape[0]

        preds_list.extend(preds.cpu().numpy().tolist())
        targets_list.extend(targets.cpu().numpy().tolist())

    for k, v in total_loss.items():
        total_loss[k] = total_loss[k] / len(loader.dataset)

    targets_list = np.array(targets_list, dtype=np.int32)
    preds_list = np.array(preds_list, dtype=np.int32)
    acc = accuracy_score(targets_list, preds_list)
    f1 = f1_score(targets_list, preds_list, average="macro")
    precision = precision_score(targets_list, preds_list, average="macro")
    recall = recall_score(targets_list, preds_list, average="macro")

    return total_loss["loss"], acc, f1, precision, recall, preds_list, targets_list


