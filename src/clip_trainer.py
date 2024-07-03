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


def train_one_epoch(model, loader, loss_fn, optimizer, epoch, device, logger):
    model.train()
    total_loss = 0.
    total_loss_ce = 0.
    total_loss_kl = 0.
    total_err = 0.

    for step, (inputs, targets, clinical_scores) in enumerate(loader):
        inputs = inputs.to(device).type(torch.float32)
        targets = targets.to(device).type(torch.long)
        clinical_scores = clinical_scores.to(device).type(torch.long)

        predictions = model(inputs)
        loss_dict = loss_fn(
            preds_cls=predictions["cls_logits"],
            preds_bag=predictions["bag_prompt_logits"],
            targets_cls=targets,
            targets_bag=clinical_scores)
        loss = loss_dict["loss"]
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item() * targets.shape[0]
        total_loss_ce += loss_dict["loss_ce"].item() * targets.shape[0]
        total_loss_kl += loss_dict["loss_kl"].item() * targets.shape[0]

        err, fps, fns = calculate_metrics(predictions["cls_logits"].detach(), targets.cpu())
        total_err += err

    total_loss = total_loss / len(loader.dataset)
    total_loss_ce = total_loss_ce / len(loader.dataset)
    total_loss_kl = total_loss_kl / len(loader.dataset)
    total_err = total_err / len(loader.dataset)
    logger.info(f"Epoch {epoch} | Train | Loss: {total_loss:.4f} | Loss_ce: {total_loss_ce:.4f} | Loss_kl: {total_loss_kl:.4f} | Acc: {1 - total_err:.4f}")
    return total_loss, total_err

@torch.no_grad()
def test_one_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.
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
            preds_bag=predictions["bag_prompt_logits"],
            targets_cls=targets,
            targets_bag=clinical_scores)

        total_loss += loss_dict["loss"].item() * targets.shape[0]

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


