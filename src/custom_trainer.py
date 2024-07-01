import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

def calculate_metrics(predictions, targets):
    _, pred = predictions.topk(1, 1, True, True)
    pred = pred.squeeze().cpu().numpy()
    real = targets.numpy()
    neq = pred != real
    err = float(neq.sum())
    fps = float(np.logical_and(pred == 1, neq).sum())
    fns = float(np.logical_and(pred == 0, neq).sum())
    return err, fps, fns

def train_one_epoch(model, loader, critetion, optimizer, epoch, device, logger, args):
    model.train()
    running_loss = 0.
    running_err = 0.
    
    for step, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device).type(torch.float32)
        targets = targets.to(device).type(torch.long)
        
        predictions = model(inputs)
        logits = predictions["logits"]
        loss = critetion(logits, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * targets.shape[0]
        err, fps, fns = calculate_metrics(logits.detach(), targets.cpu())
        running_err += err

    running_loss = running_loss / len(loader.dataset)
    running_err = running_err / len(loader.dataset)
    logger.info('Training - Epoch: [{}/{}]\tLoss: {:.3f}\tAcc: {:.3f}\t'.format(epoch + 1, args.n_epochs, running_loss, 1-running_err))
    return running_loss, running_err

@torch.no_grad()
def test_one_epoch(model, loader, critetion, device):
    model.eval()
    running_loss = 0.
    probs_list = []
    preds_list = []
    targets_list = []
    
    for step, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device).type(torch.float32)
        targets = targets.to(device).type(torch.long)
        
        preds = model(inputs)
        logits = preds["logits"]
        Y_hat = preds["Y_hat"]
        Y_prob = preds["Y_prob"]
        
        loss = critetion(logits, targets)
        running_loss += loss.item()
        
        preds_list.extend(Y_hat.cpu().numpy().tolist())
        targets_list.extend(targets.cpu().numpy().tolist())
        probs_list.extend(Y_prob.cpu().numpy().tolist())

    running_loss /= (len(loader))

    auc_list = []
    probs_list = np.array(probs_list, dtype=np.float32)
    targets_list = np.array(targets_list, dtype=np.int32)
    preds_list = np.array(preds_list, dtype=np.int32)
    acc = accuracy_score(targets_list, preds_list)
    for idx in range(probs_list.shape[1]):
        auc_list.append(roc_auc_score((targets_list==idx).astype(int), probs_list[:, idx]))

    precision = precision_score(targets_list, preds_list, average="macro")
    recall = recall_score(targets_list, preds_list, average="macro")

    return running_loss, acc, auc_list, precision, recall, preds_list, targets_list
        
        