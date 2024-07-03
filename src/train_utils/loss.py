import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class KLLoss(nn.Module):
    def __init__(self, variance=1):
        super(KLLoss, self).__init__()
        self.variance = torch.tensor(variance)

    def forward(self, preds, targets):
        # preds: (batch_size, num_classes)
        # targets: (batch_size)
        N, K = preds.size()
        preds = torch.softmax(preds, dim=1)  # 确保preds是一个有效的概率分布
        targets = targets.unsqueeze(1).expand(N, K)  # 扩展targets以广播至preds的形状

        # 计算以targets为中心的高斯分布
        gaussian_dist = torch.exp(-(torch.arange(K, device=targets.device).unsqueeze(0)) ** 2 / (2 * self.variance))
        gaussian_dist = torch.clamp(gaussian_dist, min=1e-8)
        gaussian_dist = gaussian_dist / torch.sqrt(2 * torch.tensor(torch.pi * self.variance, device=targets.device))
        gaussian_dist = gaussian_dist / torch.sum(gaussian_dist, dim=1, keepdim=True)  # 归一化以确保是概率分布

        loss = F.kl_div(torch.log(preds), torch.log(gaussian_dist), reduction='sum', log_target=True)
        return loss


class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, preds, targets):
        return F.cross_entropy(preds, targets)


class MIL_Loss(nn.Module):
    def __init__(self):
        super(MIL_Loss, self).__init__()
        self.loss_ce = CrossEntropy()
        self.loss_kl = KLLoss()

    def forward(self, preds_cls, preds_bag, targets_cls, targets_bag):
        loss_ce = self.loss_ce(preds_cls, targets_cls)
        loss_kl = self.loss_kl(preds_bag, targets_bag)
        loss = loss_ce + loss_kl
        loss_dict = {
            'loss_ce': loss_ce,
            'loss_kl': loss_kl,
            'loss': loss
        }
        return loss_dict


if __name__ == '__main__':
    kl_loss = KLLoss()
    pred = torch.randn((1, 10))
    target = torch.tensor([1])
    loss = kl_loss(pred, target)
    print(loss)
