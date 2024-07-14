import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, preds, targets):
        return F.cross_entropy(preds, targets)


class MIL_Loss(nn.Module):
    def __init__(self, lambda_bag=1.0):
        super(MIL_Loss, self).__init__()
        self.loss_ce = CrossEntropy()
        self.lambda_bag = lambda_bag

    def forward(self, preds_cls, targets_cls, **kwargs):
        loss_dict = {}
        loss_ce = self.loss_ce(preds_cls, targets_cls) * self.lambda_bag
        loss_dict["loss"] = loss_ce
        return loss_dict


