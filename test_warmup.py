from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Adam
import torch.nn as nn


def warmup_lr_lambda(current_step, warmup_steps, base_lr, max_lr):
    if current_step < warmup_steps:
        return base_lr + (max_lr - base_lr) * (current_step / warmup_steps)
    else:
        return max_lr

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 1)
    def forward(self, x):
        return self.fc(x)

model = Model()
optimizer = Adam(model.parameters(), lr=0.001)
lr_lambda = lambda step: warmup_lr_lambda(step, 5, 1.0e-6, 1.0e-4) / 1.0e-4
scheduler = LambdaLR(optimizer, lr_lambda)

for i in range(10):
    print(optimizer.param_groups[0]['lr'])
    scheduler.step()
