import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):

    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, prediction, target):
        diff = prediction - target

        loss = torch.sqrt(
            diff * diff + self.epsilon * self.epsilon
        )

        return loss.mean()


class RestorationLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.charbonnier = CharbonnierLoss()

    def forward(self, prediction, target):
        return self.charbonnier(prediction, target)