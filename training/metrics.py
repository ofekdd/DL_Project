
import torch
from torchmetrics.classification import MultilabelAveragePrecision, MultilabelF1Score

class MetricCollection(torch.nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()
        self.map = MultilabelAveragePrecision(num_labels)
        self.f1  = MultilabelF1Score(num_labels)

    def forward(self, preds, targets):
        return {
            "mAP": self.map(preds, targets),
            "F1":  self.f1(preds, targets)
        }
