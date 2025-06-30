
import torch
from torchmetrics.classification import Accuracy

class MetricCollection(torch.nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()
        # Using multiclass metrics since we're now doing single-label classification with softmax
        self.accuracy = Accuracy(task="multiclass", num_classes=num_labels)

    def forward(self, preds, targets):
        # For softmax output, convert predictions to class indices (argmax)
        pred_classes = torch.argmax(preds, dim=1)
        target_classes = torch.argmax(targets, dim=1) if targets.dim() > 1 else targets

        return {
            "Accuracy": self.accuracy(pred_classes, target_classes)
        }
