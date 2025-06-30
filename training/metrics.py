
import torch
from torchmetrics.classification import Accuracy, F1Score

class MetricCollection(torch.nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()
        self.accuracy = Accuracy(task="multiclass", num_classes=num_labels)
        self.f1 = F1Score(task="multiclass", num_classes=num_labels)

    def forward(self, preds, targets):
        # For single-label classification, we need to get the predicted class
        # by taking the argmax of the predictions
        pred_classes = torch.argmax(preds, dim=1)

        # For targets, we need to get the class index with value 1
        # If there are multiple 1s (in test data), we take the first one
        # This will be handled differently during inference
        target_classes = torch.argmax(targets, dim=1)

        return {
            "Accuracy": self.accuracy(pred_classes, target_classes),
            "F1": self.f1(pred_classes, target_classes)
        }
