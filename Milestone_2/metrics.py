import torch


class LossMetric:
    """Keeps track of the loss over an epoch"""

    def __init__(self) -> None:
        self.running_loss = 0
        self.count = 0

    def update(self, loss: float, batch_size: int) -> None:
        self.running_loss += loss * batch_size
        self.count += batch_size

    def compute(self) -> float:
        return self.running_loss / self.count

    def reset(self) -> None:
        self.running_loss = 0
        self.count = 0


class BinaryAccuracyMetric:
    """Keeps track of the accuracy for a binary classification task"""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.correct = 0
        self.total = 0

    def update(self, out: torch.Tensor, target: torch.Tensor) -> None:
        out, target = out.reshape(-1), target.reshape(-1)
        pred = torch.where(out > self.threshold, 1, 0)

        self.correct += torch.sum(pred == target)
        self.total += target.shape[0]

    def compute(self) -> float:
        return self.correct / self.total

    def reset(self) -> None:
        self.correct = 0
        self.total = 0
