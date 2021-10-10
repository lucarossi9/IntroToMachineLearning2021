from typing import Optional

import torch
import tqdm
from torch.utils.data import DataLoader

from metrics import BinaryAccuracyMetric


class Evaluator:
    """Model evaluator
    Args:
        model: model to be evaluated
        device: device on which to evaluate model
        loader: dataloader on which to evaluate model
        checkpoint_path: path to model checkpoint
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        loader: DataLoader,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        # Device
        self.device = device

        # Data
        self.loader = loader

        # Model
        self.model = model

        if checkpoint_path:
            self._load_from_checkpoint(checkpoint_path)

        # Metrics
        self.acc_metric = BinaryAccuracyMetric(threshold=0.5)

    def evaluate(self) -> float:
        """Evaluates the model
        Returns:
            (float) accuracy (on a 0 to 1 scale)
        """

        # Progress bar
        pbar = tqdm.tqdm(total=len(self.loader), leave=False)
        pbar.set_description("Evaluating... ")

        # Set to eval
        self.model.eval()

        # Loop
        for data, target in self.loader:
            with torch.no_grad():
                # To device
                data, target = data.to(self.device), target.to(self.device)

                # Forward
                out = self.model(data)

                self.acc_metric.update(out.sigmoid(), target)

                # Update progress bar
                pbar.update()

        pbar.close()

        accuracy = self.acc_metric.compute()
        print(f"Accuracy: {accuracy:.4f}\n")

        return accuracy

    def predict(self, threshold: int = 0.5) -> torch.Tensor:
        """Returns predictions for the given data
        Assumes the output of the model are the logits and applies sigmoid to the output

        Args:
            threshold: prediction threshold
        Returns:
            (torch.Tensor) Model predictions for the given data of shape [N,],
                where N is the number of samples in the data
        """

        # Progress bar
        pbar = tqdm.tqdm(total=len(self.loader), leave=False)
        pbar.set_description("Predicting... ")

        # Set to eval
        self.model.eval()

        preds = []

        # Loop
        for data, _ in self.loader:
            with torch.no_grad():
                # To device
                data = data.to(self.device)

                # Forward
                out = self.model(data)
                pred = torch.where(out.sigmoid() > threshold, 1, 0)

                preds.append(pred)
                # Update progress bar
                pbar.update()

        pbar.close()

        preds = torch.cat(preds).reshape(-1)

        return preds

    def _load_from_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        print(f"Checkpoint loaded: {checkpoint_path}")
