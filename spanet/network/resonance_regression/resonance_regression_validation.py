"""
Validation logic for Resonance Regression model.

Implements validation_step with regression metrics.
"""
from typing import Dict

import numpy as np

from spanet.options import Options
from spanet.dataset.types import Batch

from spanet.network.resonance_regression.resonance_regression_network import ResonanceRegressionNetwork


class ResonanceRegressionValidation(ResonanceRegressionNetwork):
    """Validation mixin for resonance regression model.

    Implements validation_step with regression metrics computation.
    """

    def __init__(self, options: Options, torch_script: bool = False):
        super().__init__(options, torch_script)

    def validation_step(self, batch: Batch, batch_idx: int) -> Dict[str, float]:
        """Perform a single validation step.

        Parameters
        ----------
        batch : Batch
            Input batch containing sources and regression targets.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        Dict[str, float]
            Dictionary of validation metrics.
        """
        sources, num_jets, assignment_targets, regression_targets, classification_targets = batch

        # Get predictions
        predictions = self.predict(sources)

        # Convert targets to numpy
        regression_targets_np = {
            key: value.detach().cpu().numpy()
            for key, value in regression_targets.items()
        }

        metrics = {}

        # Compute metrics for each regression target
        for key in predictions:
            pred = predictions[key]
            target = regression_targets_np[key]

            # Handle NaN targets
            valid_mask = ~np.isnan(target)
            if valid_mask.sum() == 0:
                continue

            pred_valid = pred[valid_mask]
            target_valid = target[valid_mask]

            # Compute errors
            delta = pred_valid - target_valid

            # Mean Absolute Error
            mae = np.abs(delta).mean()
            self.log(f"REGRESSION/{key}_mae", mae, sync_dist=True)
            metrics[f"{key}_mae"] = mae

            # Mean Squared Error
            mse = (delta ** 2).mean()
            self.log(f"REGRESSION/{key}_mse", mse, sync_dist=True)
            metrics[f"{key}_mse"] = mse

            # Root Mean Squared Error
            rmse = np.sqrt(mse)
            self.log(f"REGRESSION/{key}_rmse", rmse, sync_dist=True)
            metrics[f"{key}_rmse"] = rmse

            # Mean Absolute Percent Error (avoid division by zero)
            nonzero_mask = target_valid != 0
            if nonzero_mask.sum() > 0:
                mape = np.abs(delta[nonzero_mask] / target_valid[nonzero_mask]).mean()
                self.log(f"REGRESSION/{key}_mape", mape, sync_dist=True)
                metrics[f"{key}_mape"] = mape

            # Log histograms for detailed analysis
            if hasattr(self.logger, 'experiment') and hasattr(self.logger.experiment, 'add_histogram'):
                self.logger.experiment.add_histogram(
                    f"REGRESSION/{key}_residuals",
                    delta,
                    self.global_step
                )

                if nonzero_mask.sum() > 0:
                    percent_error = delta[nonzero_mask] / target_valid[nonzero_mask]
                    self.logger.experiment.add_histogram(
                        f"REGRESSION/{key}_percent_error",
                        percent_error,
                        self.global_step
                    )

        # Use MAE as validation metric (lower is better)
        if metrics:
            first_key = list(predictions.keys())[0]
            self.log("validation_mae", metrics[f"{first_key}_mae"], sync_dist=True)
            self.log("validation_mse", metrics[f"{first_key}_mse"], sync_dist=True)
            if f"{first_key}_mape" in metrics:
                self.log("validation_mape", metrics[f"{first_key}_mape"], sync_dist=True)

        return metrics

    def test_step(self, batch: Batch, batch_idx: int) -> Dict[str, float]:
        """Perform a single test step (same as validation)."""
        return self.validation_step(batch, batch_idx)
