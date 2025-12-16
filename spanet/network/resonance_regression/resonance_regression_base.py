"""
Base class for Resonance Regression model.

Inherits dataset management and optimizer configuration from JetReconstructionBase,
but simplifies by removing assignment-specific logic (particle balancing, jet balancing).
"""
from spanet.options import Options
from spanet.network.jet_reconstruction.jet_reconstruction_base import JetReconstructionBase


class ResonanceRegressionBase(JetReconstructionBase):
    """Base class for resonance regression models.

    Reuses dataset loading and optimizer configuration from JetReconstructionBase,
    while removing jet assignment-specific features like particle/jet balancing.
    """

    def __init__(self, options: Options):
        # Skip the parent __init__ and call grandparent (pl.LightningModule) directly,
        # then selectively initialize only what we need
        super(JetReconstructionBase, self).__init__()

        self.save_hyperparameters(options)
        self.options = options

        # Create datasets (reuse parent's dataset creation logic)
        self.training_dataset, self.validation_dataset, self.testing_dataset = self.create_datasets()

        # Disable assignment-specific balancing (not needed for pure regression)
        self.balance_particles = False
        self.balance_jets = False
        self.balance_classifications = False

        # Steps per epoch for learning rate scheduling
        self.steps_per_epoch = len(self.training_dataset) // (self.options.batch_size * max(1, self.options.num_gpu))
        self.total_steps = self.steps_per_epoch * self.options.epochs
        self.warmup_steps = int(round(self.steps_per_epoch * self.options.learning_rate_warmup_epochs))
