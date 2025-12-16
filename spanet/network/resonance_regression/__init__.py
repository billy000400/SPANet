"""
Resonance Regression Model

A simplified SPANet variant for pure regression tasks.
Takes jet inputs, embeds them, processes through a transformer,
and outputs regression predictions without jet assignment.

Usage:
    from spanet.network.resonance_regression import ResonanceRegressionModel

    model = ResonanceRegressionModel(options)
"""
from spanet.network.resonance_regression.resonance_regression_training import ResonanceRegressionTraining
from spanet.network.resonance_regression.resonance_regression_validation import ResonanceRegressionValidation


class ResonanceRegressionModel(ResonanceRegressionValidation, ResonanceRegressionTraining):
    """Complete Resonance Regression model combining training and validation.

    Architecture:
        Jets -> MultiInputVectorEmbedding -> JetEncoder (Transformer) -> RegressionDecoder

    Inherits:
        - ResonanceRegressionValidation: validation_step with regression metrics
        - ResonanceRegressionTraining: training_step with regression loss
        - ResonanceRegressionNetwork: forward pass
        - ResonanceRegressionBase: dataset management, optimizer configuration
    """
    pass
