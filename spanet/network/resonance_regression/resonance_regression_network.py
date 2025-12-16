"""
Network architecture for Resonance Regression model.

Architecture: Jets -> Embedding -> Transformer Encoder -> Regression Decoder
"""
from typing import Dict, NamedTuple

import numpy as np
import torch
from torch import Tensor

from spanet.options import Options
from spanet.dataset.types import Tuple, Source

from spanet.network.layers.vector_encoder import JetEncoder
from spanet.network.layers.embedding import MultiInputVectorEmbedding
from spanet.network.layers.regression_decoder import RegressionDecoder

from spanet.network.resonance_regression.resonance_regression_base import ResonanceRegressionBase


class RegressionOutputs(NamedTuple):
    """Output structure for resonance regression model."""
    event_vector: Tensor                # [B, D] - encoded event representation
    regressions: Dict[str, Tensor]      # {name: [B]} - regression predictions


class ResonanceRegressionNetwork(ResonanceRegressionBase):
    """Resonance regression network: embedding -> transformer -> regressor.

    This model takes jet inputs, embeds them, processes through a transformer encoder,
    and outputs regression predictions. No jet assignment is performed.
    """

    def __init__(self, options: Options, torch_script: bool = False):
        super().__init__(options)

        compile_module = torch.jit.script if torch_script else lambda x: x

        self.hidden_dim = options.hidden_dim

        # Embedding layer: converts jet features to hidden dimension
        self.embedding = compile_module(MultiInputVectorEmbedding(
            options,
            self.training_dataset
        ))

        # Transformer encoder: processes embedded jets
        self.encoder = compile_module(JetEncoder(options))

        # Regression decoder: predicts regression targets from event vector
        self.regression_decoder = compile_module(RegressionDecoder(
            options,
            self.training_dataset
        ))

    @property
    def event_info(self):
        return self.training_dataset.event_info

    def forward(self, sources: Tuple[Source, ...]) -> RegressionOutputs:
        """Forward pass through the network.

        Parameters
        ----------
        sources : Tuple[Source, ...]
            Input sources, each containing (data, mask) tensors.

        Returns
        -------
        RegressionOutputs
            Named tuple containing event_vector and regression predictions.
        """
        # Embed all input sources into the same latent space
        embeddings, padding_masks, sequence_masks, global_masks = self.embedding(sources)

        # Extract features using transformer encoder
        # hidden: [T, B, D] - per-jet representations
        # event_vector: [B, D] - global event representation
        hidden, event_vector = self.encoder(embeddings, padding_masks, sequence_masks)

        # Build vector dictionary for regression decoder
        # The regression decoder uses keys like "EVENT/gen_mass" and looks up "EVENT" vector
        encoded_vectors = {"EVENT": event_vector}

        # Predict regression targets
        regressions = self.regression_decoder(encoded_vectors)

        return RegressionOutputs(
            event_vector=event_vector,
            regressions=regressions
        )

    def predict(self, sources: Tuple[Source, ...]) -> Dict[str, np.ndarray]:
        """Run inference and return numpy arrays.

        Parameters
        ----------
        sources : Tuple[Source, ...]
            Input sources.

        Returns
        -------
        Dict[str, np.ndarray]
            Regression predictions as numpy arrays.
        """
        with torch.no_grad():
            outputs = self.forward(sources)
            return {
                key: value.cpu().numpy()
                for key, value in outputs.regressions.items()
            }
