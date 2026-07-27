"""
model.py
--------
CNN-LSTM with Additive Attention for EEG-based seizure prediction.

Architecture overview
---------------------
Input: (batch, 360, 5, 17)  — batch of sequences
          │
          ▼
   ┌─────────────┐
   │  CNN Block  │  Applied to each frame independently (weight sharing)
   │             │  (batch×360, 5, 17) → (batch×360, 128)
   └─────────────┘
          │
          ▼
   ┌─────────────┐
   │    LSTM     │  Processes frame sequence in chronological order
   │             │  (batch, 360, 128) → (batch, 360, 256)
   └─────────────┘
          │
          ▼
   ┌─────────────┐
   │  Attention  │  Learns which frames matter most
   │             │  (batch, 360, 256) → (batch, 256)
   └─────────────┘
          │
          ▼
   ┌─────────────┐
   │  Classifier │  Maps to seizure probability
   │             │  (batch, 256) → (batch, 1)
   └─────────────┘
          │
          ▼
   P(preictal) — raw logit (pass through sigmoid for probability)

Note: the model outputs raw logits, not probabilities.
BCEWithLogitsLoss in train.py applies sigmoid internally for
numerical stability. Use torch.sigmoid(output) for inference.

Usage
-----
from src.model import SeizurePredictor, ModelConfig

config = ModelConfig()
model  = SeizurePredictor(config)

# Forward pass
x      = torch.randn(16, 360, 5, 17)   # one batch
logits, attn_weights = model(x)
# logits      : (16, 1)   — raw predictions
# attn_weights: (16, 360) — per-frame importance scores for explainability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Model configuration — all hyperparameters in one place
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """
    All model hyperparameters in one dataclass.
    Change values here to experiment — no need to touch architecture code.

    To experiment with a different config:
        config = ModelConfig(cnn_hidden=128, lstm_hidden=512)
        model  = SeizurePredictor(config)
    """
    # Input dimensions — must match dataset.py constants
    n_frames: int = 0   # MUST be set explicitly per experiment via
                        # infer_n_frames(sequences_dir) — no longer
                        # a fixed value since duration varies across
                        # the offset/duration grid search
    n_bands:    int = 5
    n_channels: int = 17

    # CNN block
    cnn_filters_1: int = 32    # filters in first conv layer
    cnn_filters_2: int = 64    # filters in second conv layer
    cnn_output:    int = 128   # size of CNN output vector per frame

    # LSTM
    lstm_hidden:  int = 256    # hidden state size
    lstm_layers:  int = 2      # number of stacked LSTM layers
    lstm_dropout: float = 0.3  # dropout between LSTM layers

    # Attention
    attn_hidden: int = 64      # size of attention MLP hidden layer

    # Classifier head
    clf_hidden:  int = 64      # size of classifier hidden layer
    clf_dropout: float = 0.4   # dropout before final linear layer


# ---------------------------------------------------------------------------
# CNN Block — spatial feature extractor
# ---------------------------------------------------------------------------

class CNNBlock(nn.Module):
    """
    Extracts spatial features from one EEG frame.

    Treats each (5, 17) frame as a 2D input where rows = frequency bands
    and columns = electrode positions. Two convolutional layers learn
    to detect spatial patterns like 'delta elevated in temporal channels'
    or 'theta spreading across hemispheres'.

    Input  : (batch, 1, n_bands, n_channels)  — 1 input channel (grayscale)
    Output : (batch, cnn_output)              — flat feature vector per frame
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Layer 1: detect local band-channel patterns
        # kernel (2,3) — spans 2 adjacent bands and 3 adjacent channels
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=config.cnn_filters_1,
            kernel_size=(2, 3),
            padding=(1, 1),
        )
        self.bn1 = nn.BatchNorm2d(config.cnn_filters_1)

        # Layer 2: detect higher-level combinations of band-channel patterns
        self.conv2 = nn.Conv2d(
            in_channels=config.cnn_filters_1,
            out_channels=config.cnn_filters_2,
            kernel_size=(2, 3),
            padding=(1, 1),
        )
        self.bn2 = nn.BatchNorm2d(config.cnn_filters_2)

        # Global average pooling: collapse spatial dims to one value per filter
        # Produces (batch, cnn_filters_2) regardless of input spatial size
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Project to desired output size
        self.fc = nn.Linear(config.cnn_filters_2, config.cnn_output)
        self.bn3 = nn.BatchNorm1d(config.cnn_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, 1, n_bands, n_channels)

        Returns
        -------
        (batch, cnn_output)
        """
        x = F.relu(self.bn1(self.conv1(x)))   # (batch, 32, n_bands, n_channels)
        x = F.relu(self.bn2(self.conv2(x)))   # (batch, 64, n_bands, n_channels)
        x = self.gap(x)                        # (batch, 64, 1, 1)
        x = x.view(x.size(0), -1)             # (batch, 64)
        x = F.relu(self.bn3(self.fc(x)))       # (batch, 128)
        return x

# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding, added to CNN frame features before
    they enter the LSTM.

    Gives the model explicit information about each frame's position in
    the 30-minute sequence — e.g. frame 50 (early) vs frame 550 (near
    seizure onset) become distinguishable even if their band power
    values happen to look similar.

    Uses the standard fixed sin/cos encoding from Vaswani et al. (2017),
    not learned — this means it generalises to any sequence length
    without needing extra training data.
    """

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()

        # Precompute the encoding matrix once at initialisation
        position = torch.arange(max_len).unsqueeze(1).float()       # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-np.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)   # (1, max_len, d_model) — batch dim for broadcasting

        # register_buffer: saved with the model but not trained
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, n_frames, d_model) — CNN output sequence

        Returns
        -------
        (batch, n_frames, d_model) — x with positional encoding added
        """
        n_frames = x.size(1)
        return x + self.pe[:, :n_frames, :]

# ---------------------------------------------------------------------------
# Additive Attention (Bahdanau-style)
# ---------------------------------------------------------------------------

class AdditiveAttention(nn.Module):
    """
    Additive attention over a sequence of LSTM hidden states.

    Learns a scalar importance score for each frame in the sequence.
    The final context vector is the weighted sum of all LSTM outputs,
    where weights are proportional to each frame's importance score.

    This produces two outputs:
    1. context  — the weighted representation used for classification
    2. weights  — the per-frame importance scores (your explainability output)

    The weights can be plotted as a heatmap over the 30-minute sequence
    to show which time points the model focused on when making its prediction.

    Input  : (batch, n_frames, lstm_hidden)
    Output : context (batch, lstm_hidden), weights (batch, n_frames)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        # Learnable query vector — a trainable parameter representing
        # "what does a pre-ictal pattern look like?"
        self.query = nn.Parameter(
            torch.randn(config.lstm_hidden)
        )

        # Two-layer MLP that scores each frame
        self.score_mlp = nn.Sequential(
            nn.Linear(config.lstm_hidden, config.attn_hidden),
            nn.Tanh(),
            nn.Linear(config.attn_hidden, 1),
        )

    def forward(self, lstm_out: torch.Tensor) -> tuple:
        """
        Parameters
        ----------
        lstm_out : (batch, n_frames, lstm_hidden)

        Returns
        -------
        context : (batch, lstm_hidden)
        weights : (batch, n_frames)  — sums to 1.0 across frames
        """
        # Score each frame: how relevant is this frame to the query?
        # scores shape: (batch, n_frames, 1)
        scores = self.score_mlp(lstm_out)
        scores = scores.squeeze(-1)          # (batch, n_frames)

        # Convert scores to weights that sum to 1 across the time dimension
        weights = F.softmax(scores, dim=1)   # (batch, n_frames)

        # Weighted sum of LSTM outputs
        # weights unsqueezed: (batch, n_frames, 1)
        # lstm_out:           (batch, n_frames, lstm_hidden)
        context = (weights.unsqueeze(-1) * lstm_out).sum(dim=1)  # (batch, lstm_hidden)

        return context, weights


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class SeizurePredictor(nn.Module):
    """
    Full CNN-LSTM-Attention model for seizure prediction.

    Processes a 30-minute EEG sequence and outputs a seizure probability.

    Parameters
    ----------
    config : ModelConfig — all hyperparameters

    Forward input  : (batch, n_frames, n_bands, n_channels)
    Forward outputs: logits (batch, 1), attn_weights (batch, n_frames)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Stage 1: CNN spatial feature extractor
        self.cnn = CNNBlock(config)

        # Stage 1.5: Apply positional encoding
        self.pos_encoding = PositionalEncoding(
            d_model=config.cnn_output,
            max_len=config.n_frames + 10,   # small buffer above expected length
        )

        # Stage 2: LSTM temporal reasoner
        self.lstm = nn.LSTM(
            input_size=config.cnn_output,    # 128 — CNN output size
            hidden_size=config.lstm_hidden,  # 256
            num_layers=config.lstm_layers,   # 2
            dropout=config.lstm_dropout if config.lstm_layers > 1 else 0.0,
            batch_first=True,                # input is (batch, seq, features)
            bidirectional=False,             # unidirectional — clinical authenticity
        )

        # Stage 3: Additive attention
        self.attention = AdditiveAttention(config)

        # Stage 4: Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(config.lstm_hidden, config.clf_hidden),
            nn.ReLU(),
            nn.Dropout(config.clf_dropout),
            nn.Linear(config.clf_hidden, 1),
            # No sigmoid here — BCEWithLogitsLoss applies it internally
        )

        # Weight initialisation
        self._init_weights()

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Parameters
        ----------
        x : (batch, n_frames, n_bands, n_channels)

        Returns
        -------
        logits       : (batch, 1)      — raw prediction (not sigmoid-ed)
        attn_weights : (batch, n_frames) — frame importance scores
        """
        batch_size, n_frames, n_bands, n_channels = x.shape

        # --- Stage 1: CNN (applied to every frame independently) ---
        # Reshape: merge batch and frame dimensions so CNN sees
        # each frame as an independent sample
        x = x.view(batch_size * n_frames, n_bands, n_channels)
        x = x.unsqueeze(1)                  # (batch×frames, 1, n_bands, n_channels)

        cnn_out = self.cnn(x)               # (batch×frames, cnn_output=128)

        # Restore sequence structure
        cnn_out = cnn_out.view(batch_size, n_frames, -1)  # (batch, frames, 128)

        cnn_out = self.pos_encoding(cnn_out)

        # --- Stage 2: LSTM ---
        lstm_out, _ = self.lstm(cnn_out)    # (batch, frames, lstm_hidden=256)

        # --- Stage 3: Attention ---
        context, attn_weights = self.attention(lstm_out)  # (batch, 256), (batch, 360)

        # --- Stage 4: Classify ---
        logits = self.classifier(context)   # (batch, 1)

        return logits, attn_weights

    def predict_proba(self, x: torch.Tensor) -> tuple:
        """
        Convenience method for inference: returns probability instead of logit.

        Parameters
        ----------
        x : (batch, n_frames, n_bands, n_channels)

        Returns
        -------
        proba        : (batch, 1)        — P(preictal) in [0, 1]
        attn_weights : (batch, n_frames) — frame importance scores
        """
        logits, attn_weights = self.forward(x)
        proba = torch.sigmoid(logits)
        return proba, attn_weights

    def _init_weights(self):
        """
        Initialise weights for stable training.

        CNN and linear layers: Kaiming uniform (good for ReLU activations).
        LSTM: orthogonal initialisation for hidden-to-hidden weights,
              which reduces vanishing/exploding gradients in deep LSTMs.
        BatchNorm: standard (weight=1, bias=0).
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LSTM):
                for param_name, param in module.named_parameters():
                    if 'weight_ih' in param_name:
                        nn.init.kaiming_uniform_(param, nonlinearity='relu')
                    elif 'weight_hh' in param_name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in param_name:
                        nn.init.zeros_(param)
                        # Set forget gate bias to 1.0 — helps LSTM remember
                        # longer sequences (important for 30-min EEG windows)
                        n = param.size(0)
                        param.data[n // 4: n // 2].fill_(1.0)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------------
# Model summary utility
# ---------------------------------------------------------------------------

def model_summary(model: SeizurePredictor, config: ModelConfig):
    """Print a readable summary of model architecture and parameter counts."""

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    cnn_params  = sum(p.numel() for p in model.cnn.parameters())
    lstm_params = sum(p.numel() for p in model.lstm.parameters())
    attn_params = sum(p.numel() for p in model.attention.parameters())
    clf_params  = sum(p.numel() for p in model.classifier.parameters())

    print('=' * 55)
    print('SeizurePredictor — Model Summary')
    print('=' * 55)
    print(f'Input shape      : (batch, {config.n_frames}, '
          f'{config.n_bands}, {config.n_channels})')
    print()
    print(f'  CNN Block      : {cnn_params:>8,} params')
    print(f'    conv1        : {config.n_bands}×{config.n_channels} → {config.cnn_filters_1} filters')
    print(f'    conv2        : {config.cnn_filters_1} → {config.cnn_filters_2} filters')
    print(f'    output       : {config.cnn_output}d per frame')
    print()
    print(f'  LSTM           : {lstm_params:>8,} params')
    print(f'    input size   : {config.cnn_output}')
    print(f'    hidden size  : {config.lstm_hidden}')
    print(f'    layers       : {config.lstm_layers}')
    print(f'    direction    : unidirectional')
    print()
    print(f'  Attention      : {attn_params:>8,} params')
    print(f'    type         : additive (Bahdanau)')
    print(f'    query dim    : {config.lstm_hidden}')
    print(f'    output       : {config.n_frames} attention weights')
    print()
    print(f'  Classifier     : {clf_params:>8,} params')
    print(f'    hidden       : {config.clf_hidden}')
    print(f'    output       : 1 logit')
    print()
    print(f'Total parameters : {total_params:,}')
    print(f'Trainable params : {trainable_params:,}')
    print('=' * 55)


# ---------------------------------------------------------------------------
# Self-test — run directly to verify forward pass dimensions
#   python src/model.py
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('=== SeizurePredictor self-test ===\n')

    config = ModelConfig()
    model  = SeizurePredictor(config)
    model.eval()

    model_summary(model, config)

    # Simulate one batch
    batch_size = 4
    x = torch.randn(batch_size, config.n_frames, config.n_bands, config.n_channels)

    print(f'\nInput shape  : {list(x.shape)}')

    with torch.no_grad():
        logits, attn_weights = model(x)

    print(f'Logits shape        : {list(logits.shape)}')           # (4, 1)
    print(f'Attn weights shape  : {list(attn_weights.shape)}')     # (4, 360)
    print(f'Attn weights sum    : {attn_weights.sum(dim=1)}')      # should be ~1.0
    print(f'Logit range         : [{logits.min():.4f}, {logits.max():.4f}]')
    print(f'Proba range         : [{torch.sigmoid(logits).min():.4f}, '
          f'{torch.sigmoid(logits).max():.4f}]')
    print(f'Any NaN             : {torch.isnan(logits).any()}')

    print('\nSelf-test complete. Model is ready for training.')
