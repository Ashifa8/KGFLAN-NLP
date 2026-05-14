"""
src/model.py
============
MLPRelationScorer — Core novelty of KG-Flan.

Replaces ChatGPT-based relation selection with a lightweight discriminative
classifier trained on (claim/question, candidate_relation) pairs.

Architecture
------------
  Input : embed([Claim / Question] [SEP] [Relation Name])  →  384-dim vector
  Linear(384 → 256) → ReLU → Dropout(0.3)
  Linear(256 →  64) → ReLU → Dropout(0.3)
  Linear( 64 →   1) → Sigmoid  →  Relevance Score ∈ [0, 1]

~100K parameters. Trains in < 5 minutes on CPU.
Inference: microseconds per pair (vs. ~1s per LLM API call).
"""

import torch
import torch.nn as nn
from typing import List


class MLPRelationScorer(nn.Module):
    """
    Binary relevance scorer for (claim, relation) pairs.

    Parameters
    ----------
    input_dim   : int   — dimension of the sentence embedding (default 384)
    hidden_dims : list  — sizes of hidden layers (default [256, 64])
    dropout     : float — dropout probability applied after each hidden layer
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dims: List[int] = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 64]

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)
        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        """Xavier uniform init for all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch_size, input_dim)
            Pre-computed sentence embeddings.

        Returns
        -------
        Tensor of shape (batch_size, 1) — relevance score in [0, 1].
        """
        return self.network(x)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def score(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Convenience method: returns scores detached from graph."""
        self.eval()
        return self.forward(embeddings).squeeze(-1)

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────────────
# Utility: load a scorer checkpoint
# ──────────────────────────────────────────────────────────────────────

def load_scorer(ckpt_path: str, device: str = "cpu") -> MLPRelationScorer:
    """
    Load a saved MLPRelationScorer from a .pt checkpoint.

    The checkpoint is expected to contain:
        {
            "model_state_dict": ...,
            "input_dim": ...,
            "hidden_dims": ...,
            "dropout": ...,
        }
    """
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = MLPRelationScorer(
        input_dim=checkpoint.get("input_dim", 384),
        hidden_dims=checkpoint.get("hidden_dims", [256, 64]),
        dropout=checkpoint.get("dropout", 0.3),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"[MLPRelationScorer] Loaded from '{ckpt_path}' — {model.count_parameters():,} parameters")
    return model


# ──────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = MLPRelationScorer()
    print(f"Parameters : {model.count_parameters():,}")

    dummy = torch.randn(8, 384)
    scores = model.score(dummy)
    print(f"Input shape  : {dummy.shape}")
    print(f"Output shape : {scores.shape}")
    print(f"Score range  : [{scores.min():.4f}, {scores.max():.4f}]")
