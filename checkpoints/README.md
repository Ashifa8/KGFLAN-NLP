# Checkpoints

Pre-trained MLP Relation Scorer weights are **not included** in this repository due to file size.

## Training your own checkpoints

Run the training script to generate checkpoints from scratch:

```bash
# FactKG scorer  (~5 minutes on CPU, ~2 minutes on GPU)
python train.py --dataset factkg

# MetaQA scorer  (~5 minutes on CPU, ~2 minutes on GPU)
python train.py --dataset metaqa
```

Checkpoints will be saved to:
- `checkpoints/factkg_scorer.pt`
- `checkpoints/metaqa_scorer.pt`

## Checkpoint format

Each `.pt` file contains:

```python
{
    "model_state_dict": OrderedDict,   # PyTorch state dict
    "input_dim":        384,            # Embedding dimension
    "hidden_dims":      [256, 64],      # Hidden layer sizes
    "dropout":          0.3,            # Dropout probability
    "dataset":          "factkg",       # Which dataset scorer was trained on
    "best_val_loss":    float,          # Best validation loss achieved
}
```

## Loading a checkpoint

```python
from src.model import load_scorer

scorer = load_scorer("checkpoints/factkg_scorer.pt", device="cpu")
```

## Expected performance

| Scorer   | Val Accuracy | Training Pairs | Training Time (T4 GPU) |
|----------|-------------|----------------|------------------------|
| FactKG   | 85.47%      | ~11,000        | < 3 minutes            |
| MetaQA   | 95.63%      | 11,436         | < 3 minutes            |

## Hardware requirements

- **CPU**: Trains in < 5 minutes with pre-computed embeddings
- **GPU**: Any GPU with > 2 GB VRAM is sufficient (model is ~100K params)
- **RAM**: ~2 GB for embedding 11K pairs with `all-MiniLM-L6-v2`
