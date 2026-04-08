# Lab 06 — Tiny ImageNet Classification with Truncated ResNet-50

Image classification on [Tiny ImageNet](https://huggingface.co/datasets/zh-plus/tiny-imagenet) (200 classes, 64×64 images) using a truncated ResNet-50 backbone. Training is iteration-based with exponential LR decay and experiment tracking via Weights & Biases.

## Model Architecture

`TinyImageNetClassifier` (`models/resnet.py`):

- **Backbone**: ResNet-50 pretrained on ImageNet, truncated after `layer2`
  - `conv1 + bn1 + relu + maxpool` → `[B, 64, 16, 16]`
  - `layer1` → `[B, 256, 16, 16]`
  - `layer2` → `[B, 512, 8, 8]`
- **Head**: Global average pooling → Dropout(0.5) → Linear(512, 200)

## Project Structure

```
lab06/
├── config/
│   └── tiny_imagenet.yaml   # All hyperparameters and W&B settings
├── data/
│   └── tiny_imagenet.py     # DataLoader factory (HuggingFace datasets)
├── models/
│   └── resnet.py            # TruncatedResNet50 + TinyImageNetClassifier
├── utils/
│   ├── metrics.py           # accuracy(), AverageMeter
│   └── param_utils.py       # Parameter count logging
├── train.py                 # Training script
└── eval.py                  # Evaluation script
```

## Setup

```bash
# Install dependencies with uv
uv sync
```

Requirements: Python ≥ 3.11, PyTorch, torchvision, Hugging Face `datasets`, W&B, PyYAML.

## Training

```bash
uv run python train.py
```

Configuration is read from `config/tiny_imagenet.yaml`. Key hyperparameters:

| Parameter | Value |
|---|---|
| Batch size | 128 |
| Total iterations | 20,000 |
| Learning rate | 1e-3 (Adam) |
| Weight decay | 1e-4 |
| LR decay | ×0.9997 per iteration |
| Dropout | 0.5 |

Training logs `train/loss`, `train/top1_acc`, and `lr` to W&B every 50 iterations, and runs validation every 500 iterations. The best checkpoint (highest val top-1) is saved to `best_checkpoint.pth`.

## Evaluation

```bash
uv run python eval.py --checkpoint best_checkpoint.pth
# Override batch size if needed:
uv run python eval.py --checkpoint best_checkpoint.pth --batch-size 256
```

Prints validation loss and top-1 accuracy.

## Experiment Tracking

Experiments are logged to W&B under project `osai-lab06`. Set your W&B team and run name in `config/tiny_imagenet.yaml` under the `wandb:` key before training.
