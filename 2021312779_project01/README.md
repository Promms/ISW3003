# Project01 Semantic Segmentation

Final model: DeepLabV3+ with a TorchVision EfficientNet-B1 ImageNet-1K classification backbone.

## Quick Start

```bash
pip install -e .
python src/predict.py --img_dir submit/img --pred_dir submit/pred --use_ema
```

The prediction script reads images from `submit/img` and writes PNG masks with matching file names to `submit/pred`.

## Folder Structure

```text
2021312779_project01/
  src/
  checkpoints/
    model.pth
  submit/
    img/
    pred/
  2021312779_project01_report.pdf
  pyproject.toml
  README.md
```

Place input images in `submit/img/`; `submit/pred/` stores one PNG prediction per input image with the same stem.

## Install

```bash
pip install -e .
```

The code uses PyTorch, TorchVision, Pillow, OpenCV, NumPy, PyYAML, pycocotools, and WandB. It does not use HuggingFace, TIMM, PyTorch Lightning, Accelerate, Albumentations, or TorchVision pretrained segmentation weights.

## Train

Edit dataset paths in `src/training_config.yaml`, then run:

```bash
python src/train.py --config src/training_config.yaml
```

The training script supports Pascal VOC 2007/2012 training splits and optional COCO train2017 samples mapped to VOC classes. Validation uses VOC `val` only.
The best validation checkpoint is saved directly to `checkpoints/model.pth`, matching the required submission structure.
The submitted EfficientNet V67 checkpoint was fine-tuned from a previous V7 checkpoint. To reproduce that exact training path, set `training.finetune_from` in `src/training_config.yaml` to the warm-start checkpoint before running training. The final submitted checkpoint itself is provided as `checkpoints/model.pth`.

## Evaluate

```bash
python src/eval.py --config src/training_config.yaml --split val --use_ema
```

Drop `--use_ema` if the submitted checkpoint should use `model_state_dict`.

## Inference

```bash
python src/predict.py --img_dir submit/img --pred_dir submit/pred --use_ema
```

For each input such as `submit/img/0001.jpg`, the script writes `submit/pred/0001.png`. Inference always uses one deterministic 480x640 forward pass. Use `--rename_by_index` only if the input file names are not already in the required format.

## FLOPs

Measure FLOPs at the required `3x480x640` input size:

```bash
python src/measure_flops.py --height 480 --width 640
```

The script uses `torch.profiler` with a dummy tensor and does not download pretrained weights.

## Reproduce Submitted Result

1. Install dependencies with `pip install -e .`.
2. Put the submitted checkpoint at `checkpoints/model.pth`.
3. Put evaluation images in `submit/img`.
4. Run `python src/predict.py --img_dir submit/img --pred_dir submit/pred --use_ema`.
5. Submit the generated PNG files in `submit/pred`.

The submitted report file should be named `2021312779_project01_report.pdf`.
