# Project01 Semantic Segmentation

This project trains a CNN-based DeepLabV3+ semantic segmentation model with a
TorchVision MobileNetV3-Large ImageNet-1K backbone.

## Structure

```text
src/
  train.py
  eval.py
  predict.py
  semantic_segmentation.yaml
  data/
  models/
  utils/
  tools/
  scripts/
checkpoints/
  model.pth
submit/
  img/
  pred/
pyproject.toml
README.md
```

## Install

```bash
pip install -r <(python - <<'PY'
import tomllib
deps = tomllib.load(open("pyproject.toml", "rb"))["project"]["dependencies"]
print("\n".join(deps))
PY
)
```

Alternatively, install the packages listed in `pyproject.toml` manually.

## Train

Run commands from the project root.

```bash
python src/train.py --config src/semantic_segmentation.yaml
```

The default config uses VOC 2007/2012 plus optional MS-COCO training masks. MS-COCO
requires `pycocotools`, which is allowed by the project guideline for COCO usage.

## Evaluate

```bash
python src/eval.py \
  --config src/semantic_segmentation.yaml \
  --ckpt checkpoints/model.pth \
  --use_ema
```

## Predict

Place test images in `submit/img`, then run:

```bash
python src/predict.py \
  --ckpt checkpoints/model.pth \
  --img_dir submit/img \
  --pred_dir submit/pred \
  --use_ema
```

The prediction filenames match the input image stems.

## FLOPs / ONNX Structure

Fast local Conv2d-only estimate:

```bash
python src/tools/measure_gflops.py
```

Export a structure-only ONNX model for the grading site:

```bash
python src/models/export_onnx_structure.py --output model_structure.onnx
```

The ONNX input shape is `[1, 3, 480, 640]`. Weight tensor values are stripped after
export, leaving only the graph structure needed for FLOPs measurement.
