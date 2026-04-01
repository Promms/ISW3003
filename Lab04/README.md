# lab04

Practice project: implement deep learning models (starting with ResNet-50) and measure parameter counts and FLOPs.

## Structure
- `main.py`: build models and compute parameters/FLOPs
- `models/resnet50.py`: ResNet-50 implementation (complete)
- `models/*.py` (except `resnet50.py`): skeletons for students to complete
- `utils/param_utils.py`: parameter counter
- `utils/compute_utils.py`: FLOPs counter (Conv2d/Linear only, bias ignored)

## Run
```bash
python main.py
```

Options:
- `--model` 
- `--num-classes` (default 1000)
- `--batch-size` (default 1)
- `--height` (default 224)
- `--width` (default 224)
- `--device` (default cpu)

Example:
```bash
python main.py --model resnet50 --height 224 --width 224 --device cpu
```

## FLOPs Recording Guide
The tables below record FLOPs by input size for each model. Above each table, write `model name: parameter count`.

resnet50: 25,557,032
| 224x224 | 192x192 | 160x160 | 128x128 |
| --- | --- | --- | --- |
| 4,089,184,256 | 3,004,841,984 | 2,087,321,600 | 1,336,623,104 |

MobileNet v2: [TODO]
| 224x224 | 192x192 | 160x160 | 128x128 |
| --- | --- | --- | --- |
| [TODO] | [TODO] | [TODO] | [TODO] |

ResNet50 + FPN-style: [TODO]
| 640x640 | 800x800 | 1024x1024 | 800x1333 |
| --- | --- | --- | --- |
| [TODO]| [TODO] | [TODO] | [TODO] |

MonoDepth2-like U-net: [TODO]
| 192x640 | 256x832 | 320x1024 | 384x1280 |
| --- | --- | --- | --- |
| [TODO] | [TODO] | [TODO] | [TODO] |

DeepLabv3-style model: [TODO]
| 321x321 | 513x513 | 641x641 | 769x769 |
| --- | --- | --- | --- |
| [TODO] | [TODO] | [TODO] | [TODO] |
