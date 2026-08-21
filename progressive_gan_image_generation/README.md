# Progressive GAN Image Generation

GAN-based FFHQ image generation project.

## Folder Structure

```text
progressive_gan_image_generation/
  README.md
  pyproject.toml
  report.pdf
  assets/
    v2_512_random_samples.png
  checkpoints/
    model.pth
  src/
    config/
      v2_pipeline.yaml
      v2_*.yaml
    project02/
      train.py
      train_wrapper.py
      generate.py
      export.py
      config.py
      checkpoint.py
      logger.py
      resume.py
      loss.py
      data/
      eval/
      models/
        layers.py
        generator.py
        discriminator.py
      utils/
```

## Installation

```bash
pip install -e .
```

## Result

![V2 512 stabilize random-noise samples](assets/v2_512_random_samples.png)

| Metric | Result |
| --- | ---: |
| Submitted FID | 19.313 |

The final model used the V2 512x512 stabilize-stage checkpoint. The project report is included as `report.pdf`.

## Training

Edit the stage configs under `src/config/` so the data paths point to local image directories, then run the progressive wrapper:

```bash
python -m project02.train_wrapper --pipeline-config src/config/v2_pipeline.yaml
```

For each stage config, the fields that usually need local edits are:

```yaml
data:
  train_root: <training image directory for this resolution>
  valid_root: <validation image directory for this resolution>
  cache_mode: shm_files   # set to none if /dev/shm is unavailable
  shm_root: <RAM-disk cache path, only used when cache_mode is shm_files>

wandb:
  enabled: true           # set false for local/offline runs
  entity: <your wandb entity>
  project: <your wandb project>
```

The wrapper injects `training.output_dir`, strict `resume`, `init_from`, and transition teacher checkpoint paths from `src/config/v2_pipeline.yaml`, so those fields normally do not need manual edits.

By default, the training entrypoint writes the latest submission-format checkpoint to `checkpoints/model.pth`. Stage runtime configs written by `project02.train_wrapper` are kept as execution records for resume/init handoff.

## Resume

```bash
python -m project02.train --config src/config/v2_512_stabilize.yaml --resume checkpoints/model.pth
```

Strict resume validates the saved progressive stage state against the config before continuing.

## Generate Samples

```bash
python -m project02.generate --config src/config/v2_512_stabilize.yaml --checkpoint checkpoints/model.pth --output sample.png --target-resolution 1024
```

EMA weights are used when present. Add `--no-ema` to use raw generator weights. Use `--target-resolution 512` to inspect native 512 samples without the submission resize. Training does not write periodic sample PNGs; use this command for sample checks.

## FID Evaluation

```bash
python -m project02.eval.fid --real <valid_dir> --fake <fake_dir> --batch-size 32 --device cuda
```

The real and fake image directories, image counts, dimensions, and batch sizes are supplied through config or CLI arguments rather than hard-coded paths. Training can log FID as a scalar metric, but temporary fake image directories are removed after the metric is computed.

## ONNX Export

```bash
python -m project02.export --checkpoint checkpoints/model.pth --output model.onnx
```

The final submission candidate uses a native 512 generator checkpoint. The submitted ONNX and default `project02.generate` output resize that native output to 1024 through a wrapper/output policy, so 1024 output does not imply native 1024 generator training.

The export wrapper loads the checkpoint, resolves the saved inference resolution from the checkpoint config, and exports a dynamic-batch ONNX generator. When exporting older checkpoints without saved config, pass a matching config such as `--config src/config/v2_512_stabilize.yaml`.

The ONNX graph contains the generator only.

## Submitted Checkpoint

The submitted checkpoint path is:

```text
checkpoints/model.pth
```

The checkpoint schema keeps the existing generator, discriminator, EMA, optimizer, config, W&B run id, and progressive stage state.

The checkpoint is kept locally but is not tracked in Git because of its size.
