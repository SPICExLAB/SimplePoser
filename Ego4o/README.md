# Ego4o

IMU-based pose estimation using a Part-Aware VQ-VAE and IMU Transformer Encoder, based on the [Ego4o paper](https://arxiv.org/abs/2312.16174).

## Pipeline

| Stage | What | Input | Output |
|-------|------|-------|--------|
| 1 | Part-Aware VQ-VAE | HumanML3D 263-dim motion | 6 body-part codebooks |
| 2 | IMU Transformer Encoder | IMU (B, T, 6, 9) | VQ-VAE code logits (B, 6, Tt, 512) |
| 3 | Inference + optional TTO | IMU | SMPL joint rotations / positions |

## Training

### Stage 1: Part-Aware VQ-VAE

Trains the VQ-VAE to encode 263-dim HumanML3D motion into 6 body-part codebooks.

```bash
python -m Ego4o.train --config configs/ego4o_vqvae.yaml
python -m Ego4o.train --config configs/ego4o_vqvae.yaml --wandb  # with logging
```

### Stage 2: IMU Transformer Encoder

Trains the encoder to predict VQ-VAE codes from IMU input. Requires a trained Stage 1 checkpoint.

```bash
python -m Ego4o.train_stage2 --config configs/ego4o_encoder.yaml
python -m Ego4o.train_stage2 --config configs/ego4o_encoder.yaml --wandb
```

## Inference & Evaluation

### Predict (MPJPE only)

```bash
python -m Ego4o.predict --config configs/ego4o_predict.yaml
python -m Ego4o.predict --config configs/ego4o_predict.yaml --tto   # with Test-Time Optimization
python -m Ego4o.predict --config configs/ego4o_predict.yaml --save  # save results to disk
```

### Evaluate (full metrics, comparable to DynaIP/MobilePoser)

Extracts SMPL rotation matrices from the 263-dim output and evaluates using `FullMotionEvaluator` — same metrics as the other models.

```bash
python -m Ego4o.evaluate --config configs/ego4o_predict.yaml
python -m Ego4o.evaluate --config configs/ego4o_predict.yaml --tto
```

## Configs

| Config | Purpose |
|--------|---------|
| `configs/ego4o_vqvae.yaml` | Stage 1 VQ-VAE training |
| `configs/ego4o_encoder.yaml` | Stage 2 encoder training |
| `configs/ego4o_predict.yaml` | Inference & evaluation (checkpoints, TTO settings) |
