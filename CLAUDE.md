# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MTVCrafter is a 4D Motion Tokenization framework for Open-World Human Image Animation. The project consists of two main components:

1. **4DMoT (4D Motion Tokenizer)**: Encodes raw 3D motion sequences into discrete motion tokens
2. **MV-DiT (Motion-aware Video DiT)**: A diffusion transformer that generates videos using motion tokens

The project supports two model variants:

- **MV-DiT-7B**: Based on CogVideoX-T2V-5B
- **MV-DiT-17B**: Based on Wan-2.1-I2V-14B with text control capabilities

## Key Commands

### Environment Setup

```bash
make build
```

### Data Preparation

```bash
# Download NLF pose estimator model
bash fetch_data.sh

# Extract SMPL motion sequences from driving video
python scripts/process_nlf.py "your_video_directory"
```

### Inference Commands

```bash
# Run MV-DiT-7B inference
python scripts/infer_7b.py \
    --ref_image_path "data/ref_images/human.png" \
    --motion_data_path "data/sampled_data.pkl" \
    --output_path "inference_output"

# Run MV-DiT-17B inference with text control
python scripts/infer_17b.py \
    --ref_image_path "data/ref_images/woman.png" \
    --motion_data_path "data/sampled_data.pkl" \
    --output_path "inference_output" \
    --prompt "The woman is dancing on the beach, waves, sunset."
```

### Training

```bash
# Train 4DMoT tokenizer
accelerate launch scripts/train_vqvae.py
```

### Docker Commands

```bash
# Build Docker image
make build

# Run inference in Docker
make run python scripts/infer_7b.py --ref_image_path data/ref_images/human.png --motion_data_path data/sampled_data.pkl --output_path inference_output

# Run Claude-specific Docker command
make claude-run python scripts/infer_17b.py --help
```

## Code Architecture

### Core Module Structure

- **`mtvcrafter/models/`**: Core model implementations
  - `dit/`: Diffusion transformer models and pipelines
    - `mvdit_transformer_7b.py` / `mvdit_transformer_17b.py`: Transformer architectures
    - `pipeline_mtvcrafter_7b.py` / `pipeline_mtvcrafter_17b.py`: Inference pipelines
    - `wan_*.py`: Wan model components (VAE, text encoder, image encoder)
  - `motion4d/`: 4D motion tokenization
    - `vqvae.py`: Vector quantized variational autoencoder for motion
    - `loss.py`: Training losses

- **`scripts/`**: Inference and training scripts
  - `infer_7b.py` / `infer_17b.py`: Main inference entry points
  - `train_vqvae.py`: Training script for 4DMoT
  - `process_nlf.py`: SMPL motion extraction from videos
  - `draw_pose.py`: Pose visualization utilities

### Key Data Flows

1. **Motion Processing**: Raw videos → NLF pose estimation → SMPL parameters → Motion tokens (via 4DMoT)
2. **Image Processing**: Reference images → Image tokens (via VAE encoder)
3. **Video Generation**: Motion tokens + Image tokens → MV-DiT → Generated video frames

### Model Dependencies

- **Pre-trained Models Required**:
  - NLF pose estimator: `data/pretrained_weights/nlf_l_multi_0.3.2.torchscript`
  - CogVideoX-5B checkpoint (for 7B variant)
  - Wan-2-1-14B checkpoint (for 17B variant) in `mtvcrafter/wan2.1/`
  - MTVCrafter checkpoints from Hugging Face

- **Optional Enhancements**:
  - Wan2.1 LoRA weights: `mtvcrafter/wan2.1/Wan2.1_I2V_14B_FusionX_LoRA.safetensors`

## Development Notes

### Package Management

- Uses `uv` for fast dependency management (preferred)
- Traditional `pip` also supported via requirements.txt
- Python 3.11 is required (specified in pyproject.toml)

### GPU Requirements

- CUDA 12.4 support (configured in pyproject.toml)
- Large GPU memory required for 17B model inference
- Docker setup includes `--gpus all` and large shared memory (128GB)

### File Structure Conventions

- Inference scripts in `scripts/` directory
- Core models in `mtvcrafter/models/` with clear separation between DiT and Motion4D components
- Data files (weights, samples, outputs) in `data/` directory
- Reference images stored in `data/ref_images/`

### Motion Data Format

- SMPL motion sequences stored as `.pkl` files
- Contains paired motion-video data for training/inference
- Use `process_nlf.py` to extract from custom videos

## Important Implementation Details

- The project uses PyTorch with CUDA 12.4 support
- Diffusion models use different schedulers: CogVideoX schedulers for 7B, FlowMatchEulerDiscrete for 17B
- Motion tokens are generated using vector quantization on 4D spatio-temporal features
- 4D RoPE (Rotary Position Embedding) is used to maintain spatial-temporal relationships
- Classifier-free guidance is supported through unconditional motion tokens

