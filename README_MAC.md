# Running nanochat on Apple Silicon (Mac) 

This guide shows how to run nanochat on your Mac with full Apple Silicon optimization using MLX.

---

## Quick Start

### 1. Install Dependencies

```bash
# Install MLX support
uv sync --extra mlx
```

### 2. Verify Installation

```bash
# Check MLX is installed
python -c "import mlx.core; print('✅ MLX installed')"

# Check backend detection
python -c "from nanochat.common import BACKEND; print(f'Backend: {BACKEND}')"
# Should output: Backend: mlx
```

### 3. Run Training

```bash
# Run the speedrun (auto-detects Mac and uses MLX)
bash speedrun.sh

# Or run individual training scripts
python -m scripts.base_train -- --depth=12  # Smaller model for Mac
```

---

## Differences from Linux/NVIDIA

### What's Different on Mac

| Feature | Mac (MLX) | Linux (PyTorch/CUDA) |
|---------|-----------|----------------------|
| **Backend** | MLX | PyTorch + CUDA |
| **GPUs** | Single device | Multi-GPU (8x) |
| **Memory** | Unified (8-128GB) | 640GB (8x80GB) |
| **Speed** | ~10-20% of 8xH100 | Full speed |
| **Launch** | `python` | `torchrun` |
| **Best for** | Development, small models | Production training |

### What Works

✅ All training scripts (base, mid, sft)
✅ Inference and evaluation
✅ Chat CLI and web interface
✅ Model checkpoints (compatible between backends)
✅ Same model architecture

### What's Limited

⚠️ No distributed training (single M-chip only)
⚠️ Memory limited by your Mac's unified memory
⚠️ Slower training than NVIDIA H100s
⚠️ AdamW only (no Muon optimizer yet)

---

## Recommended Settings for Mac

### Small Model (Fast iteration)
```bash
python -m scripts.base_train -- \
  --depth=8 \
  --device_batch_size=16 \
  --num_iterations=1000
```

### Medium Model (Good balance)
```bash
python -m scripts.base_train -- \
  --depth=12 \
  --device_batch_size=8 \
  --num_iterations=5000
```

### Large Model (If you have lots of RAM)
```bash
python -m scripts.base_train -- \
  --depth=16 \
  --device_batch_size=4 \
  --num_iterations=10000
```

---

## Memory Guidelines

Approximate memory usage per model size:

| Depth | Parameters | Memory (approx) | Recommended Mac |
|-------|------------|-----------------|-----------------|
| d8 | ~140M | 8-12GB | M1 8GB+ |
| d12 | ~310M | 16-24GB | M1 Pro 16GB+ |
| d16 | ~560M | 28-40GB | M2 Max 32GB+ |
| d20 | ~870M | 40-60GB | M3 Max 64GB+ |

If you run out of memory:
1. Reduce `--depth`
2. Reduce `--device_batch_size`
3. Reduce `--max_seq_len`

---

## Performance Tips

### For Faster Training
- Use smaller models (d8-d12)
- Reduce sequence length: `--max_seq_len=1024`
- Monitor memory: `Activity Monitor → Memory`

### For Best Quality
- Use larger batch size if memory allows
- More iterations: `--num_iterations=10000`
- Follow same training recipe as Linux

---

## Inference (Fast on Mac!)

Inference is very fast on Apple Silicon:

```bash
# Chat interface (recommended)
python -m scripts.chat_web

# CLI chat
python -m scripts.chat_cli -p "Tell me a joke"

# Interactive chat
python -m scripts.chat_cli
```

---

## Common Issues

### "MLX not found"
```bash
# Install MLX
uv sync --extra mlx

# Or manually
pip install mlx
```

### "Out of memory"
Reduce model size or batch size:
```bash
python -m scripts.base_train -- \
  --depth=8 \
  --device_batch_size=4
```

### "Backend is pytorch instead of mlx"
Check MLX installation:
```bash
python -c "import mlx.core"
```

Force MLX backend:
```bash
export NANOCHAT_BACKEND=mlx
python -m scripts.base_train
```

---

## What to Expect

### Training Times (M2 Max, d12 model)

- **Base training** (d12): ~8-12 hours
- **Midtraining**: ~2-4 hours
- **SFT**: ~1-2 hours

(vs. ~1-2 hours total on 8xH100)

### Quality

Same quality as Linux training:
- ✅ Same model architecture
- ✅ Same training data
- ✅ Compatible checkpoints
- ⚠️ Minor numerical differences (different hardware)

---

## Development Workflow

Recommended workflow for Mac users:

1. **Develop locally on Mac** 
   - Iterate quickly on small models
   - Test changes with d8-d12 models
   - Use inference for demos

2. **Train production models on Linux** 
   - Full-scale training on NVIDIA GPUs
   - Use larger models (d20+)
   - Faster training with multi-GPU

3. **Load trained models on Mac** 
   - Models trained on Linux work on Mac
   - Fast inference for local testing
   - Checkpoints are cross-platform

---

## Example Session

```bash
# 1. Setup
cd nanochat
uv sync --extra mlx
source .venv/bin/activate

# 2. Train a small model (quick test)
python -m scripts.base_train -- \
  --depth=8 \
  --num_iterations=100 \
  --run=test_mac

# 3. Chat with your model
python -m scripts.chat_web

# 4. Visit http://localhost:8000 in your browser
```

---

## Hardware Recommendations

| Mac Model | RAM | Best Model Size | Training Speed |
|-----------|-----|-----------------|----------------|
| M1 8GB | 8GB | d6-d8 | Slow |
| M1 Pro 16GB | 16GB | d8-d10 | Moderate |
| M2 Max 32GB | 32GB | d10-d14 | Good |
| M3 Max 64GB | 64GB | d14-d18 | Great |
| M3 Max 128GB | 128GB | d18-d22 | Excellent |

---

## Advanced Options

### Force PyTorch on Mac
If CUDA-capable eGPU:
```bash
export NANOCHAT_BACKEND=pytorch
python -m scripts.base_train
```

### Custom Data Directory
```bash
export NANOCHAT_BASE_DIR=/path/to/storage
python -m scripts.base_train
```

### Disable WandB Logging
```bash
python -m scripts.base_train -- --run=dummy
```

---

## Getting Help

- Full details: [MLX_INTEGRATION_COMPLETE.md](MLX_INTEGRATION_COMPLETE.md)
- Design doc: [MLX_INTEGRATION_PLAN.md](MLX_INTEGRATION_PLAN.md)
- Main README: [README.md](README.md)

---

