# CSE 151B Competition -- Banana Bread

## Hardware & Runtime
- **GPU Type Used:** NVIDIA A100 (40GB) / RTX 4090 (or equivalent with $\ge$ 16GB VRAM for INT8/BF16 loading)
- **Approximate Total Inference Time:** ~20-30 minutes for the full 943-question private test set (utilizing vLLM's chunked generation with `N_VOTE=3` majority voting).

## Model Weights & Setup
You do not need to download or place any model weights manually. The codebase is configured to pull both the base model and our fine-tuned weights automatically from the Hugging Face Hub during initialization.

### Prerequisites
Make sure you have the required dependencies installed:

`pip install vllm transformers torch pandas` or (`pip install -r requirements.txt`)
