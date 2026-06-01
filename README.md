# CSE 151B Competition -- Banana Bread

## Hardware & Runtime
- **GPU Type Used:** NVIDIA RTX 6000 Blackwell or A30 (depending on Datahub availability)

## Model Weights & Setup
You do not need to download or place any model weights manually. The codebase is configured to pull both the base model and our fine-tuned weights automatically from the Hugging Face Hub during initialization.

### Prerequisites
Make sure you have the required dependencies installed:

`pip install vllm transformers torch pandas` or (`pip install -r requirements.txt`)

`run_inference()` should run automatically if you run `python run_inference.py`
