# Prompt Optimizer (Mac, llama.cpp + FastAPI)

Evolutionary prompt optimization on open LLMs, Metal-accelerated via llama.cpp.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
export CMAKE_ARGS="-DLLAMA_METAL=on"; export FORCE_CMAKE=1
pip install -r requirements.txt

# Download a quantized model (choose one)
pip install huggingface_hub
huggingface-cli login  # if needed (Llama)
./scripts/download_model.sh mistral

# Run the API
./scripts/run_dev.sh
