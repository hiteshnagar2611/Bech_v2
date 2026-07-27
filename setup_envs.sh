#!/bin/bash
set -e

TORCH_INDEX="https://download.pytorch.org/whl/cu124"

echo "=== Setting up ESM Meta environment (ESM1b + ESM2) ==="
conda run -n esm_meta pip uninstall torch -y 2>/dev/null || true
conda run -n esm_meta pip install --index-url "$TORCH_INDEX" "torch>=2.1,<2.5"
conda run -n esm_meta pip install fair-esm transformers "numpy<2" pandas scikit-learn matplotlib seaborn tqdm
echo "esm_meta done"

echo "=== Setting up ESM3 environment ==="
conda run -n esm3_env pip uninstall torch -y 2>/dev/null || true
conda run -n esm3_env pip install --index-url "$TORCH_INDEX" "torch>=2.1,<2.5"
conda run -n esm3_env pip install "esm>=3.1" huggingface_hub "numpy<2" pandas scikit-learn matplotlib seaborn tqdm
echo "esm3_env done"

echo "=== Environments ready ==="

echo "=== Setting up ProtT5 environment ==="
conda create -n prott5_env python=3.10 -y 2>/dev/null || true
conda run -n prott5_env pip install --index-url "$TORCH_INDEX" "torch>=2.1,<2.5" 2>/dev/null || true
conda run -n prott5_env pip install transformers "numpy<2" pandas scikit-learn matplotlib seaborn tqdm 2>/dev/null || true
echo "prott5_env done"

echo "=== Setting up DNA environment ==="
conda create -n dna_env python=3.10 -y 2>/dev/null || true
conda run -n dna_env pip install --index-url "$TORCH_INDEX" "torch>=2.1,<2.5" 2>/dev/null || true
conda run -n dna_env pip install "transformers==4.44.2" tokenizers sentencepiece "numpy<2" pandas scikit-learn matplotlib seaborn tqdm accelerate einops 2>/dev/null || true
CUDA_HOME=/ibdc-hpc/apps1/cuda-12.4 conda run -n dna_env pip install flash-attn --no-build-isolation 2>/dev/null || true
echo "dna_env done"

echo "=== All environments ready ==="
