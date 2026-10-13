# Llama vision inference notebook

This repository contains a notebook for Meta Llama 3.2 11B Vision Instruct: load the model, download a sample image, and generate an image-based response. It is an inference example, not a complete autonomous agent application.

## Run

Use a Jupyter environment with a GPU and enough memory to load the 11B model. Access to the gated model must be approved on Hugging Face. The first notebook cell installs Transformers, Accelerate, bitsandbytes, PyTorch, and nbformat.

Provide authentication through your environment before starting Jupyter:

```sh
export HF_TOKEN='<your Hugging Face token>'
jupyter lab
```

Open `Llama_load (1).ipynb` and run the cells in order. The authentication cell fails with an actionable message if `HF_TOKEN` is missing. Do not place credentials in notebook source or committed output.

Model loading and inference require network access and a suitable GPU; they are not verified by the repository's static notebook check.

## Public pretrained CPU vision lane

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.txt
python cpu_vision.py
# Analyze a local image:
python cpu_vision.py --image photograph.png
python -m http.server 8000 --directory cpu-vision-output
```

This executes a pinned, publicly accessible pretrained 5.72M-parameter ViT-tiny checkpoint on actual images, saves top-class probabilities/entropy and a browser report, and verifies deterministic inference without parameter mutation. First execution downloads weights and the public Hugging Face sample image; no account or token is required. It is ImageNet classification, not a replacement claim for the retained Llama-3.2-11B gated image-reasoning experiment, an autonomous agent, or calibrated confidence. Checkpoint: https://huggingface.co/WinKawaks/vit-tiny-patch16-224 .
