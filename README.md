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
