FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

RUN pip install transformers peft accelerate bitsandbytes scipy pyyaml requests

# flash-attn requires CUDA compiler tools (devel image) and no build isolation
RUN pip install flash-attn --no-build-isolation

WORKDIR /workspace
