FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

RUN pip install transformers peft accelerate bitsandbytes scipy pyyaml requests

# flash-attn cannot be cross-compiled (Mac ARM -> AMD64) — install manually on the pod after start:
#   pip install flash-attn --no-build-isolation

WORKDIR /workspace
