FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

RUN pip install transformers peft accelerate bitsandbytes scipy pyyaml requests

WORKDIR /workspace
