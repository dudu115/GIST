#!/bin/bash


sudo apt-get update
sudo apt-get install -y python3-dev build-essential ninja-build
sudo apt-get install ffmpeg libsm6 libxext6  -y

cd trl
pip install -e .

pip install torch torchvision torchaudio

pip install transformers
pip install accelerate

pip install deepspeed --no-deps
pip install mpi4py

pip install qwen-vl-utils
pip install flash-attn
pip install trl
pip install nltk
pip install fastapi
pip install python-multipart
pip install uvicorn
pip install peft
pip install datasets
pip install bitsandbytes
pip install flash-attn

pip install openai
pip install distro
pip install httpx
pip install anyio
pip install typing-extensions

pip install latex2sympy2-extended
pip install math_verify
pip install jsonlines

pip install Pillow
pip install numpy

pip install regex
pip install deepspeed
pip install decord timm opencv-python imageio


echo "All dependencies installed successfully!"
