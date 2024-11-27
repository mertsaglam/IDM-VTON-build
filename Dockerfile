
# Base image with CUDA 12.1 support
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_DRIVER_CAPABILITIES=all \
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Etc/UTC

# Install system dependencies and Python 3.10.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    build-essential \
    libopencv-dev \
    ffmpeg \
    git \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as the default Python version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Copy and install dependencies from the builder requirements file
COPY builder/requirements.txt /builder_requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /builder_requirements.txt && \
    rm /builder_requirements.txt

# Install additional Python packages explicitly
RUN pip install --no-cache-dir \
    torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install xformers==0.0.24 bitsandbytes==0.43.0 gradio huggingface_hub==0.25.2 matplotlib

# Install git-lfs and clean up
RUN apt-get update && apt-get install -y git-lfs && git lfs install && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Clone the project repository
RUN git clone --branch cloner https://github.com/mertsaglam/IDM-VTON-build.git /app

# Install dependencies from the cloned project requirements file
RUN pip install --no-cache-dir -r /app/requirements.txt

# Ensure the start.sh script is executable
COPY src/start.sh /app/src/start.sh
RUN chmod +x /app/src/start.sh

# Set the working directory
WORKDIR /app

# Download model files using aria2c
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M 'https://huggingface.co/yisol/IDM-VTON/resolve/main/densepose/model_final_162be9.pkl' -d './models/IDM-VTON/densepose' -o 'model_final_162be9.pkl'
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M 'https://huggingface.co/yisol/IDM-VTON/resolve/main/humanparsing/parsing_atr.onnx'  -d './models/IDM-VTON/humanparsing' -o 'parsing_atr.onnx'
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M 'https://huggingface.co/yisol/IDM-VTON/resolve/main/humanparsing/parsing_lip.onnx' -d './models/IDM-VTON/humanparsing' -o 'parsing_lip.onnx'
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M 'https://huggingface.co/yisol/IDM-VTON/resolve/main/image_encoder/config.json' -d './models/IDM-VTON/image_encoder' -o 'config.json'
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M 'https://huggingface.co/yisol/IDM-VTON/resolve/main/image_encoder/model.safetensors' -d './models/IDM-VTON/image_encoder' -o 'model.safetensors'
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M 'https://huggingface.co/yisol/IDM-VTON/resolve/main/openpose/ckpts/body_pose_model.pth' -d './models/IDM-VTON/openpose/ckpts' -o 'body_pose_model.pth'
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M 'https://huggingface.co/yisol/IDM-VTON/resolve/main/scheduler/scheduler_config.json' -d './models/IDM-VTON/scheduler' -o 'scheduler_config.json'
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M 'https://huggingface.co/yisol/IDM-VTON/resolve/main/text_encoder/config.json' -d './models/IDM-VTON/text_encoder' -o 'config.json'
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M 'https://huggingface.co/yisol/IDM-VTON/resolve/main/text_encoder/model.safetensors' -d './models/IDM-VTON/text_encoder' -o 'model.safetensors'
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M 'https://huggingface.co/yisol/IDM-VTON/resolve/main/text_encoder_2/config.json' -d './models/IDM-VTON/text_encoder_2' -o 'config.json'
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M 'https://huggingface.co/yisol/IDM-VTON/resolve/main/text_encoder_2/model.safetensors' -d './models/IDM-VTON/text_encoder_2' -o 'model.safetensors'
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M 'https://huggingface.co/yisol/IDM-VTON/resolve/main/tokenizer/merges.txt' -d './models/IDM-VTON/tokenizer' -o 'merges.txt'
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M 'https://huggingface.co/yisol/IDM-VTON/resolve/main/tokenizer/special_tokens_map.json'   -d './models/IDM-VTON/tokenizer' -o 'special_tokens_map.json'
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M 'https://huggingface.co/yisol/IDM-VTON/resolve/main/tokenizer/tokenizer_config.json' -d './models/IDM-VTON/tokenizer' -o 'tokenizer_config.json'
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M 'https://huggingface.co/yisol/IDM-VTON/resolve/main/tokenizer/vocab.json' -d './models/IDM-VTON/tokenizer' -o 'vocab.json'
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M 'https://huggingface.co/yisol/IDM-VTON/resolve/main/tokenizer_2/merges.txt' -d './models/IDM-VTON/tokenizer_2' -o 'merges.txt'
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M 'https://huggingface.co/yisol/IDM-VTON/resolve/main/tokenizer_2/special_tokens_map.json' -d './models/IDM-VTON/tokenizer_2' -o 'special_tokens_map.json'
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M 'https://huggingface.co/yisol/IDM-VTON/resolve/main/tokenizer_2/special_tokens_map.json' -d './models/IDM-VTON/tokenizer_2' -o 'special_tokens_map.json'
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M 'https://huggingface.co/yisol/IDM-VTON/resolve/main/tokenizer_2/vocab.json' -d './models/IDM-VTON/tokenizer_2' -o 'vocab.json'
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M 'https://huggingface.co/yisol/IDM-VTON/resolve/main/unet/config.json' -d './models/IDM-VTON/unet' -o 'config.json'
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M 'https://huggingface.co/yisol/IDM-VTON/resolve/main/unet/diffusion_pytorch_model.bin' -d './models/IDM-VTON/unet' -o 'diffusion_pytorch_model.bin'
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M 'https://huggingface.co/yisol/IDM-VTON/resolve/main/unet_encoder/config.json' -d './models/IDM-VTON/unet_encoder' -o 'config.json'
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M 'https://huggingface.co/yisol/IDM-VTON/resolve/main/unet_encoder/diffusion_pytorch_model.safetensors' -d './models/IDM-VTON/unet_encoder' -o 'diffusion_pytorch_model.safetensors'
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M 'https://huggingface.co/yisol/IDM-VTON/resolve/main/vae/config.json' -d './models/IDM-VTON/vae' -o 'config.json'
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M 'https://huggingface.co/yisol/IDM-VTON/resolve/main/vae/diffusion_pytorch_model.safetensors' -d './models/IDM-VTON/vae' -o 'diffusion_pytorch_model.safetensors'
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M 'https://huggingface.co/yisol/IDM-VTON/resolve/main/model_index.json' -d './models/IDM-VTON' -o 'model_index.json'
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M 'https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/config.json' -d './models/sdxl-vae-fp16-fix' -o 'config.json'
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M 'https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/diffusion_pytorch_model.bin' -d './models/sdxl-vae-fp16-fix' -o 'diffusion_pytorch_model.bin'
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M 'https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/diffusion_pytorch_model.safetensors' -d './models/sdxl-vae-fp16-fix' -o 'diffusion_pytorch_model.safetensors'
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M 'https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl.vae.safetensors' -d './models/sdxl-vae-fp16-fix' -o 'sdxl.vae.safetensors'
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M 'https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors' -d './models/sdxl-vae-fp16-fix' -o 'sdxl_vae.safetensors'



# Clean up pip cache
RUN pip cache purge

# Set the entrypoint to start the application
ENTRYPOINT ["/app/src/start.sh"]
