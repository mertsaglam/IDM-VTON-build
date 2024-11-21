FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 and system dependencies
RUN apt-get update && \
apt-get install -y --no-install-recommends \
software-properties-common \
build-essential \
python3.10-dev \
python3-pip \
python3-tk \
apt-utils \
curl \
wget \
vim \
sudo \
git \
ffmpeg \
libsm6 \
libxext6 && \
apt-get clean && \
rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy project files
COPY IDM-VTON /app/IDM-VTON

# Install Python dependencies
RUN pip3 install --upgrade pip && \
pip3 install --no-cache-dir \
torch==2.2.1 \
torchvision==0.17.1 \
torchaudio==2.2.1 \
--index-url https://download.pytorch.org/whl/cu121 && \
pip3 install --no-cache-dir -r /app/IDM-VTON/requirements.txt && \
rm -rf /root/.cache/pip

# Set Hugging Face cache directory
ENV HF_HOME='/app/models'

# Ensure model cache directory exists
RUN mkdir -p /app/models

# Copy preloaded models to the cache directory
COPY hub /app/models/hub

# Set the default command to run the application
CMD ["python3", "/app/IDM-VTON/handler.py"]
