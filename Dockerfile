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
COPY IDM-VTON-build /app/IDM-VTON-build

# Copy the script to download models
COPY download_models.sh /app/download_models.sh

# Run the script to download models
RUN chmod +x /app/download_models.sh && /app/download_models.sh

# Install Python dependencies
RUN pip3 install --upgrade pip && \
pip3 install --no-cache-dir \
torch==2.2.1 \
torchvision==0.17.1 \
torchaudio==2.2.1 \
--index-url https://download.pytorch.org/whl/cu121 && \
pip3 install --no-cache-dir -r /app/IDM-VTON/requirements.txt && \
rm -rf /root/.cache/pip

# Set the default command to run the application
CMD ["python3", "/app/IDM-VTON/handler.py"]
