FROM python:3.10.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_DRIVER_CAPABILITIES=all \
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1

# Setup system packages
COPY builder/setup.sh /setup.sh
RUN /bin/bash /setup.sh && \
    rm /setup.sh

COPY builder/requirements.txt /requirements.txt 
RUN pip install -r /requirements.txt && \
    rm /requirements.txt

# Install git-lfs and clean up apt cache
RUN apt-get update && apt-get install -y git-lfs && git lfs install && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Clone the project from the 'cloner' branch directly into /app
RUN git clone --branch cloner https://github.com/mertsaglam/IDM-VTON-build.git /app

WORKDIR /app

# Install Python dependencies and clean up pip cache
RUN pip install -r /app/requirements.txt && \
    pip install huggingface_hub==0.25.2 matplotlib && \
    pip cache purge

# Add the download_models.sh script
COPY builder/download_models.sh /app/download_models.sh

# Make the download_models.sh script executable and run it
RUN chmod +x /app/download_models.sh && bash /app/download_models.sh

CMD /app/src/start.sh