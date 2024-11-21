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

# Install git-lfs
RUN apt-get update && apt-get install -y git-lfs && git lfs install

# Clone the project from the 'cloner' branch
RUN git clone --branch cloner https://github.com/mertsaglam/IDM-VTON-build.git /IDM-VTON-build

WORKDIR /IDM-VTON-build

RUN pip install -r requirements.txt && pip cache purge

RUN pip install huggingface_hub==0.25.2 matplotlib

RUN chmod +x download_models.sh && bash ./download_models.sh


CMD /IDM-VTON-build/src/start.sh