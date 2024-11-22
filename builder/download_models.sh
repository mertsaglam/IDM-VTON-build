
#!/bin/bash

# Define the models to download
models=(
    "yisol/IDM-VTON"
    "madebyollin/sdxl-vae-fp16-fix"
)

# Define the destination folder
dest_folder="/app/models"

# Create the destination folder if it doesn't exist
mkdir -p $dest_folder

# Download each model
for model in "${models[@]}"; do
    echo "Downloading $model..."
    git lfs install
    git clone https://huggingface.co/$model $dest_folder/$(basename $model)
done

echo "All models downloaded successfully."