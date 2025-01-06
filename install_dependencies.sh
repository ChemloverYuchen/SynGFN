#!/bin/bash

WHL_DIR="./whl_packages"

# Check if the directory exists, if not, create it
if [ ! -d "${WHL_DIR}" ]; then
    echo "Creating directory ${WHL_DIR} for storing .whl files..."
    mkdir -p ${WHL_DIR}
else
    echo "Directory ${WHL_DIR} already exists. Skipping creation..."
fi

# Function to download a .whl file if it does not already exist
download_whl() {
    local url=$1
    local filepath=${WHL_DIR}/$(basename ${url})

    if [ ! -f "${filepath}" ]; then
        echo "Downloading ${url}..."
        wget -P ${WHL_DIR} ${url}
    else
        echo "File ${filepath} already exists. Skipping download..."
    fi
}

# Step 1: Install torch
echo "Installing PyTorch 2.1.0 with CUDA 11.8..."
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Step 2: Install general dependencies
echo "Installing general dependencies from requirements.txt..."
pip install -r requirements.txt

# Step 3: Download and install specific .whl dependencies
echo "Downloading and installing specific .whl dependencies into ${WHL_DIR}..."

# Download and install pyg-lib
download_whl https://data.pyg.org/whl/torch-2.1.0%2Bcu118/pyg_lib-0.4.0%2Bpt21cu118-cp39-cp39-linux_x86_64.whl
pip install ${WHL_DIR}/pyg_lib-0.4.0+pt21cu118-cp39-cp39-linux_x86_64.whl

# Download and install torch-cluster
download_whl https://data.pyg.org/whl/torch-2.1.0%2Bcu118/torch_cluster-1.6.3%2Bpt21cu118-cp39-cp39-linux_x86_64.whl
pip install ${WHL_DIR}/torch_cluster-1.6.3+pt21cu118-cp39-cp39-linux_x86_64.whl

# Download and install torch-scatter
download_whl https://data.pyg.org/whl/torch-2.1.0%2Bcu118/torch_scatter-2.1.2%2Bpt21cu118-cp39-cp39-linux_x86_64.whl
pip install ${WHL_DIR}/torch_scatter-2.1.2+pt21cu118-cp39-cp39-linux_x86_64.whl

# Download and install torch-sparse
download_whl https://data.pyg.org/whl/torch-2.1.0%2Bcu118/torch_sparse-0.6.18%2Bpt21cu118-cp39-cp39-linux_x86_64.whl
pip install ${WHL_DIR}/torch_sparse-0.6.18+pt21cu118-cp39-cp39-linux_x86_64.whl

# Download and install torch-spline-conv
download_whl https://data.pyg.org/whl/torch-2.1.0%2Bcu118/torch_spline_conv-1.2.2%2Bpt21cu118-cp39-cp39-linux_x86_64.whl
pip install ${WHL_DIR}/torch_spline_conv-1.2.2+pt21cu118-cp39-cp39-linux_x86_64.whl

echo "All dependencies have been installed successfully."
