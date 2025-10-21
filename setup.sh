#!/bin/bash
set -e

echo "=========================================="
echo "Gillies Workshop Hackathon 2025 Setup"
echo "=========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on a CUDA-enabled system
print_status "Checking CUDA availability..."
if ! command -v nvidia-smi &> /dev/null; then
    print_warning "nvidia-smi not found. Make sure you're running on a GPU instance."
else
    nvidia-smi
    print_status "CUDA detected successfully!"
fi

# Install Miniconda if not already installed
if ! command -v conda &> /dev/null; then
    print_status "Miniconda not found. Installing Miniconda..."

    # Download Miniconda installer
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh

    # Install Miniconda
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3

    # Initialize conda
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash

    # Clean up
    rm /tmp/miniconda.sh

    print_status "Miniconda installed successfully!"
else
    print_status "Miniconda already installed."
    eval "$(conda shell.bash hook)"
fi

# Install uv package manager
if ! command -v uv &> /dev/null; then
    print_status "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    print_status "uv installed successfully!"
else
    print_status "uv already installed."
fi

# Create conda environment
ENV_NAME="hackathon-2025"
print_status "Creating conda environment: $ENV_NAME..."

if conda env list | grep -q "^$ENV_NAME "; then
    print_warning "Environment $ENV_NAME already exists. Removing it..."
    conda env remove -n $ENV_NAME -y
fi

# Create environment with Python 3.10 (compatible with RAPIDS)
conda create -n $ENV_NAME python=3.10 -y
print_status "Conda environment created successfully!"

# Activate the environment
print_status "Activating conda environment..."
source activate $ENV_NAME

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install RAPIDS packages from NVIDIA PyPI
print_status "Installing RAPIDS packages (this may take several minutes)..."
pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu13==25.10.*" \
    "dask-cudf-cu13==25.10.*" \
    "cuml-cu13==25.10.*" \
    "cugraph-cu13==25.10.*" \
    "nx-cugraph-cu13==25.10.*" \
    "cuxfilter-cu13==25.10.*" \
    "cucim-cu13==25.10.*" \
    "pylibraft-cu13==25.10.*" \
    "raft-dask-cu13==25.10.*" \
    "cuvs-cu13==25.10.*"

print_status "RAPIDS packages installed successfully!"

# Install honeybee-ml
print_status "Installing honeybee-ml..."
pip install honeybee-ml

# Install JupyterLab and common data science packages
print_status "Installing JupyterLab and essential packages..."
pip install jupyterlab ipywidgets matplotlib seaborn plotly scikit-learn pandas numpy

# Install HuggingFace CLI for dataset downloads
print_status "Installing HuggingFace CLI..."
pip install huggingface_hub[cli]

# Create data directory
DATA_DIR="$HOME/hackathon-data"
print_status "Creating data directory at $DATA_DIR..."
mkdir -p "$DATA_DIR"

# Download datasets from HuggingFace
print_status "Downloading datasets from HuggingFace (Lab-Rasool/hackathon-2025)..."
print_warning "Note: Dataset downloads are large (~700GB total). This may take a while..."

# Download train split
print_status "Downloading training dataset..."
huggingface-cli download Lab-Rasool/hackathon-2025 \
    --repo-type dataset \
    --local-dir "$DATA_DIR/train" \
    --include "train/*"

# Download test split
print_status "Downloading test dataset..."
huggingface-cli download Lab-Rasool/hackathon-2025 \
    --repo-type dataset \
    --local-dir "$DATA_DIR/test" \
    --include "test/*"

print_status "Datasets downloaded to $DATA_DIR"

# Create a quick verification script
print_status "Creating verification script..."
cat > "$HOME/verify_setup.py" << 'EOF'
#!/usr/bin/env python3
"""Verify that the hackathon environment is set up correctly."""

import sys

def check_import(module_name, package_name=None):
    """Try to import a module and report status."""
    package_name = package_name or module_name
    try:
        __import__(module_name)
        print(f"✓ {package_name} is installed")
        return True
    except ImportError:
        print(f"✗ {package_name} is NOT installed")
        return False

print("Verifying Hackathon 2025 Environment Setup")
print("=" * 50)

# Check RAPIDS packages
rapids_packages = [
    ("cudf", "cuDF"),
    ("dask_cudf", "Dask-cuDF"),
    ("cuml", "cuML"),
    ("cugraph", "cuGraph"),
    ("cucim", "cuCIM"),
    ("pylibraft", "pyLibRaft"),
    ("cuvs", "cuVS"),
]

# Check other packages
other_packages = [
    ("honeybee_ml", "honeybee-ml"),
    ("jupyterlab", "JupyterLab"),
    ("pandas", "Pandas"),
    ("numpy", "NumPy"),
    ("sklearn", "scikit-learn"),
]

print("\nRAPIDS Packages:")
rapids_ok = all(check_import(mod, name) for mod, name in rapids_packages)

print("\nOther Packages:")
other_ok = all(check_import(mod, name) for mod, name in other_packages)

print("\n" + "=" * 50)
if rapids_ok and other_ok:
    print("✓ All packages installed successfully!")
    sys.exit(0)
else:
    print("✗ Some packages are missing. Please review the setup.")
    sys.exit(1)
EOF

chmod +x "$HOME/verify_setup.py"

# Run verification
print_status "Running environment verification..."
python "$HOME/verify_setup.py"

# Create activation helper script
print_status "Creating activation helper script..."
cat > "$HOME/activate_hackathon.sh" << EOF
#!/bin/bash
# Activation script for Hackathon 2025 environment
eval "\$(conda shell.bash hook)"
conda activate $ENV_NAME
export HACKATHON_DATA="$DATA_DIR"
echo "Hackathon 2025 environment activated!"
echo "Data directory: \$HACKATHON_DATA"
echo ""
echo "To start JupyterLab, run: jupyter lab --ip=0.0.0.0 --port=8888 --no-browser"
EOF

chmod +x "$HOME/activate_hackathon.sh"

echo ""
echo "=========================================="
echo "${GREEN}Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Environment name: $ENV_NAME"
echo "Data directory: $DATA_DIR"
echo ""
echo "To activate the environment in a new shell, run:"
echo "  source ~/activate_hackathon.sh"
echo ""
echo "Or manually activate with:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To start JupyterLab:"
echo "  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser"
echo ""
echo "To verify your setup:"
echo "  python ~/verify_setup.py"
echo ""
echo "Dataset location: $DATA_DIR"
echo "  - Training data: $DATA_DIR/train/"
echo "  - Test data: $DATA_DIR/test/"
echo ""
echo "Happy coding! 🚀"
echo ""
