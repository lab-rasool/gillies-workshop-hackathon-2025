# Brev Launchable Setup Guide

This guide provides step-by-step instructions for creating a Brev launchable for the Gillies Workshop Hackathon 2025.

## Prerequisites

- Brev account at [brev.nvidia.com](https://brev.nvidia.com)
- GitHub repository: https://github.com/lab-rasool/gillies-workshop-hackathon-2025
- Basic understanding of GPU computing and machine learning

## Launchable Configuration

### Step 1: Files and Runtime

1. Navigate to [brev.nvidia.com](https://brev.nvidia.com)
2. Click **Launchables** tab in the top navigation
3. Click **Create Launchable**
4. Select **Git Repository** as the code source
5. Enter repository URL:
   ```
   https://github.com/lab-rasool/gillies-workshop-hackathon-2025
   ```
6. Select **VM Mode** as the runtime (recommended)
7. Click **Next**

**Why VM Mode?**
- Provides Ubuntu 22.04 with Docker, Python, and CUDA pre-installed
- Allows custom setup scripts for package installation
- More flexible than containers for complex environments
- Easier for participants to customize their setup

### Step 2: Configure the Runtime

1. In the setup script section, upload or paste the contents of `setup.sh` from this repository

   **Paste the following script:**
   ```bash
   #!/bin/bash
   # See setup.sh in the repository for the full script
   ```

   **Or upload the file directly**: `setup.sh`

2. The setup script will automatically:
   - Install Miniconda
   - Install uv package manager
   - Create a conda environment named `hackathon-2025`
   - Install all RAPIDS packages (cuDF, cuML, cuGraph, etc.)
   - Install honeybee-ml and JupyterLab
   - Download train and test datasets from HuggingFace
   - Create helper scripts for easy environment activation

3. Click **Next**

### Step 3: Jupyter and Networking

1. **Jupyter Notebook Experience**: Select **Yes**
   - This provides participants with one-click access to JupyterLab
   - Ideal for interactive data exploration and model development
   - Automatically configures Jupyter to run on port 8888

2. **Pre-expose tunnels/services**:
   - Add port **8888** for JupyterLab
     - Protocol: TCP
     - Port: 8888
     - Label: "JupyterLab"
     - Public: Yes (or configure authentication as needed)

3. Click **Next**

### Step 4: Compute Configuration

**Recommended GPU Configuration:**

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA L40S (48GB VRAM) |
| CPU | 16+ vCPUs |
| RAM | 64GB system memory |
| Storage | 2048GB SSD (for datasets ~700GB, consider larger) |
| CUDA | 13.0+ |

**Why L40S?**
- 48GB VRAM sufficient for large pathology images
- Excellent performance for RAPIDS libraries
- Good balance of cost and performance
- Supports mixed precision training

**Alternative GPU Options:**

1. **Budget Option**: RTX 4090 (24GB VRAM)
   - Lower cost
   - May struggle with largest whole-slide images
   - Good for initial development and testing

2. **High Performance**: A100 (80GB VRAM)
   - Maximum memory for largest datasets
   - Best performance for deep learning
   - Higher cost

3. **Balanced**: A10G (24GB VRAM) or L4 (24GB VRAM)
   - Mid-range cost
   - Adequate for most tasks

**Storage Considerations:**
- Training dataset: ~481GB (600 pathology slides)
- Test dataset: ~213GB (265 pathology slides)
- Total dataset size: ~700GB
- Recommended storage: 200GB minimum (if downloading only train or test split)
- Full dataset: 1TB+ recommended

**Cost Optimization Tips:**
- Participants can download only the training split initially
- Test split can be downloaded when ready to make submissions
- Use lower GPU tiers for development, scale up for training

### Step 5: Final Review

1. **Name your Launchable**:
   ```
   Gillies Workshop Hackathon 2025
   ```

2. **Description** (optional but recommended):
   ```
   GPU-accelerated environment for the 2025 Dr. Robert Gillies Machine Learning
   Workshop at Moffitt Cancer Center. Pre-configured with RAPIDS, honeybee-ml,
   and competition datasets for survival prediction using pathology imaging.
   ```

3. **Preview the Deploy Page**:
   - Review the configuration summary
   - Check that setup script is included
   - Verify GPU and storage settings
   - Confirm Jupyter is enabled

4. Click **Create Launchable**

5. **Save the shareable link** - this is what you'll distribute to participants

## Post-Creation Configuration

### Sharing the Launchable

1. Navigate to the **Launchables** tab
2. Find your created launchable
3. Options for sharing:
   - **Copy Link**: Direct URL to deploy page
   - **Copy Markdown Badge**: Embed in README or documentation
   - **Toggle Access**: Control who can view/deploy

**Recommended Access Settings:**
- Public: Yes (for open hackathon)
- Require authentication: Optional (for tracking participants)

### Testing the Launchable

Before distributing to participants, test the launchable yourself:

1. Deploy the launchable using the shareable link
2. Wait for setup script to complete (~15-30 minutes for full setup)
3. Verify environment:
   ```bash
   source ~/activate_hackathon.sh
   python ~/verify_setup.py
   ```
4. Check dataset downloads:
   ```bash
   ls -lh ~/hackathon-data/train/
   ls -lh ~/hackathon-data/test/
   ```
5. Test JupyterLab access (should auto-open or available on port 8888)
6. Run a simple test notebook to verify RAPIDS and honeybee-ml

### Monitoring Usage

Track launchable metrics from the Brev dashboard:
- **Views**: Number of times deploy page was viewed
- **Deployments**: Number of active instances deployed
- **Resource Usage**: GPU hours consumed

## Participant Instructions

Once the launchable is created, provide participants with:

### Quick Start Instructions

1. **Access the Launchable**:
   - Click the launchable link: [INSERT_YOUR_LAUNCHABLE_URL]
   - Sign in to Brev (create account if needed)

2. **Deploy**:
   - Click **Deploy Launchable**
   - Wait for instance to provision (~2-3 minutes)
   - Setup script will run automatically (~15-30 minutes)

3. **Access JupyterLab**:
   - Click the JupyterLab button when setup completes
   - Or access via the exposed port 8888

4. **Activate Environment**:
   ```bash
   source ~/activate_hackathon.sh
   ```

5. **Verify Setup**:
   ```bash
   python ~/verify_setup.py
   ```

6. **Start Coding**:
   - Datasets are in `~/hackathon-data/`
   - Repository code is in `~/gillies-workshop-hackathon-2025/`
   - Create notebooks or scripts in your preferred directory

### Troubleshooting for Participants

**Setup script failed:**
- Check the instance logs
- Re-run setup script manually: `bash ~/gillies-workshop-hackathon-2025/setup.sh`

**JupyterLab not accessible:**
- Start manually: `jupyter lab --ip=0.0.0.0 --port=8888 --no-browser`
- Check port 8888 is exposed in Brev settings

**Out of disk space:**
- Check storage: `df -h`
- Delete unnecessary files or request larger storage

**CUDA errors:**
- Verify GPU: `nvidia-smi`
- Check CUDA version: `nvcc --version`
- Ensure instance has GPU attached

## Advanced Configuration

### Custom Container (Alternative to VM Mode)

If you prefer using a custom container instead of VM Mode:

1. Create a Dockerfile:
   ```dockerfile
   FROM rapidsai/rapidsai:25.10-cuda13.0-runtime-ubuntu22.04-py3.10

   # Install additional packages
   RUN pip install honeybee-ml jupyterlab ipywidgets

   # Copy setup scripts
   COPY setup.sh /opt/setup.sh
   RUN chmod +x /opt/setup.sh

   # Download datasets (optional - can be done at runtime)
   # RUN /opt/setup.sh

   EXPOSE 8888

   CMD ["bash"]
   ```

2. In Brev Step 2, select **Custom Container**
3. Upload your Docker Compose or Dockerfile
4. Configure as needed

**Pros of Container Approach:**
- Faster startup (if datasets pre-downloaded)
- More reproducible environment
- Pre-configured everything

**Cons of Container Approach:**
- Larger container size (~50GB+ with datasets)
- Less flexible for participants to modify
- Requires container registry (Docker Hub, NVIDIA NGC)

### Environment Variables

Consider adding these environment variables in Brev settings:

- `HACKATHON_DATA`: Path to dataset directory (set in setup script)
- `CUDA_VISIBLE_DEVICES`: Control which GPUs are visible
- `RAPIDS_NO_INITIALIZE`: Skip RAPIDS initialization (for faster imports)
