# Gillies Workshop Hackathon 2025 - WSI Preprocessing & Embedding Extraction

Whole Slide Image (WSI) preprocessing toolkit for the **Dr. Robert Gillies Machine Learning Workshop at Moffitt Cancer Center**. This project demonstrates how to extract tissue patches from pathology slides and generate foundation model embeddings using the **honeybee-ml** library.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Tutorial: Extracting Tissue Patches](#tutorial-extracting-tissue-patches)
- [Tutorial: Generating Embeddings](#tutorial-generating-embeddings)
- [Output Format](#output-format)
- [Complete Example](#complete-example)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Overview

This repository contains tools for preprocessing whole slide images (WSI) from cancer pathology datasets. The workflow includes:

1. **Loading WSI slides** - Read large pathology images (e.g., `.tif` files)
2. **Extracting tissue patches** - Divide slides into manageable tiles
3. **Tissue detection** - Identify high-quality tissue regions (classical + deep learning methods)
4. **Embedding generation** - Extract 1024-dimensional feature vectors using UNI foundation model
5. **Saving outputs** - Store embeddings with metadata for downstream analysis

**Use Cases:**
- Cancer survival prediction
- Tumor classification
- Biomarker discovery
- Multi-modal learning (pathology + clinical data)

---

## Installation

### Prerequisites

- **Python**: 3.12+ (recommended)
- **CUDA**: 11.3+ (for GPU acceleration)
- **Storage**: ~2GB for models + space for slide data

### Setup Steps

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd gillies-workshop-hackathon-2025
   ```

2. **Run setup script:**
   ```bash
   bash setup.sh
   ```

   This installs:
   - `honeybee-ml` - Medical imaging toolkit
   - `huggingface_hub` - Model downloading
   - `numpy`, `matplotlib` - Core scientific libraries
   - RAPIDS packages for GPU-accelerated data science

3. **Download UNI model** (done automatically on first run):
   ```python
   from huggingface_hub import hf_hub_download

   uni_model_path = hf_hub_download(
       repo_id='MahmoodLab/UNI',
       filename='pytorch_model.bin',
       cache_dir='./weights/uni_cache'
   )
   ```

---

## Quick Start

```python
from pathlib import Path
import numpy as np
from honeybee.processors import PathologyProcessor
from huggingface_hub import hf_hub_download

# 1. Initialize processor with UNI model
uni_model_path = hf_hub_download(
    repo_id='MahmoodLab/UNI',
    filename='pytorch_model.bin',
    cache_dir='./weights/uni_cache'
)
processor = PathologyProcessor(model="uni", model_path=uni_model_path)

# 2. Load a WSI slide
wsi_path = Path("path/to/slide.tif")
wsi = processor.load_wsi(wsi_path, tile_size=128, max_patches=500)

# 3. Extract patches
patches = []
for addr in list(wsi.iterateTiles()):
    tile = wsi.getTile(addr, writeToNumpy=True)
    if tile is not None:
        patches.append(tile[:, :, :3])  # RGB only

# 4. Generate embeddings
patches_array = np.stack(patches, axis=0)
embeddings = processor.generate_embeddings(patches_array, batch_size=32)

print(f"Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
# Output: Generated 500 embeddings of dimension 1024
```

---

## Tutorial: Extracting Tissue Patches

### 1. Load a Whole Slide Image

The `PathologyProcessor` class handles loading large WSI files and dividing them into tiles.

```python
from pathlib import Path
from honeybee.processors import PathologyProcessor

# Initialize processor
processor = PathologyProcessor(model="uni", model_path="path/to/uni_model.bin")

# Load WSI with tiling parameters
wsi_path = Path("../hackathon-data/train/images/P0202.tif")
wsi = processor.load_wsi(
    wsi_path,
    tile_size=128,        # Each tile is 128x128 pixels
    max_patches=500,      # Limit number of patches (optional)
    verbose=False
)

# Inspect the loaded slide
print(f"Slide dimensions: {wsi.slide.width}x{wsi.slide.height}")
print(f"Tile size: {wsi.tileSize}x{wsi.tileSize}")
print(f"Tile grid: {wsi.numTilesInX}x{wsi.numTilesInY} = {wsi.numTilesInX * wsi.numTilesInY} tiles")
```

**Output:**
```
Slide dimensions: 6000x7669
Tile size: 128x128
Tile grid: 46x59 = 2714 tiles
```

### 2. Extract Individual Patches

Iterate through the tile grid and extract patch images:

```python
import numpy as np

# Pre-load all tiles
tiles_cache = {}
for addr in list(wsi.iterateTiles()):
    tile = wsi.getTile(addr, writeToNumpy=True)

    # Validate tile
    if tile is not None and len(tile.shape) >= 3 and tile.shape[2] >= 3:
        tile_rgb = tile[:, :, :3]  # Extract RGB channels (drop alpha if present)
        tiles_cache[addr] = tile_rgb

print(f"Extracted {len(tiles_cache)} patches")
print(f"Patch shape: {list(tiles_cache.values())[0].shape}")
```

**Output:**
```
Extracted 2714 patches
Patch shape: (128, 128, 3)
```

**Key Points:**
- `addr` is a tuple `(x, y)` representing tile coordinates in the grid
- `getTile()` returns numpy array with shape `(H, W, C)`
- Some formats (e.g., TIFF) may have 4 channels (RGBA), so extract RGB with `[:, :, :3]`
- Caching tiles improves performance if you need them multiple times

### 3. Tissue Detection (Classical Methods)

Honeybee provides classical computer vision methods for tissue detection:

```python
from PIL import Image

# Get thumbnail of the slide
thumbnail = np.asarray(wsi.slide)
if thumbnail.shape[-1] == 4:
    thumbnail = thumbnail[:, :, :3]

# Method 1: Otsu thresholding
tissue_mask_otsu = processor.detect_tissue(thumbnail, method="otsu")

# Method 2: HSV color-based detection
tissue_mask_hsv = processor.detect_tissue(thumbnail, method="hsv")

# Method 3: Combined approach
tissue_mask_combined = processor.detect_tissue(thumbnail, method="otsu_hsv")

# Calculate tissue percentage
tissue_pct = np.sum(tissue_mask_otsu) / tissue_mask_otsu.size * 100
print(f"Tissue coverage: {tissue_pct:.1f}%")
```

**Available Methods:**
- `"otsu"` - Automatic thresholding based on grayscale histogram
- `"hsv"` - Color-based detection in HSV space (targets pinkish tissue)
- `"otsu_hsv"` - Intersection of both methods for higher precision

### 4. Tissue Detection (Deep Learning)

For more accurate tissue segmentation, use the deep learning tissue detector:

```python
from honeybee.models.TissueDetector.tissue_detector import TissueDetector

# Load pre-trained DenseNet121 tissue detector
tissue_detector = TissueDetector(
    model_path="./weights/deep-tissue-detector_densenet_state-dict.pt"
)

# Attach to WSI and run detection
wsi.tissue_detector = tissue_detector
wsi.detectTissue()

print(f"Analyzed {len(wsi.tileDictionary)} tiles")
```

**Understanding the Output:**

The deep learning detector classifies each tile into 3 categories:

```python
for address, tile_info in wsi.tileDictionary.items():
    artifact_level = tile_info['artifactLevel']    # Probability of artifact (0-1)
    background_level = tile_info['backgroundLevel']  # Probability of background (0-1)
    tissue_level = tile_info['tissueLevel']         # Probability of tissue (0-1)

    print(f"Tile {address}: Tissue={tissue_level:.3f}, Background={background_level:.3f}, Artifact={artifact_level:.3f}")
```

**Example Output:**
```
Tile (0, 0): Tissue=0.023, Background=0.971, Artifact=0.006
Tile (5, 3): Tissue=0.892, Background=0.098, Artifact=0.010
Tile (10, 7): Tissue=0.754, Background=0.231, Artifact=0.015
```

### 5. Filter High-Quality Tissue Patches

Select only patches with high tissue probability:

```python
TISSUE_THRESHOLD = 0.5  # Adjust based on your quality requirements

tissue_patches = []
tissue_coordinates = []
tissue_scores = []

for address, tile_info in wsi.tileDictionary.items():
    if tile_info.get('tissueLevel', 0) > TISSUE_THRESHOLD:
        # Get cached tile
        if address in tiles_cache:
            tissue_patches.append(tiles_cache[address])
            tissue_coordinates.append(address)
            tissue_scores.append(tile_info['tissueLevel'])

print(f"High-quality tissue patches: {len(tissue_patches)}")
print(f"Mean tissue score: {np.mean(tissue_scores):.3f}")
```

**Output:**
```
High-quality tissue patches: 1090
Mean tissue score: 0.742
```

**Threshold Selection Tips:**
- `> 0.3` - Permissive (includes some background)
- `> 0.5` - Balanced (recommended for most use cases)
- `> 0.7` - Strict (only high-confidence tissue)
- `> 0.9` - Very strict (excludes edge regions)

---

## Tutorial: Generating Embeddings

Foundation models like UNI extract rich feature representations from pathology images. These embeddings can be used for downstream tasks like survival prediction, tumor classification, and biomarker discovery.

### 1. Initialize UNI Model

UNI (Universal Pathology Foundation Model) generates 1024-dimensional embeddings:

```python
from honeybee.processors import PathologyProcessor
from huggingface_hub import hf_hub_download

# Download UNI model from HuggingFace (cached after first download)
uni_model_path = hf_hub_download(
    repo_id='MahmoodLab/UNI',
    filename='pytorch_model.bin',
    cache_dir='./weights/uni_cache'
)
print(f"UNI model ready: {uni_model_path}")

# Initialize processor with model path
processor = PathologyProcessor(model="uni", model_path=uni_model_path)
```

**Supported Models:**
- `"uni"` - UNI (1024-dim embeddings) - Requires model_path
- `"virchow2"` - Virchow2 (varies) - Requires model_path
- `"remedis"` - RemedIs (varies) - Requires model_path

### 2. Generate Embeddings (Batch Processing)

Batch processing is much faster than processing patches one-by-one:

```python
import numpy as np

# Stack patches into a single numpy array
patches_array = np.stack(tissue_patches, axis=0)  # Shape: (N, 128, 128, 3)
print(f"Patches array shape: {patches_array.shape}")

# Generate embeddings in batches
embeddings = processor.generate_embeddings(
    patches_array,
    batch_size=32  # Process 32 patches at a time (adjust based on GPU memory)
)

print(f"Embeddings shape: {embeddings.shape}")
print(f"Embedding stats - Mean: {embeddings.mean():.4f}, Std: {embeddings.std():.4f}")
```

**Output:**
```
Patches array shape: (1090, 128, 128, 3)
Embeddings shape: (1090, 1024)
Embedding stats - Mean: 0.0073, Std: 1.2373
```

**Batch Size Guidelines:**
- **GPU 8GB**: batch_size=16-32
- **GPU 16GB**: batch_size=32-64
- **GPU 24GB+**: batch_size=64-128

### 3. Generate Embedding for a Single Patch (Optional)

If you only need to process one or a few patches:

```python
# Single patch
single_patch = tissue_patches[0]  # Shape: (128, 128, 3)

# Add batch dimension
single_patch_batch = np.expand_dims(single_patch, axis=0)  # Shape: (1, 128, 128, 3)

# Generate embedding
embedding = processor.generate_embeddings(single_patch_batch, batch_size=1)
print(f"Single embedding shape: {embedding.shape}")  # (1, 1024)
```

### 4. Understanding Embedding Properties

UNI embeddings are designed to capture:
- **Morphological features** - Cell shapes, tissue architecture
- **Staining patterns** - H&E color distributions
- **Spatial relationships** - Cell density, organization
- **Diagnostic patterns** - Cancer vs. normal, tumor subtypes

```python
# Analyze embedding statistics
print(f"Embedding dimension: {embeddings.shape[1]}")
print(f"Non-zero features: {np.count_nonzero(embeddings[0])}")
print(f"Value range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")

# Check for NaN/inf values
print(f"Contains NaN: {np.any(np.isnan(embeddings))}")
print(f"Contains Inf: {np.any(np.isinf(embeddings))}")
```

**Output:**
```
Embedding dimension: 1024
Non-zero features: 1024
Value range: [-5.234, 6.891]
Contains NaN: False
Contains Inf: False
```

---

## Output Format

### Saving Embeddings to NPZ Files

Save embeddings with metadata for downstream analysis:

```python
import numpy as np

# Prepare metadata
metadata = {
    'patient_id': 'P0202',
    'model': 'uni',
    'num_patches': len(tissue_patches),
    'tile_size': 128,
    'tissue_threshold': 0.5,
    'slide_width': wsi.slide.width,
    'slide_height': wsi.slide.height,
    'embedding_dim': embeddings.shape[1]
}

# Save to compressed NPZ file
output_path = Path("./data/P0202_embeddings.npz")
np.savez_compressed(
    output_path,
    embeddings=embeddings,              # (N, 1024) - Feature vectors
    coordinates=np.array(tissue_coordinates),  # (N, 2) - Tile positions
    tissue_levels=np.array(tissue_scores),     # (N,) - Quality scores
    metadata=np.array(metadata, dtype=object)  # Dict with metadata
)

print(f"Saved: {output_path} ({output_path.stat().st_size / 1024**2:.2f} MB)")
```

### Loading Embeddings

```python
import numpy as np

# Load NPZ file
data = np.load('./data/P0202_embeddings.npz', allow_pickle=True)

# Access individual arrays
embeddings = data['embeddings']        # (1090, 1024)
coordinates = data['coordinates']      # (1090, 2)
tissue_levels = data['tissue_levels']  # (1090,)
metadata = data['metadata'].item()     # dict

# Print info
print(f"Patient: {metadata['patient_id']}")
print(f"Model: {metadata['model']}")
print(f"Patches: {len(embeddings)}")
print(f"Embedding dimension: {embeddings.shape[1]}")

# Verify data integrity
assert embeddings.shape[0] == coordinates.shape[0] == len(tissue_levels)
assert embeddings.shape[1] == 1024
assert np.all(tissue_levels > 0.5)
print("Data integrity verified")
```

**NPZ File Structure:**
```
P0202_embeddings.npz (4.0 MB)
embeddings: (1090, 1024) float32 - UNI feature vectors
coordinates: (1090, 2) int64 - (x, y) tile grid positions
tissue_levels: (1090,) float32 - Tissue probability scores
metadata: dict - Slide and processing metadata
```