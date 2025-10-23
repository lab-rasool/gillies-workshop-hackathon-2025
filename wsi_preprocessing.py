from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from honeybee.processors import PathologyProcessor
from honeybee.models.TissueDetector.tissue_detector import TissueDetector
from huggingface_hub import hf_hub_download

# Configuration
WSI_PATH = Path("../hackathon-data/train/images/P0202.tif")
OUTPUT_DIR = Path("tissue_detector_outputs")
DATA_OUTPUT_DIR = Path("./data")
TRAIN_SLIDES = ["P0202.tif", "P0203.tif", "P0207.tif"]
TISSUE_THRESHOLD = 0.5  # Minimum tissue probability to extract embeddings

# Create output directories
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_OUTPUT_DIR.mkdir(exist_ok=True)

# Download UNI model from HuggingFace if not already cached
print("Initializing UNI foundation model...")
uni_model_path = hf_hub_download(
    repo_id='MahmoodLab/UNI',
    filename='pytorch_model.bin',
    cache_dir='./weights/uni_cache'
)
print(f"✓ UNI model ready: {uni_model_path}")

# Initialize processor with UNI model
processor = PathologyProcessor(model="uni", model_path=uni_model_path)

##############################################################################
# PART 1: Load WSI and Tiles
##############################################################################

# Load WSI with smaller tile size for more detailed classification
# Using 128x128 tiles gives us ~170 tiles instead of 9
wsi = processor.load_wsi(WSI_PATH, tile_size=128, max_patches=500, verbose=False)
print(f"✓ WSI loaded: {wsi.slide.width}x{wsi.slide.height}")
print(f"  Tile size: {wsi.tileSize}x{wsi.tileSize}")
print(f"  Tiles: {wsi.numTilesInX}x{wsi.numTilesInY} = {wsi.numTilesInX * wsi.numTilesInY}")

# Pre-load all tiles BEFORE running tissue detection
print("  Pre-loading all tiles...")
tiles_data = []
for addr in list(wsi.iterateTiles()):
    tile = wsi.getTile(addr, writeToNumpy=True)
    if tile is not None and len(tile.shape) >= 3 and tile.shape[2] >= 3:
        tile_rgb = tile[:, :, :3]
        tiles_data.append((addr, tile_rgb))

print(f"✓ Pre-loaded {len(tiles_data)} tiles")

##############################################################################
# PART 2: Run Deep Learning Tissue Detection
##############################################################################
# Load tissue detector
tissue_detector = TissueDetector(model_path="./weights/deep-tissue-detector_densenet_state-dict.pt")
print("✓ Tissue detector loaded")

# Add tissue detector and run detection
wsi.tissue_detector = tissue_detector
wsi.detectTissue()

print(f"✓ Tissue detection completed | Total tiles analyzed: {len(wsi.tileDictionary)}")

##############################################################################
# PART 3: Extract Predictions and Create Heatmaps
##############################################################################
# Collect predictions from all tiles
artifact_levels = []
background_levels = []
tissue_levels = []
tile_positions = []

for address, tile_info in wsi.tileDictionary.items():
    if 'artifactLevel' in tile_info and 'backgroundLevel' in tile_info and 'tissueLevel' in tile_info:
        artifact_levels.append(tile_info['artifactLevel'])
        background_levels.append(tile_info['backgroundLevel'])
        tissue_levels.append(tile_info['tissueLevel'])
        tile_positions.append(address)

artifact_levels = np.array(artifact_levels)
background_levels = np.array(background_levels)
tissue_levels = np.array(tissue_levels)

print(f"  Tiles with predictions: {len(artifact_levels)}")
print(f"  Artifact levels - Mean: {artifact_levels.mean():.3f}, Range: [{artifact_levels.min():.3f}, {artifact_levels.max():.3f}]")
print(f"  Background levels - Mean: {background_levels.mean():.3f}, Range: [{background_levels.min():.3f}, {background_levels.max():.3f}]")
print(f"  Tissue levels - Mean: {tissue_levels.mean():.3f}, Range: [{tissue_levels.min():.3f}, {tissue_levels.max():.3f}]")

# Create probability heatmaps
grid_size_x = wsi.numTilesInX
grid_size_y = wsi.numTilesInY

artifact_map = np.zeros((grid_size_y, grid_size_x))
background_map = np.zeros((grid_size_y, grid_size_x))
tissue_map = np.zeros((grid_size_y, grid_size_x))

for idx, (x, y) in enumerate(tile_positions):
    artifact_map[y, x] = artifact_levels[idx]
    background_map[y, x] = background_levels[idx]
    tissue_map[y, x] = tissue_levels[idx]

##############################################################################
# PART 4: Visualization 1 - Probability Heatmaps
##############################################################################

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Original thumbnail
thumbnail = np.asarray(wsi.slide)
if thumbnail.shape[-1] == 4:
    thumbnail = thumbnail[:, :, :3]

axes[0, 0].imshow(thumbnail)
axes[0, 0].set_title('Original WSI Thumbnail', fontsize=14, fontweight='bold')
axes[0, 0].axis('off')

# Artifact probability
im1 = axes[0, 1].imshow(artifact_map, cmap='Reds', vmin=0, vmax=1)
axes[0, 1].set_title(f'Artifact Probability\n(Mean: {artifact_levels.mean():.3f})', fontsize=12)
axes[0, 1].axis('off')
plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, label='Probability')

# Background probability
im2 = axes[1, 0].imshow(background_map, cmap='Blues', vmin=0, vmax=1)
axes[1, 0].set_title(f'Background Probability\n(Mean: {background_levels.mean():.3f})', fontsize=12)
axes[1, 0].axis('off')
plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, label='Probability')

# Tissue probability
im3 = axes[1, 1].imshow(tissue_map, cmap='Greens', vmin=0, vmax=1)
axes[1, 1].set_title(f'Tissue Probability\n(Mean: {tissue_levels.mean():.3f})', fontsize=12)
axes[1, 1].axis('off')
plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, label='Probability')

plt.suptitle('Deep Learning Tissue Detector - 3-Class Probability Maps\nDenseNet121 Architecture',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_tissue_detector_predictions.png", dpi=200, bbox_inches='tight')
plt.close()

print(f"✓ Saved: {OUTPUT_DIR / '01_tissue_detector_predictions.png'}")

##############################################################################
# PART 5: Visualization 2 - Classical vs Deep Learning Comparison
##############################################################################

# Run classical tissue detection methods
classical_methods = {
    'otsu': processor.detect_tissue(thumbnail, method="otsu"),
    'hsv': processor.detect_tissue(thumbnail, method="hsv"),
    'otsu_hsv': processor.detect_tissue(thumbnail, method="otsu_hsv")
}

fig = plt.figure(figsize=(20, 12))
gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

# Original (top left)
ax = fig.add_subplot(gs[0, 0])
ax.imshow(thumbnail)
ax.set_title('Original WSI', fontsize=14, fontweight='bold')
ax.axis('off')

# Classical methods (row 1)
for idx, (method, mask) in enumerate(classical_methods.items()):
    tissue_pct = np.sum(mask) / mask.size * 100
    ax = fig.add_subplot(gs[0, idx+1])
    ax.imshow(mask, cmap='gray')
    ax.set_title(f'{method.upper()} (Classical)\n{tissue_pct:.1f}% tissue', fontsize=11)
    ax.axis('off')

# Deep learning probability maps (row 2)
ax = fig.add_subplot(gs[1, 0])
ax.axis('off')

ax = fig.add_subplot(gs[1, 1])
ax.imshow(artifact_map, cmap='Reds', vmin=0, vmax=1)
ax.set_title('DL: Artifact', fontsize=11)
ax.axis('off')

ax = fig.add_subplot(gs[1, 2])
ax.imshow(background_map, cmap='Blues', vmin=0, vmax=1)
ax.set_title('DL: Background', fontsize=11)
ax.axis('off')

ax = fig.add_subplot(gs[1, 3])
ax.imshow(tissue_map, cmap='Greens', vmin=0, vmax=1)
ax.set_title('DL: Tissue', fontsize=11)
ax.axis('off')

# Thresholded tissue masks at different confidence levels (row 3)
thresholds = [0.3, 0.5, 0.7, 0.9]
for idx, thresh in enumerate(thresholds):
    tissue_mask_thresh = tissue_map > thresh
    tissue_pct = np.sum(tissue_mask_thresh) / tissue_mask_thresh.size * 100

    ax = fig.add_subplot(gs[2, idx])
    ax.imshow(tissue_mask_thresh, cmap='gray')
    ax.set_title(f'DL: Tissue > {thresh}\n{tissue_pct:.1f}% tissue', fontsize=11)
    ax.axis('off')

plt.suptitle('Classical vs Deep Learning Tissue Detection Methods',
             fontsize=16, fontweight='bold', y=0.98)
plt.savefig(OUTPUT_DIR / "02_tissue_detector_comparison.png", dpi=200, bbox_inches='tight')
plt.close()

print(f"✓ Saved: {OUTPUT_DIR / '02_tissue_detector_comparison.png'}")

##############################################################################
# PART 6: UNI Embedding Extraction for Tissue Patches
##############################################################################

print("\n" + "="*80)
print("PART 6: Extracting UNI Embeddings for Tissue Patches")
print("="*80)

for slide_filename in TRAIN_SLIDES:
    slide_path = Path("../hackathon-data/train/images") / slide_filename
    patient_id = slide_filename.replace(".tif", "")

    print(f"\n{'='*60}")
    print(f"Processing slide: {patient_id}")
    print(f"{'='*60}")

    # Load WSI
    wsi_embed = processor.load_wsi(slide_path, tile_size=128, max_patches=500, verbose=False)
    print(f"  Loaded: {wsi_embed.slide.width}x{wsi_embed.slide.height}")
    print(f"  Tiles: {wsi_embed.numTilesInX}x{wsi_embed.numTilesInY} = {wsi_embed.numTilesInX * wsi_embed.numTilesInY}")

    # Pre-load all tiles and run tissue detection
    print(f"  Pre-loading tiles...")
    tiles_cache = {}
    for addr in list(wsi_embed.iterateTiles()):
        tile = wsi_embed.getTile(addr, writeToNumpy=True)
        if tile is not None and len(tile.shape) >= 3 and tile.shape[2] >= 3:
            tile_rgb = tile[:, :, :3]
            tiles_cache[addr] = tile_rgb

    print(f"  Cached {len(tiles_cache)} tiles")

    # Run tissue detection
    wsi_embed.tissue_detector = tissue_detector
    wsi_embed.detectTissue()
    print(f"  Tissue detection completed: {len(wsi_embed.tileDictionary)} tiles analyzed")

    # Filter tissue patches based on threshold
    tissue_patches = []
    tissue_coordinates = []
    tissue_scores = []

    for address, tile_info in wsi_embed.tileDictionary.items():
        if 'tissueLevel' in tile_info and tile_info['tissueLevel'] > TISSUE_THRESHOLD:
            # Get the cached tile image
            if address in tiles_cache:
                tissue_patches.append(tiles_cache[address])
                tissue_coordinates.append(address)
                tissue_scores.append(tile_info['tissueLevel'])

    print(f"  Tissue patches (>{TISSUE_THRESHOLD} threshold): {len(tissue_patches)}")

    if len(tissue_patches) == 0:
        print(f"  ⚠ No tissue patches found for {patient_id}, skipping...")
        continue

    # Generate UNI embeddings for tissue patches in batch
    print(f"  Generating UNI embeddings (batch processing)...")
    # Stack patches into a single numpy array: (N, H, W, C)
    patches_array = np.stack(tissue_patches, axis=0)
    print(f"  Patches array shape: {patches_array.shape}")

    # Generate embeddings using batch processing (much faster than one-by-one)
    embeddings = processor.generate_embeddings(patches_array, batch_size=32)  # Shape: (N, 1024)
    print(f"  ✓ Generated {len(embeddings)} embeddings")
    coordinates = np.array(tissue_coordinates)  # Shape: (N, 2)
    scores = np.array(tissue_scores)  # Shape: (N,)

    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Embedding stats - Mean: {embeddings.mean():.4f}, Std: {embeddings.std():.4f}")

    # Save to NPZ file
    output_path = DATA_OUTPUT_DIR / f"{patient_id}_embeddings.npz"
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        coordinates=coordinates,
        tissue_levels=scores,
        metadata=np.array({
            'patient_id': patient_id,
            'model': 'uni',
            'num_patches': len(tissue_patches),
            'tile_size': wsi_embed.tileSize,
            'tissue_threshold': TISSUE_THRESHOLD,
            'slide_width': wsi_embed.slide.width,
            'slide_height': wsi_embed.slide.height,
            'embedding_dim': embeddings.shape[1]
        }, dtype=object)
    )

    # Get file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved: {output_path} ({file_size_mb:.2f} MB)")

print("\n" + "="*80)
print("UNI Embedding Extraction Complete!")
print(f"Saved {len(TRAIN_SLIDES)} embedding files to: {DATA_OUTPUT_DIR}")
print("="*80)

