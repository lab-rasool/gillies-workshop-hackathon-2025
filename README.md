# Dr. Robert Gillies Machine Learning Workshop - Hackathon 2025

Welcome to the 2025 Cancer Machine Learning Hackathon at Moffitt Cancer Center!

## Quick Start with Brev

If you're using the Brev GPU instance, the environment is pre-configured. Simply run:

```bash
# Activate the hackathon environment
source ~/activate_hackathon.sh

# Start JupyterLab (if not auto-started)
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

Your datasets are available at: `~/hackathon-data/`

## Manual Setup (Non-Brev Users)

If you're setting up on your own GPU instance:

```bash
# Clone the repository
git clone https://github.com/lab-rasool/gillies-workshop-hackathon-2025.git
cd gillies-workshop-hackathon-2025

# Run the setup script
chmod +x setup.sh
./setup.sh
```

The setup script will:
- Install Miniconda (if needed)
- Install uv package manager
- Create a conda environment with Python 3.10
- Install RAPIDS packages (cuDF, cuML, cuGraph, etc.)
- Install honeybee-ml and JupyterLab
- Download training and test datasets from HuggingFace

## Competition Overview

This hackathon features **5 competition tracks** focused on survival prediction using pathology imaging and clinical data:

### Competition Tracks

1. **Medical Imaging AI**
   - Focus: Image classification and segmentation
   - Metrics: AUROC (30%), Sensitivity (30%), Specificity (20%), Dice (20%)

2. **Digital Pathology**
   - Focus: Pathology slide analysis
   - Metrics: F1 (35%), Accuracy (25%), Precision (20%), Recall (20%)

3. **NLP Clinical Reports**
   - Focus: Clinical text analysis
   - Metrics: Entity F1 (35%), BLEU (25%), ROUGE (25%), Accuracy (15%)

4. **Genomics & Bioinformatics**
   - Focus: Multi-omics data analysis
   - Metrics: Accuracy (30%), MCC (30%), AUROC (25%), F1 Weighted (15%)

5. **Multimodal Integration**
   - Focus: Combining multiple data modalities
   - Metrics: C-Index (40%), AUROC (30%), Accuracy (20%), Calibration (10%)

### Prizes

- **1st Place**: $5,000
- **2nd Place**: $3,000
- **3rd Place**: $1,500
- **Track Winners**: $500 each

## Dataset Information

The competition uses the **Moffitt Cancer Center Pathology Dataset** with:

- **Training Set**: 600 patients with complete clinical outcomes
- **Test Set**: 265 patients (outcomes withheld for evaluation)

### Data Location

After setup, datasets are available at:
```
~/hackathon-data/
├── train/
│   ├── clinical_with_pathology.csv  # 600 patients, 30 columns
│   └── images/                      # 600 whole-slide pathology images (.tif)
└── test/
    ├── test.csv                     # 265 patients, 21 columns (no outcomes)
    └── images/                      # 265 whole-slide pathology images (.tif)
```

### Dataset Features

**Clinical Data (30 columns in training, 21 in test)**:
- Demographics (age, gender, race, ethnicity)
- Tumor characteristics (stage, grade, subtype, histology)
- Treatment information (surgery, chemotherapy, radiation)
- Pathology reports (tumor size, margins, lymph nodes)

**Pathology Images**:
- Format: Whole-slide images (.tif)
- Average size: ~800MB per image
- Resolution: High-resolution digital pathology slides

**Survival Outcomes (training only)**:
- `vital_status`: Alive or deceased
- `overall_survival_days`: Time to death or last follow-up
- `overall_survival_event`: Binary survival outcome
- `days_to_death`, `days_to_last_followup`
- `progression_or_recurrence`, `days_to_progression`, `days_to_recurrence`

## Submission Guidelines

### Submission Format

Each track has a specific CSV format requirement:

**Medical Imaging AI**:
```csv
patient_id,prediction,probability
```

**Digital Pathology**:
```csv
slide_id,x,y,class,probability
```

**NLP Clinical Reports**:
```csv
report_id,entity,start,end,label
```

**Genomics & Bioinformatics**:
```csv
sample_id,subtype,confidence
```

**Multimodal Integration**:
```csv
patient_id,outcome,risk_score,confidence
```

### Submission Rules

- **Daily Limit**: 5 submissions per day per track
- **Team Size**: Maximum 5 members per team
- **File Size**: Maximum 50MB per submission
- **Leaderboard**: Auto-refreshes every 30 seconds

### How to Submit

1. Visit the leaderboard: [TBD - Leaderboard URL]
2. Register your team
3. Select your competition track
4. Upload your predictions CSV file
5. View your score and ranking in real-time

## Environment Details

### Python Packages

The environment includes:

**RAPIDS Suite (GPU-accelerated)**:
- cuDF: GPU DataFrames
- cuML: GPU Machine Learning
- cuGraph: GPU Graph Analytics
- cuCIM: GPU Image Processing
- Dask-cuDF: Distributed GPU computing

**Core Libraries**:
- honeybee-ml: Medical imaging ML toolkit
- JupyterLab: Interactive notebooks
- pandas, numpy: Data manipulation
- scikit-learn: ML algorithms
- matplotlib, seaborn, plotly: Visualization

### GPU Requirements

**Recommended Minimum**:
- GPU: NVIDIA L40S (48GB VRAM)
- RAM: 32GB system memory
- Storage: 100GB available (datasets are large)
- CUDA: 13.0 or higher

### Verify Your Setup

Run the verification script to check all packages:
```bash
python ~/verify_setup.py
```

Expected output:
```
✓ cuDF is installed
✓ Dask-cuDF is installed
✓ cuML is installed
✓ cuGraph is installed
✓ cuCIM is installed
✓ honeybee-ml is installed
✓ JupyterLab is installed
...
```

## Getting Started with Development

### Load Training Data

```python
import pandas as pd
import cudf

# Load clinical data
train_df = cudf.read_csv('~/hackathon-data/train/clinical_with_pathology.csv')

# View first few rows
print(train_df.head())

# Check survival outcomes
print(train_df[['vital_status', 'overall_survival_days', 'overall_survival_event']].describe())
```

### Load Pathology Images

```python
from cucim import CuImage
import os

# Load a single whole-slide image
image_path = os.path.expanduser('~/hackathon-data/train/images/patient_001.tif')
slide = CuImage(image_path)

print(f"Image shape: {slide.shape}")
print(f"Image resolution: {slide.resolutions}")

# Extract a tile from the slide
tile = slide.read_region((0, 0), (512, 512), level=0)
```

### Example: Simple Survival Model

```python
import cudf
import cuml
from cuml.ensemble import RandomForestClassifier

# Load training data
train_df = cudf.read_csv('~/hackathon-data/train/clinical_with_pathology.csv')

# Select features (example)
feature_cols = ['age_at_diagnosis', 'tumor_stage_numeric', 'tumor_grade_numeric']
X_train = train_df[feature_cols]
y_train = train_df['overall_survival_event']

# Train a model
model = RandomForestClassifier(n_estimators=100, max_depth=10)
model.fit(X_train, y_train)

# Load test data
test_df = cudf.read_csv('~/hackathon-data/test/test.csv')
X_test = test_df[feature_cols]

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Create submission file
submission = cudf.DataFrame({
    'patient_id': test_df['patient_id'],
    'outcome': predictions,
    'risk_score': probabilities[:, 1],
    'confidence': probabilities[:, 1]
})

submission.to_csv('submission.csv', index=False)
```

## Tips for Success

### Leverage GPU Acceleration

Use RAPIDS libraries for significant speedups:
```python
# CPU (slow)
import pandas as pd
df = pd.read_csv('large_file.csv')

# GPU (fast)
import cudf
df = cudf.read_csv('large_file.csv')
```

### Feature Engineering

- Extract features from pathology images (texture, color, morphology)
- Combine clinical and imaging features
- Engineer time-based features (survival duration, time to progression)

### Model Selection

Consider ensemble approaches:
- Gradient Boosting (XGBoost, LightGBM, cuML GBM)
- Random Forests (cuML RF)
- Deep Learning (CNNs for images, transformers for text)
- Survival analysis (Cox proportional hazards, DeepSurv)

### Cross-Validation

Use stratified K-fold for survival data:
```python
from cuml.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

## Resources

- **HuggingFace Dataset**: [Lab-Rasool/hackathon-2025](https://huggingface.co/datasets/Lab-Rasool/hackathon-2025)
- **RAPIDS Documentation**: [docs.rapids.ai](https://docs.rapids.ai)
- **honeybee-ml Documentation**: [pypi.org/project/honeybee-ml](https://pypi.org/project/honeybee-ml/)
- **Competition Leaderboard**: [TBD]
- **Workshop Website**: [Moffitt Cancer Center ML Workshop](https://moffitt.org/)

## Troubleshooting

### CUDA Out of Memory

If you run out of GPU memory:
```python
# Clear GPU memory
import cudf
cudf.utils.clear_device_memory()

# Or use Dask for larger-than-memory processing
import dask_cudf
ddf = dask_cudf.read_csv('large_file.csv')
```

### Slow Data Loading

Use batch processing for large images:
```python
# Process images in batches
batch_size = 10
for i in range(0, len(image_paths), batch_size):
    batch = image_paths[i:i+batch_size]
    process_batch(batch)
```

### Package Import Errors

If you encounter import errors:
```bash
# Reactivate the environment
conda activate hackathon-2025

# Verify installation
python ~/verify_setup.py

# Reinstall if needed
pip install --force-reinstall honeybee-ml
```

## Support

For technical issues or questions:
- Create an issue in this repository
- Contact workshop organizers at [TBD - Contact Email]
- Visit the help desk during the workshop

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

- **Moffitt Cancer Center** for providing the dataset
- **NVIDIA RAPIDS** for GPU-accelerated computing tools
- **Brev** for providing GPU infrastructure

---

Good luck and happy coding! May the best model win!
