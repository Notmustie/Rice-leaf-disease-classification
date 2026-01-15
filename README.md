# Rice Leaf Disease Classification – Reproducible Methodology

This repository presents a reproducible deep learning methodology for rice leaf disease classification.  
The work emphasizes dataset integrity, fair model comparison, and confidence-aware inference, and compares a custom convolutional neural network (CNN) with a pretrained EfficientNet model under identical evaluation conditions.

The repository is designed to support academic reporting, methodological transparency, and deployment-oriented analysis. Rather than focusing only on model implementation, this project formalizes the complete experimental pipeline from dataset acquisition to evaluation and analytics-ready outputs.

---

## Methodological Contributions

The main contributions of this work are:

- Dataset integrity enforcement through validation and stratified splitting  
- Duplicate image detection and removal to prevent data leakage  
- Fair performance comparison between a custom CNN and EfficientNet using a shared evaluation protocol  
- Confidence-aware inference with optional rejection of uncertain predictions  
- Analytics-ready outputs for downstream evaluation, visualization, and reporting  

---

## Reproducibility and Setup

To reproduce the experiments presented in this repository, users must first configure the Python execution environment and then acquire the dataset. The dataset itself is not included in the repository due to size and licensing constraints.

### Python Environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Dataset Acquisition (Kaggle)

This repository does not redistribute the dataset. Users are expected to download the dataset directly from Kaggle under Kaggle’s licensing terms.

### Configure Kaggle API Credentials

1. Go to Kaggle → Account → API → Create New Token  
2. Download `kaggle.json`

Place the file at:

- macOS / Linux: `~/.kaggle/kaggle.json`
- Windows: `C:\Users\<You>\.kaggle\kaggle.json`

Set permissions (macOS / Linux):

```bash
chmod 600 ~/.kaggle/kaggle.json
```

### Install Kaggle CLI

```bash
pip install kaggle
```

### Download the Dataset

```bash
python src/data/download_kaggle.py \
  --dataset "OWNER/DATASET" \
  --out_dir "data/raw/kaggle" \
  --unzip
```

### Standardize into Pipeline Format

```bash
python src/data/standardize.py --kaggle_root "data/raw/kaggle/DATASET_NAME"
```

---
## Explainability (Grad-CAM)

We provide a reproducible Grad-CAM pipeline that generates:
- overlay visualizations (per image)
- heatmaps (per image)
- `xai_summary.csv` with confidence, rejection flag, and attribution metrics (focus score, entropy)
- `xai_agg.csv` for report tables (accepted vs rejected)

### Run
```bash
pip install -r requirements.txt
python scripts/run_xai_gradcam.py
```

## Pipeline Overview

The workflow implemented in this repository consists of the following stages:

1. Dataset preparation and structural validation  
2. Duplicate image detection and removal  
3. Stratified train, test, and prediction split  
4. Model training using:
   - a custom CNN architecture  
   - a pretrained EfficientNet model  
5. Model evaluation and comparative analysis  
6. Inference with confidence thresholding and optional rejection  
7. Export of prediction results for visualization and analytics  

---

## Repository Structure

```text
Rice-leaf-disease-classification/
├─ src/
│  ├─ data/
│  ├─ features/
│  ├─ models/
│  ├─ inference/
│  ├─ evaluation/
│  └─ warehouse/
├─ artifacts/
├─ outputs/
├─ reports/
├─ .gitignore
├─ LICENSE
├─ README.md
└─ requirements.txt
```

---

## Notes

- The dataset is intentionally excluded from version control.
- All scripts are designed to be executed from the repository root.
- The pipeline supports both experimental reproducibility and post-deployment analysis.
- The repository structure and methodology are suitable for academic reporting, thesis work, and extension into production-oriented systems.
