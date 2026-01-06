# Rice Leaf Disease Classification – Reproducible Methodology

This repository presents a reproducible deep learning methodology for rice leaf disease classification.  
The work emphasizes dataset integrity, fair model comparison, and confidence-aware inference, and compares a custom convolutional neural network (CNN) with a pretrained EfficientNet model under identical evaluation conditions.

The repository is designed to support academic reporting, methodological transparency, and deployment-oriented analysis.

---

## Methodological Contributions

The main contributions of this work are:

- **Dataset integrity enforcement** through validation and stratified splitting  
- **Duplicate image detection and removal** to prevent data leakage  
- **Fair performance comparison** between a custom CNN and EfficientNet using a shared evaluation protocol  
- **Confidence-aware inference** with optional rejection of uncertain predictions  
- **Analytics-ready outputs** for downstream evaluation and dashboarding  

---

## Pipeline Overview

The overall workflow implemented in this repository follows these stages:

1. Dataset preparation and validation  
2. Duplicate image detection and removal  
3. Stratified train / test / prediction split  
4. Model training (Custom CNN and EfficientNet)  
5. Model evaluation and comparative analysis  
6. Inference with confidence thresholding  
7. Export of prediction results for visualization and analysis  

---

## Repository Structure

```text
rice-leaf-disease-methodology/
├─ configs/
│  └─ class_names.txt
├─ data/
│  ├─ raw/
│  ├─ interim/
│  └─ processed/
├─ artifacts/              # trained models and training histories
├─ outputs/                # inference CSV files
├─ reports/
│  └─ evaluation/          # metrics, confusion matrices, comparisons
└─ src/
   ├─ data/
   │  ├─ make_dataset.py
   │  ├─ deduplicate.py
   │  ├─ split.py
   │  └─ validate.py
   ├─ features/
   │  └─ preprocessing.py
   ├─ models/
   │  ├─ custom_cnn.py
   │  ├─ efficientnet.py
   │  ├─ losses.py
   │  └─ train.py
   ├─ inference/
   │  └─ predict.py
   ├─ evaluation/
   │  ├─ metrics.py
   │  ├─ confusion_matrix.py
   │  ├─ plot_confusion_matrix.py
   │  ├─ compare_models.py
   │  └─ stat_tests.py
   └─ warehouse/
      ├─ schema.sql
      └─ load_predictions.py
