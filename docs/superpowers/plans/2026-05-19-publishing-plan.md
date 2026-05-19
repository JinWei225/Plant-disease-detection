# Plant Disease Detection Publishing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create documentation and publish the plant disease detection model to GitHub and Hugging Face.

**Architecture:** Specialized documentation approach with a GitHub README focused on code usage and evaluation, and a Hugging Face Model Card with YAML metadata for discoverability.

**Tech Stack:** TensorFlow, Keras, EfficientNet, Hugging Face Hub, Git.

---

### Task 1: Generate requirements.txt

**Files:**
- Create: `requirements.txt`

- [ ] **Step 1: Export dependencies from pyproject.toml**

Run: `cat pyproject.toml` to verify dependencies.
Then create `requirements.txt`.

```text
ipykernel>=7.2.0
jupyter>=1.1.1
kagglehub>=1.0.0
matplotlib>=3.10.9
numpy>=2.4.4
scikit-learn>=1.8.0
seaborn>=0.13.2
tensorflow>=2.21.0
pandas
```

- [ ] **Step 2: Commit requirements.txt**

```bash
git add requirements.txt
git commit -m "feat: add requirements.txt for easier installation"
```

### Task 2: Create GitHub README.md

**Files:**
- Create/Modify: `README.md`

- [ ] **Step 1: Write GitHub README.md content**

```markdown
# Plant Disease Detection using EfficientNet

This project implements a deep learning model to detect 38 different types of plant diseases using the PlantVillage dataset. The model is based on the EfficientNet architecture, providing a balance between accuracy and computational efficiency.

## Features
- Detects diseases across various crops (Apple, Corn, Tomato, Potato, etc.).
- Includes 38 distinct classes (diseased and healthy).
- Inference script for single image prediction and full test set evaluation.

## Installation

### Using uv (Recommended)
```bash
uv sync
```

### Using pip
```bash
pip install -r requirements.txt
```

## Usage

### Predict a Single Image
```bash
python inference.py
```
Select option `1` and provide the path to your image.

### Run Full Evaluation
```bash
python inference.py
```
Select option `2` to run evaluation on the test directories.

## Evaluation Results

The model was evaluated on a test set of 511 images.

- **Overall Accuracy:** 41.68%

### Classification Report (Summary)
| Metric | Value |
| :--- | :--- |
| Total Images | 511 |
| Correct Predictions | 213 |
| Accuracy | 41.68% |

### Confusion Matrix
![Confusion Matrix](evaluation_confusion_matrix.png)

## Model Hosting
The trained model is hosted on Hugging Face: [Nefflymicn/PlantVillage-plant-disease-detection](https://huggingface.co/Nefflymicn/PlantVillage-plant-disease-detection)

## License
Apache-2.0
```

- [ ] **Step 2: Commit README.md**

```bash
git add README.md
git commit -m "docs: update README.md with detailed metrics and usage instructions"
```

### Task 3: Create Hugging Face Model Card

**Files:**
- Create: `HF_README.md` (Temporary name, will be uploaded as README.md to HF)

- [ ] **Step 1: Write Hugging Face Model Card content**

```markdown
---
language: en
license: apache-2.0
tags:
- image-classification
- keras
- tensorflow
- plant-disease
- efficientnet
datasets:
- PlantVillage
metrics:
- accuracy
library_name: keras
---

# PlantVillage Plant Disease Detection (EfficientNet)

This model is a Keras implementation of EfficientNet trained to classify 38 types of plant diseases from the PlantVillage dataset.

## Model Description
- **Architecture:** EfficientNet (Keras .keras format)
- **Task:** Image Classification (38 classes)
- **Dataset:** PlantVillage (color images)

## Evaluation Results
The model achieves an overall accuracy of **41.68%** on the provided test set.

### Metrics
- **Accuracy:** 41.68%
- **Total Test Images:** 511

## Intended Use
This model is intended for educational and research purposes in the field of agricultural plant pathology. It can be used to identify potential diseases from leaf images.

## Limitations
- Performance may vary significantly across different plant species and disease types.
- The model was trained on the PlantVillage dataset and may not generalize well to images taken in different environmental conditions or on different leaf backgrounds.

## How to Use

```python
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Load model
model = load_model("plant_disease_efficientnet.keras")

# Preprocess image (EfficientNet expectation)
# ... see inference.py for details
```

## Repository
GitHub: [JinWei225/Plant-disease-detection](https://github.com/JinWei225/Plant-disease-detection)
```

- [ ] **Step 2: Commit HF_README.md**

```bash
git add HF_README.md
git commit -m "docs: add Hugging Face Model Card"
```

### Task 4: Push to GitHub and Hugging Face

- [ ] **Step 1: Push codebase to GitHub**

Run: `git push origin main`

- [ ] **Step 2: Create Hugging Face repository**

Run: `hf_hub_create_repo(name="Nefflymicn/PlantVillage-plant-disease-detection", repo_type="model")`

- [ ] **Step 3: Upload files to Hugging Face**

Upload:
- `plant_disease_efficientnet.keras`
- `class_names.txt`
- `inference.py`
- `HF_README.md` (as `README.md`)

Use `huggingface_hub` tools or CLI.

```bash
# Example CLI commands if using hf tool
hf upload Nefflymicn/PlantVillage-plant-disease-detection plant_disease_efficientnet.keras
hf upload Nefflymicn/PlantVillage-plant-disease-detection class_names.txt
hf upload Nefflymicn/PlantVillage-plant-disease-detection inference.py
hf upload Nefflymicn/PlantVillage-plant-disease-detection HF_README.md --path-in-repo README.md
```

- [ ] **Step 4: Cleanup temporary files**

Run: `rm HF_README.md`
And commit cleanup.
