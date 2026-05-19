# Design Doc: Plant Disease Detection Publishing

- **Date:** 2026-05-19
- **Topic:** Publishing to GitHub and Hugging Face with specialized documentation.

## Goals
- Create a comprehensive `README.md` for GitHub with detailed metrics.
- Create a specialized `README.md` (Model Card) for Hugging Face.
- Push the model and relevant files to Hugging Face.
- Push the codebase to GitHub.

## Documentation Structure

### GitHub README.md
- **Title:** Plant Disease Detection using EfficientNet
- **Introduction:** Deep learning model for detecting 38 types of plant diseases using the PlantVillage dataset.
- **Installation:** Instructions using `uv` or `pip`.
- **Usage:** Documentation for `inference.py` (Predict image, Run evaluation).
- **Evaluation:**
  - Accuracy: (Sourced from `evaluation_report.md`)
  - Classification Report: (Sourced from `evaluation_report.md`)
  - Confusion Matrix: Embedding `evaluation_confusion_matrix.png`.
- **Model Hosting:** Link to the Hugging Face model repository.

### Hugging Face Model Card (README.md)
- **YAML Metadata:**
  ```yaml
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
  ```
- **Description:** Keras model based on EfficientNetV2 (or V1 as per `inference.py`) trained on the PlantVillage dataset.
- **Evaluation:** Detailed accuracy and performance metrics.
- **How to Use:** Snippets using `tensorflow` and `keras`.

## Publishing Workflow
1. **GitHub**:
   - Update/Create `README.md`.
   - Git add, commit, and push to `origin main`.
2. **Hugging Face**:
   - Create repo `Nefflymicn/PlantVillage-plant-disease-detection` using `hf` CLI or tool.
   - Upload `plant_disease_efficientnet.keras`, `class_names.txt`, `inference.py`, and the generated Model Card.

## Verification
- Confirm GitHub repository is updated.
- Confirm Hugging Face repository exists and files are uploaded.
- Verify Model Card renders correctly on HF.
