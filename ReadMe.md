**Research Thesis Ontario Tech University** --- Abhinav Sharma 

The details of this project can be found in the 'report' folder

# Hybrid-ConVIRT

Hybrid-ConVIRT is a vision-language model designed for medical image-text representation learning.  
This repository provides the **training methodology, evaluation setup, and implementation details** for the Hybrid-ConVIRT framework.

---

##  Methodology

The training pipeline of Hybrid-ConVIRT can be divided into **three major stages**:

### 1. ConVIRT-Style Training on CheXpert
- Uses **100,000 images** from the CheXpert dataset and their generated captions.
- Trains image and text encoders in a **contrastive learning framework**.
- Produces a **Contrastive Similarity Matrix (CSM)**, which reflects how well embeddings separate positive pairs from negatives.

### 2. Inference through Pre-trained MedCLIP
- Runs the same 100,000 images through a **pre-trained MedCLIP** model.
- Generates a **Predicted Similarity Matrix (PSM)** using MedCLIP’s image and text encoders.

### 3. Fusion & Hybrid-ConVIRT Training
- Fuses the **CSM** and **PSM** into a **Fused Similarity Matrix (FSM)**.
- Trains Hybrid-ConVIRT toward FSM, using:
  - Vision Transformer (ViT) initialized with **ImageNet weights**.
  - Text encoder initialized with **CXR-BERT** (specialized for chest X-rays).
- Incorporates **image transformations** during training.

---

##  Contrastive Similarity Matrix (CSM)

To build the CSM:
- Train image/text encoders in a **paired manner** (image + true caption).
- Each caption describes the diagnosis/findings in the X-ray report.
- Positive pairs are pulled closer, negative pairs pushed apart in the embedding space.

⚠️ Since ConVIRT’s pre-trained weights are not public, we re-train from scratch using the ConVIRT-style approach.

---

## Prompt Engineering & Ensembling

To enhance training, we employ **prompt ensembling**:
- Many chest X-ray datasets provide **labels** instead of full radiology reports.
- Labels are transformed into **contextual prompts** using metadata placeholders (`{age}`, `{sex}`, `{positive classes}`, etc.).
- Four prompt templates are used, generating **4 distinct text samples per image**.

**Example templates:**

1. `This {frontal lateral plane} X-Ray in the {ap pa} orientation reveals {positive classes} in Patient {patient}, a {age} year old {sex}.`
2. `The X-Ray scan taken in the {frontal lateral plane} plane and {ap pa} orientation shows {positive classes} in a {age} year old {sex}.`
3. `Patient {patient}, a {age} year old {sex}, presents with {positive classes} as seen in this {frontal lateral plane} X-Ray in the {ap pa} orientation.`
4. `The {frontal lateral plane} X-Ray image in the {ap pa} orientation displays {positive classes} in a {age} year old {sex}, Patient {patient}.`

---

##  Model Training & Environment

### Libraries & Frameworks
- **Data Handling**: `os`, `numpy`, `pandas`
- **Model Development**: `torch`, `torch.nn`, `torch.optim`
- **Datasets**: `torch.utils.data`
- **Image Processing**: `PIL`
- **Evaluation**: `sklearn.metrics` (accuracy, precision, recall, F1, confusion matrix)
- **Cross-validation**: `StratifiedKFold`
- **Visualization**: `matplotlib`
- **Class Imbalance**: `imblearn` (SMOTE)
- **Pre-trained Models**:
  - `transformers` (CLIPProcessor, CLIPModel)
  - `MedCLIP` (MedCLIPProcessor, MedCLIPModel, MedCLIPVisionModelViT)
  - `BioViL` image encoder from Microsoft’s Health Multimodal repo

### Hardware
- **GPUs**: RTX 2080 Ti & Titan V
- Dataset size: ~100,000 images & captions

---

## 🔧 Hyperparameter Tuning

- **Tool Used**: [Optuna](https://optuna.org/) (Bayesian optimization-based)
- **Why Optuna?** Efficient search compared to grid/random search.

### ConVIRT Training
- Encoder: **CXR-BERT (specialized)**
- Tuned hyperparameters: batch size, learning rate, epochs
- **Best results**:
  - Learning rate: `1.034 × 10⁻⁵`
  - Batch size: `16`
  - Epochs: ~55 (to avoid overfitting)

### Hybrid-ConVIRT Training
- Tuned hyperparameters: learning rate, batch size, temperature
- **Best results**:
  - Learning rate: `1.7 × 10⁻⁵`
  - Temperature: `0.0578`
  - Loss: `0.239`
  - Epochs: ~50 (validation loss plateaued)

---

## Evaluation

### Linear Probing
- A **linear classifier** is trained on frozen Hybrid-ConVIRT image encoder features.
- Purpose: assess **representation quality** and **transferability** without fine-tuning.

### Metrics
- **Macro-Averaging** is used (binary + multi-class):
  - Treats all classes equally, regardless of imbalance.
  - Especially important for medical datasets where minority classes are critical.

**Implemented with scikit-learn:**
```python
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(labels, preds, average='macro', zero_division=0)
recall = recall_score(labels, preds, average='macro', zero_division=0)
f1 = f1_score(labels, preds, average='macro', zero_division=0)

