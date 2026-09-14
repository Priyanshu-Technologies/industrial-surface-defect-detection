# 🏭 SteelGuard AI

<p align="center">

<img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
<img src="https://img.shields.io/badge/Torchvision-Computer%20Vision-FF6F00?style=for-the-badge&logo=pytorch&logoColor=white" alt="Torchvision">
<img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn">
<img src="https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
<img src="https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">
<img src="https://img.shields.io/badge/Render-Deployed-46E3B7?style=for-the-badge&logo=render&logoColor=white" alt="Render">
<img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">

</p>

<p align="center">

### 🔬 AI-powered visual inspection for hot-rolled steel surfaces

<strong>Classify steel surface defects with deep learning while first checking whether the uploaded image belongs to the expected steel-inspection domain.</strong>

</p>

<p align="center">

<a href="https://steelguard-ai.onrender.com">
<img src="https://img.shields.io/badge/🚀%20LIVE%20DEMO-SteelGuard%20AI-success?style=for-the-badge" alt="Live Demo">
</a>

<a href="https://github.com/Priyanshu-Technologies/industrial-surface-defect-detection">
<img src="https://img.shields.io/badge/📦%20SOURCE-GitHub-black?style=for-the-badge&logo=github" alt="GitHub">
</a>

</p>

---

## 🚀 Live Demo

### 🌐 SteelGuard AI

**https://steelguard-ai.onrender.com**

SteelGuard AI is an end-to-end industrial computer vision application deployed as a Dockerized Streamlit web service on Render.

The application analyzes a suitable steel-surface inspection image and can return:

- 🔍 Predicted defect class
- 📊 Prediction confidence
- 📈 Probability distribution across all supported defect classes
- 🛡️ Steel-domain validation result
- 🎯 Confidence-based decision
- ⚙️ Model and inspection metadata

> ⚠️ **IMPORTANT — INPUT REQUIREMENTS**
>
> SteelGuard AI is designed for **grayscale industrial steel-surface inspection imagery**, especially close-up, scan-like images similar to those represented in the NEU-DET dataset used during development.
>
> **Do not use ordinary mobile-phone photographs, selfies, landscapes, factory-floor photographs, full machinery photographs, or unrelated color images.**
>
> A photograph of a steel object is not necessarily the same thing as an industrial steel-surface inspection image.
>
> The application can accept common image file formats such as JPG, PNG, and BMP, but the **intended model domain is grayscale steel-surface inspection imagery**.

---

## 🏭 About the Project

SteelGuard AI is a practical exploration of **industrial computer vision, transfer learning, domain awareness, and ML deployment**.

The project began as a six-class steel surface defect classifier using the **NEU Surface Defect Database (NEU-DET)**.

During development, an important limitation became clear:

> A model trained specifically on steel inspection images should not blindly classify every image a user uploads.

That led to the development of a second model — a **Steel Domain Gate** — which checks whether the uploaded image belongs to the expected steel/non-steel domain before the six-class defect classifier is allowed to make a prediction.

The final system therefore follows this architecture:

~~~text
                    📤 UPLOADED IMAGE
                           │
                           ▼
                ┌──────────────────────┐
                │   🛡️ STEEL DOMAIN    │
                │        GATE          │
                │   Steel / Non-Steel  │
                └──────────┬───────────┘
                           │
                 ┌─────────┴─────────┐
                 │                   │
              ❌ NON-STEEL         ✅ STEEL
                 │                   │
                 ▼                   ▼
        UNSUPPORTED IMAGE       6-CLASS DEFECT
                                CLASSIFICATION
                                      │
                                      ▼
                              🎯 CONFIDENCE CHECK
                                      │
                           ┌──────────┴──────────┐
                           │                     │
                        ≥ 70%                  < 70%
                           │                     │
                           ▼                     ▼
                  DEFECT CLASSIFIED       MANUAL INSPECTION
~~~

This makes SteelGuard more than a basic image classifier.

The project covers the complete workflow:

~~~text
Industrial Problem
        ↓
Dataset Exploration
        ↓
Image Preprocessing
        ↓
Data Augmentation
        ↓
Transfer Learning
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Domain Classification
        ↓
Threshold Calibration
        ↓
Production Inference
        ↓
Streamlit Web Application
        ↓
Docker Container
        ↓
Cloud Deployment
~~~

---

## 🎯 Problem Statement

Surface defects are an important quality-control concern in manufacturing.

The objective of SteelGuard AI is to explore whether deep learning can automatically classify common steel surface defects from industrial inspection imagery.

### The main question

> **Can a computer vision model learn to identify common hot-rolled steel surface defects while also recognizing when an uploaded image is outside the intended inspection domain?**

---

## 🔬 Supported Defect Classes

SteelGuard AI currently supports six defect classes.

| # | Defect | Description |
|---|---|---|
| 1 | 🌀 **Crazing** | Fine crack-like patterns appearing across the steel surface |
| 2 | ⚫ **Inclusion** | Foreign material or inclusions visible within the surface |
| 3 | 🟫 **Patches** | Irregular patch-like surface abnormalities |
| 4 | 🕳️ **Pitted Surface** | Small pits or localized surface depressions |
| 5 | 🧱 **Rolled-in Scale** | Scale defects embedded during rolling |
| 6 | ➖ **Scratches** | Linear scratches or marks across the surface |

---

## 📸 Input Guidelines

### ✅ Recommended

~~~text
✓ Grayscale steel-surface image
✓ Close-up inspection imagery
✓ Scan-like industrial image
✓ Clearly visible surface texture or defect
✓ Visual appearance similar to NEU-DET imagery
✓ Consistent inspection-style image acquisition
~~~

### ❌ Not Recommended

~~~text
✗ Mobile-phone photographs
✗ Selfies
✗ People
✗ Landscapes
✗ Factory-floor photographs
✗ Full machinery photographs
✗ General steel object photographs
✗ Random internet images
✗ Unrelated color photographs
~~~

### Why?

The six-class classifier learned specific visual features from industrial steel-surface inspection data.

A regular camera image may introduce completely different:

- lighting
- perspective
- scale
- background
- image quality
- surface appearance
- camera characteristics

Therefore, the model should be treated as a **specialized industrial inspection model**, not a general-purpose image classifier.

---

## 📊 Dataset

### NEU Surface Defect Database / NEU-DET

The main defect classification workflow uses:

- **1,800 total images**
- **1,440 training images**
- **360 validation images**
- **6 defect classes**
- Grayscale steel surface imagery

Dataset split:

~~~text
NEU-DET
│
├── Training
│   └── 1,440 images
│
└── Validation
    └── 360 images
~~~

The raw dataset is intentionally excluded from version control.

---

## 🧠 Main Defect Classifier

The main model uses **ResNet18 transfer learning**.

~~~text
Input Image
     ↓
Grayscale Conversion
     ↓
3-Channel Representation
     ↓
Resize → 224 × 224
     ↓
Normalization
     ↓
ResNet18 Backbone
     ↓
Feature Extraction
     ↓
6-Class Fully Connected Layer
     ↓
Class Probabilities
~~~

The final classification layer was adapted to predict the six supported defect categories.

### Why ResNet18?

ResNet18 provides a practical balance between:

- 🧠 Representation power
- ⚡ Training speed
- 💻 Computational requirements
- 📦 Deployment practicality
- 📚 Simplicity and interpretability for an educational project

---

## 🏋️ Main Classifier Training

The main classifier was trained using PyTorch.

Training configuration:

| Setting | Value |
|---|---|
| Architecture | ResNet18 |
| Epochs | 10 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss | CrossEntropyLoss |
| Input Size | 224 × 224 |
| Number of Classes | 6 |

The first training stage freezes the ResNet18 backbone and trains the final classifier layer.

The notebook recorded:

~~~text
Total parameters:     11,179,590
Trainable parameters:     3,078
~~~

---

## 📈 Main Classifier Results

The best recorded validation accuracy was:

# 🎯 **98.33%**

The training history reached:

~~~text
Epoch 1  → 75.83%
Epoch 2  → 96.94%
Epoch 3  → 96.11%
Epoch 4  → 97.50%
Epoch 5  → 97.50%
Epoch 6  → 95.56%
Epoch 7  → 97.22%
Epoch 8  → 98.06%
Epoch 9  → 97.50%
Epoch 10 → 98.33%
~~~

The best model checkpoint is stored as:

~~~text
models/best_model.pth
~~~

---

## 📊 Model Evaluation

The notebook includes several evaluation stages rather than relying only on a single accuracy number.

### Classification Report

The model is evaluated using:

- Precision
- Recall
- F1-score
- Support

### Confusion Matrix

A confusion matrix is used to examine which steel defect classes are most frequently confused with each other.

### Misclassification Analysis

Incorrect predictions are also visualized so that the model's failure cases can be inspected.

### Single-Image Inference

Individual steel inspection images can be passed through the trained model to obtain:

- Predicted class
- Confidence
- Class probabilities

---

## The CIFAR-10 Experiment

This is where the project got more interesting .

The original defect classifier could classify **known steel images**, but that raised another question:

> What happens when a user uploads something that isn't actually a steel inspection image?

Instead of allowing the six-class model to confidently guess a defect anyway, a separate **domain classifier** was created.

For non-steel examples, the project used **CIFAR-10**.

### CIFAR-10 was NOT used to train the defect classifier.

It was used to build negative examples for the **steel/non-steel domain classification task**.

~~~text
NEU-DET
   │
   └── Teaches:
       "What steel defect is this?"

CIFAR-10 + Steel Images
   │
   └── Teaches:
       "Does this image belong to the expected
        steel/non-steel visual domain?"
~~~

The notebook downloaded:

~~~text
CIFAR-10 training images: 50,000
CIFAR-10 validation images: 10,000
~~~

A balanced subset was then selected.

---

## 🛡️ Steel Domain Gate

The Steel Domain Gate is a second ResNet18-based binary classifier.

Its classes are:

~~~text
0 → non_steel
1 → steel
~~~

The prepared domain dataset contains:

| Split | Steel | Non-Steel | Total |
|---|---:|---:|---:|
| Training | 1,440 | 1,440 | 2,880 |
| Validation | 360 | 360 | 720 |

The dataset was intentionally balanced.

---

## 🧠 Domain Model Architecture

The domain model also uses ResNet18 transfer learning.

~~~text
Input Image
      ↓
Grayscale Conversion
      ↓
224 × 224
      ↓
ResNet18
      ↓
Binary Classifier
      ↓
Steel Probability
~~~

Initially, the backbone was frozen and only the final classifier was trained.

The model was then fine-tuned by unfreezing the final ResNet block (`layer4`).

---

## 🏋️ Domain Training

### Stage A — Classifier Head

~~~text
Epoch 1 → 94.44% validation accuracy
Epoch 2 → 97.78%
Epoch 3 → 97.64%
~~~

### Stage B — Fine-Tuning Layer4

~~~text
Fine-tune 1 → 99.86%
Fine-tune 2 → 100.00%
Fine-tune 3 → 100.00%
Fine-tune 4 → 100.00%
Fine-tune 5 → 100.00%
~~~

The best domain model was restored after training.

---

## 📊 Domain Model Evaluation

At the initial probability threshold of 0.5, the prepared validation dataset produced:

~~~text
Validation Accuracy: 100.00%

Non-Steel:
Precision = 1.0000
Recall    = 1.0000
F1        = 1.0000

Steel:
Precision = 1.0000
Recall    = 1.0000
F1        = 1.0000

Confusion Matrix:

[[360   0]
 [  0 360]]
~~~

### Important interpretation

This result is measured on the specific domain validation dataset created in the project.

It does **not** mean that the domain gate can perfectly identify every possible image found in the real world.

The negative examples were based on a balanced subset of CIFAR-10, so real-world domain diversity is much larger.

---

## 🎚️ Domain Threshold Calibration

Rather than relying permanently on a simple 0.5 probability threshold, the project also performed threshold calibration.

The calibrated steel-domain threshold was:

# **0.9551**

The threshold was derived from observed steel probabilities.

At this stricter threshold, the validation results became:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Non-Steel | 0.8696 | 1.0000 | 0.9302 |
| Steel | 1.0000 | 0.8500 | 0.9189 |

Overall calibrated validation accuracy:

# **92.50%**

This demonstrates a classic classification trade-off:

> Increasing the acceptance threshold can make the system stricter about accepting an image as steel, but can also reject some legitimate steel examples.

---

## 🔄 Final Production Inference Pipeline

The production inference code performs the following sequence:

~~~text
1. Receive uploaded image
            ↓
2. Prepare image
            ↓
3. Run Steel Domain Gate
            ↓
      ┌─────┴─────┐
      │           │
   Non-Steel    Steel
      │           │
      ▼           ▼
 Unsupported    Preprocess
   Image            ↓
                ResNet18
             Defect Classifier
                    ↓
              Softmax Output
                    ↓
              Highest Class
                    ↓
            Confidence Check
                    ↓
             ┌──────┴──────┐
             │             │
          ≥ 70%           < 70%
             │             │
             ▼             ▼
       Defect Classified  Manual Inspection
~~~

The production predictor performs the domain validation **before** six-class defect classification.

---

## 🎯 Confidence Threshold

After the image passes the steel-domain gate, the defect classifier produces class probabilities.

The current confidence threshold is:

# **70%**

### If confidence ≥ 70%

```text
DEFECT CLASSIFIED
