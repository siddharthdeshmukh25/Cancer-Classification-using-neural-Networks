# 🧬 Breast Cancer Classification using Neural Networks

A deep learning classification project that predicts whether a breast tumor is **Malignant (cancerous)** or **Benign (non-cancerous)** using a Multi-Layer Perceptron (Neural Network) built with TensorFlow/Keras.

> 🎓 **Internship Project** — Machine Learning / Deep Learning

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Neural Network Architecture](#-neural-network-architecture)
- [Implementation](#-implementation)
- [Results & Key Findings](#-results--key-findings)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Tech Stack](#-tech-stack)
- [Limitations & Future Scope](#-limitations--future-scope)

---

## 📖 Project Overview

Breast cancer is one of the most common cancers worldwide. Doctors analyze Fine Needle Aspirate (FNA) images of tumor cells and extract 30 measurements (radius, texture, perimeter, area, etc.). Manual diagnosis is slow, expert-dependent, and error-prone.

**Goal:** Build a Neural Network that automatically classifies a tumor as Malignant or Benign with high accuracy — a fast, consistent second-opinion tool for medical professionals.

---

## 📊 Dataset

**Breast Cancer Wisconsin (Diagnostic)** — built into scikit-learn.

| Property | Value |
|---|---|
| Total Records | 569 samples |
| Features | 30 numeric measurements |
| Classes | 2 (Binary Classification) |
| Malignant (0) | 212 samples (37.3%) |
| Benign (1) | 357 samples (62.7%) |
| Missing Values | 0 (clean dataset) |

**Feature Categories:** 10 base measurements (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension) × 3 statistics each (mean, standard error, worst).

---

## 🔬 Methodology

1. **Data Collection** — Load dataset from sklearn into a pandas DataFrame
2. **Data Inspection** — Checked shape (569×31), dtypes, missing values (none found)
3. **Feature/Label Split** — X = 30 features, Y = label (0/1)
4. **Train-Test Split** — 80/20 split (`test_size=0.2`, `random_state=2`) → 455 train / 114 test
5. **Feature Scaling** — `StandardScaler` (fit on train only, transform on test → prevents data leakage)
6. **Model Building** — Compact MLP with Keras `Sequential` API
7. **Compilation** — Adam optimizer + Sparse Categorical Crossentropy loss
8. **Training** — 10 epochs with `validation_split=0.1`
9. **Evaluation** — Test accuracy, confusion matrix, classification report
10. **Predictive System** — End-to-end inference on new patient data

---

## 🏗 Neural Network Architecture

```
Input Layer (30 features)
        ↓
Dense Hidden Layer (20 neurons, ReLU activation)
        ↓
Output Layer (2 neurons, Sigmoid activation)
```

| Layer | Output Shape | Parameters |
|---|---|---|
| Dense (Hidden) | (None, 20) | 620 |
| Dense (Output) | (None, 2) | 42 |
| **Total** | | **662 trainable params** |

**Design choices:**
- **ReLU** in hidden layer → fast convergence, avoids vanishing gradients
- **Sigmoid** in output → gives probability [P(malignant), P(benign)]
- **Sparse categorical crossentropy** → labels are integers (0/1), not one-hot
- **Adam optimizer** → adaptive learning rate, industry default

---

## 💻 Implementation

Core training pipeline (from `Breast_Cancer_Classification_with_Neural_Network.ipynb`):

```python
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

# Scale features (fit on train only!)
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Build the MLP
tf.random.set_seed(3)
model = keras.Sequential([
    keras.layers.Input(shape=(30,)),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(2, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
history = model.fit(X_train_std, Y_train, validation_split=0.1, epochs=10)
```

**Predictive system** converts raw probabilities to labels via `np.argmax([P_malignant, P_benign])`.

---

## 📈 Results & Key Findings

| Metric | Value |
|---|---|
| **Test Accuracy** | **94.7% – 96.5%** (best run: 96.49%) |
| Validation Accuracy | 93.5% – 97.8% |
| Test Loss | 0.12 – 0.15 |
| Misclassifications | 4–6 out of 114 test samples |
| Precision (both classes) | 0.93 – 0.97 |
| Recall (both classes) | 0.93 – 0.97 |

**Sample Confusion Matrix (96.49% run):**

```
                 Predicted
               Malig   Benign
Actual Malig  [  43   |   2  ]
       Benign [   2   |  67  ]
```

**Key findings:**
- ✅ Tiny network (662 params) achieves >95% accuracy — feature scaling + clean data matter more than model size
- ✅ No significant overfitting — training and validation curves track closely
- ✅ Model correctly predicted the notebook's sample patient as **Benign**
- ⚠️ Run-to-run variance (94.7–96.5%) comes from TensorFlow's non-deterministic initialization — averaging over multiple runs gives ~95–96%

---

## 📁 Project Structure

```
Cancer Classification using neural Networks/
│
├── Breast_Cancer_Classification_with_Neural_Network.ipynb   # Main project notebook (executed)
├── Breast_Cancer_NN_Project_Presentation.pptx               # 20-slide presentation
├── run_breast_cancer_nn.py                                  # Headless verification runner script
├── build_ppt.py                                             # PPT generator script
├── accuracy_plot.png                                        # Training accuracy curve
├── loss_plot.png                                            # Training loss curve
├── ppt_architecture.png                                     # Neural network diagram
├── ppt_confusion_matrix.png                                 # Confusion matrix visualization
├── README.md                                                # This documentation
└── requirements.txt                                         # Python dependencies
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Jupyter Notebook
```bash
jupyter notebook Breast_Cancer_Classification_with_Neural_Network.ipynb
```
Run all cells top-to-bottom (`Kernel → Restart & Run All`).

### 3. Or run the quick verification script
```bash
python run_breast_cancer_nn.py
```
This executes the full pipeline (train → evaluate → predict) and prints all metrics, saving plots as PNG.

### 4. Sample prediction
The notebook's final cell demonstrates the predictive system:
```python
input_data = (11.76, 21.6, 74.72, ...)   # 30 tumor measurements
# Output: [[0.145, 0.868]] → [1] → "The tumor is Benign"
```

---

## 🛠 Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.10 | Core language |
| TensorFlow / Keras | 2.12 | Neural network building & training |
| Scikit-learn | 1.4 | Dataset, preprocessing, metrics |
| Pandas | 2.3 | DataFrame handling |
| NumPy | 1.26 | Array operations |
| Matplotlib | 3.10 | Visualizations |
| Jupyter | — | Development environment |

---

## ⚠️ Limitations & Future Scope

**Limitations:**
- Small dataset (569 rows) — deep learning shines on larger data
- Single train/test split — no cross-validation
- No regularization (Dropout / early stopping) yet
- Assistive tool only — cannot replace clinical judgment

**Future improvements:**
- K-Fold cross-validation for stable estimates
- Early stopping + Dropout for regularization
- ROC-AUC, F1-score evaluation
- Comparison with Logistic Regression, SVM, Random Forest
- Model deployment via Flask/Streamlit
- SHAP/LIME explainability

---

## 👤 Author

**Internship Submission** — Machine Learning Project
*Submitted to: Naviotech Solution (info@naviotechsolution.com)*

---
*🤖 Generated with Codebuff*
