# DA5401 – Assignment 4: GMM-Based Synthetic Sampling for Imbalanced Data

**Name:** Bavge Deepak Rajkumar  
**Roll Number:** NA22B031  

---

## Overview

This assignment explores a **probabilistic approach** to addressing extreme class imbalance in the **Credit Card Fraud Detection** dataset.

We use a **Gaussian Mixture Model (GMM)** to learn the distribution of the minority class (fraud), and then generate realistic synthetic samples from this learned distribution.

We build and compare three logistic regression classifiers:
- **Model 1:** Baseline (no rebalancing)
- **Model 2:** GMM-based oversampling of minority class (full augmentation)
- **Model 3:** CBU (clustering-based undersampling of majority) + GMM oversampling to build a compact 50k-point balanced dataset

---

## Dataset Used

- **Source:** [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Shape:** 284,807 rows × 31 columns
- **Features:**
  - `V1` to `V28`: PCA-transformed features
  - `Time`, `Amount`: Raw features
  - `Class`: Target variable (0 = Non-fraud, 1 = Fraud)

> **Imbalance:** Fraud cases are only ~0.17% of total transactions

---

## Assignment Breakdown

### Part A: Baseline Model & Data Analysis

- Loaded dataset, explored feature structure and class imbalance
- Trained **Model 1 (Baseline)** using Logistic Regression on imbalanced training set
- Evaluated on original test set using:
  - Precision, Recall, F1-score (for class = 1)
  - Confusion Matrix and ROC–AUC

### Part B: GMM-Based Resampling

#### Q1: Theory — Why GMM over SMOTE?
- SMOTE assumes linearity and local homogeneity
- GMM models data as a mixture of Gaussians, capturing multimodal clusters
- GMM is **probabilistic, density-aware**, and respects complex fraud patterns

#### Q2: Fitting GMM
- Trained GMM on **scaled training data from fraud class only**
- Selected number of components (k=3) using **BIC**
- Verified convergence and component structure

#### Q3: GMM-Based Oversampling
- Used the fitted GMM to generate synthetic minority samples
- Combined them with original training set to create **Model 2's training data**

#### Q4: CBU + GMM (Compact 50k Training Set)
- Used KMeans (k=4) to **undersample majority** (25k samples) proportionally
- Used GMM to **oversample fraud** up to 25k samples
- Final balanced training set (50k rows) used for **Model 3**

---

## Part C: Evaluation & Analysis

All models were evaluated on the **same original imbalanced test set** for fair comparison.

### Results Summary (Fraud Class)

| Model                  | Precision | Recall | F1-score | ROC–AUC |
|------------------------|-----------|--------|----------|---------|
| Baseline (Imbalanced)  | 0.86      | 0.62   | 0.72     | 0.955   |
| GMM-Augmented          | 0.09      | 0.86   | 0.15     | 0.968   |
| CBU + GMM              | 0.08      | 0.86   | 0.15     | 0.969   |

> GMM-based models **greatly improve recall**, capturing more frauds, but **sacrifice precision** due to increased false positives. ROC–AUC improves slightly, indicating better class separability overall.

---

## How to Run

1. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Open `gmm.ipynb` in any Jupyter environment
3. Run all cells from top to bottom
4. Optional: Inspect or export figures (confusion matrix, BIC plots, ROC curves)

> **Note:** All code, visualizations, and markdown responses are contained in `gmm.ipynb`. No other files are required to run the notebook.

---

## Key Learnings

- GMM allows **statistically grounded oversampling**, unlike naive SMOTE
- BIC is effective for choosing the number of GMM components
- Class imbalance can be tackled not only via oversampling but **structure-aware undersampling** (CBU)
- ROC–AUC is a useful global metric, but **precision-recall trade-off** must be managed

---

## Final Recommendation

**Model 3 (CBU + GMM)** is the best choice for deployment:

- **Recall = 0.86**, matching the best fraud detection sensitivity
- Slightly better **ROC–AUC (0.969)** than other models
- Compact, fixed-size training set (50k rows) allows faster training
- Offers the best **balance between sensitivity and efficiency**

> **Use GMM-based generation with clustering-based undersampling (CBU)** in real-world fraud detection pipelines when the goal is to **maximize detection without overloading downstream systems**.

---
