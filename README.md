# Multi-output Machine Learning for ADHD and Sex Prediction using fMRI, Demographics, Socio economic and Behavioral Data

**WiDS Datathon 2025 Project**  
**Team Members:** Pooja Pandit, Mushaer Ahmed, Neha Chaudhary, Partha Vemuri  
**Institution:** University of Arizona

## 🧠 Project Overview

This repository contains the codebase and methodology for our submission to the WiDS Datathon 2025. Our project explores patterns in brain connectivity and behavioral data to predict:
- ADHD Diagnosis (binary classification)
- Sex (Female) (binary classification)

The challenge, developed in collaboration with Cornell University, UC Santa Barbara, the Child Mind Institute, and the Reproducible Brain Charts project, focuses on sex-based differences in neurodevelopment, particularly ADHD, where underdiagnosis in females is a critical issue.

## 🎯 Objectives

- Jointly predict ADHD status and sex using supervised multi-output learning.
- Apply dimensionality reduction (PCA) and feature selection to handle high-dimensional connectome data.
- Evaluate and compare multiple machine learning models for robust and interpretable predictions.
- Explore gender-related diagnostic disparities in ADHD prediction.

## 🧩 Dataset Description

The dataset included:
- **Categorical Metadata:** Demographics (e.g., race, parental occupation)
- **Quantitative Metadata:** Psychological questionnaire scores (e.g., SDQ, APQ)
- **fMRI Connectome Matrices:** 19,901 features representing regional brain connectivity

| Data Split | Participants | Feature Types           |
|------------|--------------|--------------------------|
| Train      | 1,213        | Categorical, Quantitative, fMRI Connectomes |
| Test       | 304          | Same as above            |

## ⚙️ Methodology

### 1. **Preprocessing**
- **Imputation:** Mode for categorical, KNN for quantitative
- **Scaling:** Standardization using `StandardScaler`
- **Data Merging:** Merged all files on `participant_id`

### 2. **Dimensionality Reduction**
- **PCA (Principal Component Analysis):** Applied to high-dimensional fMRI data to reduce noise and prevent overfitting. Retained 98% variance, reduced up to 1,068 components
- **Model-Based Feature Selection:** Random Forest feature importance used to select informative features and retain top predictors

### 3. **Modeling Approach**
- Implemented using `MultiOutputClassifier` from `scikit-learn` to simultaneously predict both ADHD and Sex.
- Models evaluated:
  - **Random Forest (RF)**
  - **Logistic Regression (LR)**
  - **LightGBM (LGBM)**
 
### 4. **Model Phases**
Our development cycle involved:
1. Training on raw data with null removal
2. PCA and feature selection without imputation along with Cross-validation and model tuning
3. Test set transformation and prediction without imputation
4. Training on raw data with imputation
5. PCA and feature selection with imputation along with Cross-validation and model tuning
6. Test set transformation and prediction with imputation
7. Evaluation using accuracy, classification reports, and confusion matrices for each phase.

### 5. **Evaluation**
- Accuracy, Precision, Recall, F1-score
- Confusion matrices analyzed for both ADHD and sex prediction tasks

## 📈 Results Summary

| Approach                         | Model              | ADHD Accuracy | Sex Accuracy | Avg Accuracy |
|----------------------------------|--------------------|----------------|--------------|--------------|
| Model Feature Selection + Impute | Logistic Regression | **0.7964**     | 0.6513       | **0.7238**   |
| PCA + Imputation                | Random Forest       | 0.6851         | 0.6562       | 0.6706       |
| Model Feature Selection          | LightGBM           | 0.7657         | 0.6156       | 0.6906       |

> Logistic Regression emerged as the most effective model for ADHD prediction, while class imbalance significantly affected sex prediction accuracy.

## 💡 Key Insights

- ADHD prediction using integrated fMRI and behavioral features is feasible with strong accuracy (up to 0.80).
- Sex prediction suffers due to class imbalance — future work should explore resampling and fairness-aware algorithms.
- Dimensionality reduction and ensemble models improved efficiency and generalizability.

## 🧪 Reproducibility

The entire project is implemented in **Python** using **Jupyter Notebooks**. Notebooks include:
- `INFO536_preprocess.ipynb`
- `INFO536_train.ipynb`

📂 Repository: [WiDS_Datathon_2025 GitHub Repo](https://github.com/panditpooja/WiDS_Datathon_2025)

## 📚 References

- Attoe, D.E., Climie, E.A. (2023). Miss Diagnosis: A Systematic Review of ADHD in Adult Women.
- Breiman, L. (2001). Random Forests.
- Ke et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree.
- Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python.

## 🙏 Acknowledgments

We thank the organizers of the **WiDS Datathon 2025** and the **Healthy Brain Network** for the data and challenge. This project represents a collaborative first datathon experience for all team members.

---

