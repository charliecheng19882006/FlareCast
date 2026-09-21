# FlareCast — Chronic Pain Prediction with Machine Learning

FlareCast is a collaborative machine learning project designed to predict chronic pain outcomes using structured patient and lifestyle data.

The system uses **XGBoost** to model relationships between factors such as sleep, stress, medication usage, exercise, weather conditions, and patient-specific information.

FlareCast supports two machine learning tasks:

- **Classification** — predict whether a patient is likely to experience a pain flare-up
- **Regression** — predict the expected severity of a patient's pain

The project also incorporates **SHAP-based model explainability** to help understand how different features contribute to model predictions.

---

## Features

- Chronic pain flare-up prediction
- Pain severity prediction
- XGBoost classification and regression models
- Patient-level cross-validation
- Model performance evaluation
- SHAP-based model explainability
- Model artifact persistence for downstream inference
- Web interface for interacting with the prediction system

---

## Machine Learning Pipeline

```text
Patient & Lifestyle Data
          |
          v
   Data Preprocessing
          |
          v
    Feature Preparation
          |
          v
       XGBoost
       /      \
      /        \
Classifier    Regressor
    |             |
    v             v
 Flare-up      Pain Level
 Prediction    Prediction
      \           /
       \         /
          v
      Evaluation
          |
          v
    SHAP Analysis
          |
          v
   Model Artifacts
```

---

## Model Selection

We selected **XGBoost** because the project uses structured healthcare data containing multiple patient, lifestyle, and environmental features.

Chronic pain can be influenced by complex and non-linear relationships between these factors, making tree-based gradient boosting models a strong fit for the problem.

XGBoost was selected because it:

- Performs well on structured tabular data
- Captures non-linear relationships and feature interactions
- Works effectively with relatively small datasets
- Provides strong predictive capabilities without requiring deep learning
- Integrates well with SHAP for model explainability

We also considered linear models, Random Forest, and neural networks.

Linear models may not capture complex interactions between chronic pain factors, while neural networks generally benefit from larger datasets and can be more difficult to interpret. Random Forest was also considered as a strong alternative for tabular data.

For this project, XGBoost provided a good balance of predictive performance, robustness, and explainability.

---

## Classification Model

The classification pipeline predicts whether a patient is likely to experience a chronic pain flare-up.

The model is trained using **XGBoost Classifier** and evaluated using multiple classification metrics:

- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
- Confusion Matrix

The training pipeline also generates:

- ROC curves
- Precision-Recall curves
- Confusion matrix visualizations

Using multiple metrics provides a more complete view of classifier performance than relying on accuracy alone.

---

## Regression Model

The regression pipeline predicts the expected severity of a patient's pain.

The model is trained using **XGBoost Regressor** and produces a continuous estimate of pain level.

Regression performance is evaluated using:

- Root Mean Squared Error (RMSE)

This complements the classification model by estimating not only whether a flare-up may occur, but also the expected level of pain.

---

## Patient-Level Cross-Validation

Healthcare datasets may contain multiple observations belonging to the same patient.

A random row-level split could place observations from the same patient in both the training and validation datasets, potentially introducing data leakage and producing overly optimistic evaluation results.

To reduce this risk, the training pipelines use **GroupKFold** with patients as groups.

```text
Training Fold

Patient A ─┐
Patient B ─┼── Training Data
Patient C ─┘


Validation Fold

Patient D ───── Validation Data
```

All observations belonging to a patient remain within the same fold.

This provides a more realistic evaluation of how the model generalizes to patients who were not included in the corresponding training fold.

---

## Model Explainability

FlareCast incorporates **SHAP (SHapley Additive exPlanations)** to improve model interpretability.

SHAP can help explain:

- Which features have the greatest influence on predictions
- How individual features contribute to model output
- The relative importance of patient and lifestyle factors

This is particularly useful for chronic pain prediction because model outputs may depend on interactions among multiple patient, behavioral, and environmental variables.

---

## Technology Stack

### Machine Learning

- Python
- XGBoost
- scikit-learn
- SHAP
- Pandas
- NumPy

### Visualization

- Matplotlib

### Application

- Python prediction pipeline
- Web-based user interface

---

## Model Training

The project contains separate training pipelines for classification and regression:

```text
train_classifier.py
train_regressor.py
```

The overall training workflow is:

```text
Load Dataset
     |
     v
Prepare Features
     |
     v
Create Patient Groups
     |
     v
Patient-Level GroupKFold
     |
     v
Train XGBoost Model
     |
     v
Evaluate Performance
     |
     v
Generate Evaluation Results
     |
     v
SHAP Analysis
     |
     v
Save Model & Feature Artifacts
```

The saved artifacts can then be used by the prediction pipeline for downstream inference.

---

## Project Structure

```text
FlareCast/
├── backend/
│   ├── chronic_pain_training_data.csv
│   ├── config.py
│   ├── predict.py
│   ├── requirements.txt
│   ├── shap_utils.py
│   ├── show_average.py
│   ├── train.py
│   └── update_labels.py
│
├── frontend/
│
├── train_classifier.py
├── train_regressor.py
└── README.md
```

---

## My Contributions

FlareCast was developed collaboratively as a team project.

My primary responsibility was the development of the machine learning training pipelines.

I contributed:

- Developed the **XGBoost classification pipeline** for predicting chronic pain flare-ups
- Developed the **XGBoost regression pipeline** for predicting pain severity
- Implemented patient-level cross-validation using `GroupKFold`
- Evaluated classification performance using Precision, Recall, F1, ROC-AUC, and PR-AUC
- Evaluated regression performance using RMSE
- Generated ROC curves, Precision-Recall curves, and confusion matrix visualizations
- Integrated SHAP-based explainability into the model training workflow
- Saved trained models and feature metadata for downstream inference

The broader FlareCast application, including other backend and frontend components, was developed collaboratively by the project team.

---

## Why Patient-Level Validation Matters

For machine learning applications involving repeated patient observations, evaluation methodology is especially important.

If observations from the same patient appear in both training and validation datasets, the model may indirectly learn patient-specific patterns and produce evaluation results that do not accurately represent performance on unseen patients.

Using patient-level grouping helps reduce this leakage and provides a more meaningful estimate of generalization.

---

## Future Improvements

Potential improvements to the machine learning system include:

- Evaluate additional machine learning algorithms
- Perform systematic hyperparameter optimization
- Expand the training dataset
- Improve feature engineering
- Add model calibration analysis
- Expand patient-level evaluation
- Add automated model tests
- Containerize model inference
- Deploy model serving to the cloud
- Add model monitoring and drift detection

---

## Project Purpose

FlareCast was developed as a collaborative AI project exploring how machine learning can be applied to structured healthcare data for chronic pain prediction.

The project demonstrates an end-to-end machine learning workflow including:

- Structured data modeling
- XGBoost classification and regression
- Patient-aware cross-validation
- Model performance evaluation
- Model explainability with SHAP
- Model artifact generation for downstream inference

My work on the project focused primarily on developing and evaluating the **XGBoost classification and regression training pipelines**.
