# FlareCast — Chronic Pain Prediction

FlareCast is a machine learning application designed to predict chronic pain outcomes using structured patient and lifestyle data.

The project uses **XGBoost** for two prediction tasks:

- **Classification:** Predict whether a patient is likely to experience a pain flare-up.
- **Regression:** Predict the expected severity of the patient's pain.

The models use patient-related features such as sleep, stress, medication usage, exercise, weather conditions, and other contextual factors.

To improve model reliability, patient-level cross-validation is used to reduce data leakage between training and validation sets. SHAP is integrated to explain how individual features contribute to model predictions.
