# Hackrare2026_RRP_App1

# Overview

This project builds a patient-specific predictive modeling framework to support therapy decisions in RRP.
The system:
Stratifies patient severity and criticality
Predicts probability of response to:
Medical therapy
Surgical therapy
Ranks treatment strategies using rule-based scoring
Visualizes treatment trade-offs using a Response Map

It integrates:
Clinical feature engineering
Gradient boosting models (XGBoost)
Transparent rule-based logic
Interactive Streamlit UI


# Modeling Approach
Algorithm
XGBoost (Gradient Boosted Decision Trees)
OneHotEncoding for categorical features
Numeric features passed through
80/20 train-test split
ROC-AUC evaluation
Trained Models
recurrence_model.pkl
medical_response_model.pkl

surgical_response_model.pkl
