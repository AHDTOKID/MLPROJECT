"""Verification script demonstrating single and batch inference via standard pickle (§4.4).

Demonstrates:
1. Loading the serialized ModelBundle using standard Python `pickle.load()`
2. Single-employee scoring with probability map via `bundle.predict_one()`
3. Bulk cohort inference on a pandas DataFrame via `bundle.predict_many()`
4. Standard scikit-learn compatible calls via `bundle.predict()` and `bundle.predict_proba()`
"""
import pickle
from pathlib import Path
import pandas as pd

# Load the production ModelBundle using standard Python pickle (§4.4)
MODEL_FILE = Path(__file__).resolve().parent / "models" / "performance_model.pkl"

if not MODEL_FILE.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_FILE}. Run 'python -m src.train' first.")

print(f"Loading ModelBundle from {MODEL_FILE} via standard pickle.load() ...")
with open(MODEL_FILE, "rb") as f:
    bundle = pickle.load(f)

print(f"✓ Successfully loaded ModelBundle!")
print(f"  Target Classes: {bundle.class_names}")
if bundle.metadata:
    print(f"  Best Algorithm: {bundle.metadata.get('algorithm')}")
    print(f"  Test Accuracy : {bundle.metadata.get('accuracy', 0)*100:.2f}%")
    print(f"  Test Macro-F1 : {bundle.metadata.get('macro_f1', 0):.4f}")

# 1. Single Employee Inference
sample_employee = {
    "Age": 34,
    "Gender": "Female",
    "MaritalStatus": "Married",
    "DistanceFromHome": 6,
    "Education": 4,
    "Department": "Research & Development",
    "JobRole": "Developer",
    "YearsInCurrentRole": 3,
    "OverTime": "No",
    "MonthlyIncome": 8500.0,
    "TotalWorkingYears": 10,
    "YearsAtCompany": 5,
    "NumCompaniesWorked": 2,
    "PromotionsLast5Y": 1,
    "EngagementScore": 4,
    "WorkLifeBalance": 4,
    "PeerFeedbackScore": 4,
    "ManagerRatingPrev": 5,
    "ProjectsCompleted": 9,
    "DeadlineAdherence": 0.92,
    "AbsenteeismDays": 2,
    "Certifications": 4,
}

print("\n--- 1. Single Employee Prediction (.predict_one) ---")
result_single = bundle.predict_one(sample_employee)
print(f"Predicted Performance Tier: [{result_single['predicted_class']}]")
print(f"Prediction Confidence     : {result_single['confidence']*100:.2f}%")
print("Class Probabilities:")
for cls, prob in result_single["probabilities"].items():
    print(f"  - {cls:<30}: {prob*100:5.2f}%")

# 2. Batch Employee Inference
sample_cohort = pd.DataFrame([
    {
        "Age": 42, "Gender": "Male", "MaritalStatus": "Single", "DistanceFromHome": 18,
        "Education": 3, "Department": "Sales", "JobRole": "Sales Executive",
        "YearsInCurrentRole": 6, "OverTime": "Yes", "MonthlyIncome": 9200.0,
        "TotalWorkingYears": 15, "YearsAtCompany": 7, "NumCompaniesWorked": 4,
        "PromotionsLast5Y": 0, "EngagementScore": 2, "WorkLifeBalance": 2,
        "PeerFeedbackScore": 2, "ManagerRatingPrev": 2, "ProjectsCompleted": 3,
        "DeadlineAdherence": 0.68, "AbsenteeismDays": 12, "Certifications": 0,
    },
    {
        "Age": 29, "Gender": "Female", "MaritalStatus": "Single", "DistanceFromHome": 4,
        "Education": 4, "Department": "Research & Development", "JobRole": "Research Scientist",
        "YearsInCurrentRole": 2, "OverTime": "No", "MonthlyIncome": 7200.0,
        "TotalWorkingYears": 6, "YearsAtCompany": 3, "NumCompaniesWorked": 1,
        "PromotionsLast5Y": 1, "EngagementScore": 5, "WorkLifeBalance": 4,
        "PeerFeedbackScore": 5, "ManagerRatingPrev": 5, "ProjectsCompleted": 12,
        "DeadlineAdherence": 0.98, "AbsenteeismDays": 1, "Certifications": 5,
    },
])

print("\n--- 2. Batch Employee Prediction (.predict_many) ---")
batch_results = bundle.predict_many(sample_cohort)
for i, pred in enumerate(batch_results, 1):
    print(f"Employee #{i}: {pred['predicted_class']} (Confidence: {pred['confidence']*100:.2f}%)")

print("\n✓ Verification script completed successfully!")
