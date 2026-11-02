# Employee Performance Prediction Platform

> **A machine-learning based web application with explainable insights, helping HR teams make data-driven, defensible, and transparent decisions.**
>
> *Chandigarh University — B.E-CSE Mini Project (Phase 1–2)*  
> *Project Guide: Dr. Vishal Dutt*  
> *Project Team:*  
> - **Tokid Ahmad**: ML & Data Lead  
> - **Somraj Deb**: Backend & Database Lead  
> - **Devi Prasad**: Frontend, Visualization & QA Lead  

---

## 🚀 Quick Start

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate calibrated synthetic dataset (6,000 records)
python data/generate_data.py

# 4. Train, benchmark, and serialize models & diagnostic figures
python -m src.train

# 5. Run standard pickle verification script (§4.4)
python predict_example.py

# 6. Run automated test suite (19 tests)
pytest

# 7. Launch the full-stack web application
python -m app.app
# → Open http://localhost:5003 in your browser
# → Quick Sign-In Personas:
#     * HR Manager: hr / hr123
#     * System Admin: admin / admin123
#     * Team Lead: lead / lead123
```

---

## 🏗️ 4-Layer System Architecture (§2.1, Figure 1)

1. **Presentation Layer (Web UI)**:
   - Interactive responsive dashboard built with modern Tailwind design and Chart.js.
   - Persona-based navigation (Admin, HR Manager, Team Lead).
   - Real-time single-employee scoring with preset loading ("Top Performer", "Burnout Alert", "Stagnant Role").
   - Interactive SHAP waterfall / attribution table and automated operational risk alerts.
   - Bulk cohort CSV drag-and-drop ingestion with template download and roster export.
   - Live Model Zoo leaderboard & high-resolution diagnostic image displays.
   - Executive PDF & CSV analytical report downloads.

2. **Application Layer (Flask REST API)**:
   - JWT authentication (`/api/login`, `/api/auth/me`).
   - Role-Based Access Control (RBAC) enforcing distinct privilege tiers for `Admin`, `HR Manager`, and `Team Lead`.
   - Data Management endpoints for employee CRUD (`/api/employees`) and departments (`/api/departments`).
   - Inference endpoints for single scoring (`/api/predict`) and cohort batches (`/api/predict_bulk`).
   - Report generator exporting publication-ready analytical PDFs (`/api/reports/export/pdf`) and CSV datasets (`/api/reports/export/csv`).
   - Admin routes for user governance (`/api/admin/users`), audit logs (`/api/admin/audit_logs`), and model retraining (`/api/admin/retrain`).

3. **Machine Learning Layer (Python / scikit-learn / SHAP)**:
   - Self-contained scikit-learn pipeline with custom `FunctionTransformer` feature engineering.
   - Outlier clamping and median/mode imputation to prevent training-serving skew or inference crashes.
   - Supervised model zoo: Multinomial Logistic Regression (Winner), Gradient Boosting, and Random Forest.
   - SHAP-based local feature attribution engine and automated detection of operational risks:
     * **Burnout Alert**: Overtime + low work-life balance + high absenteeism.
     * **Career Stagnation Risk**: Prolonged tenure in role without promotions.
     * **Retention / Flight Risk**: Low engagement score + perception gap.
     * **Delivery & Upskilling Strengths**: High deadline adherence, projects completed, and professional certifications.

4. **Data Layer (3NF Relational Database & Artifact Persistence)**:
   - Strictly normalized Third Normal Form (3NF) relational database schema via SQLAlchemy ORM (§2.6, Figure 3):
     * `DEPARTMENT` (`dept_id`, `dept_name`, `location`)
     * `USER` (`user_id`, `name`, `username`, `email`, `password_hash`, `role`)
     * `EMPLOYEE` (`emp_id`, `dept_id`, `name`, `age`, `gender`, `job_role`, `monthly_income`, `years_at_company`, `overtime`, `education_level`)
     * `PERFORMANCE_RECORD` (`record_id`, `emp_id`, `review_date`, `performance_rating`, `job_satisfaction`, `work_life_balance`)
     * `ML_MODEL` (`model_id`, `algorithm`, `version`, `accuracy`, `f1_score`, `trained_at`)
     * `PREDICTION` (`pred_id`, `emp_id`, `model_id`, `user_id`, `predicted_class`, `probability`, `features_json`, `shap_json`, `risk_factors`)
     * `AUDIT_LOG` (`log_id`, `user_id`, `action`, `details`, `timestamp`)
   - Model Registry binaries (`models/performance_model.pkl`, `models/pipeline.pkl`, `models/performance_model.joblib`).

---

## 📊 Benchmark Model Leaderboard (§4.2)

Models trained on an 80/20 stratified split (4,800 train records / 1,200 test records) with 5-fold cross-validation:

| Evaluated Algorithm | 5-Fold CV Macro-F1 | Test Set Accuracy | Test Macro-F1 | Multiclass ROC-AUC (OvR) | Log Loss | Status |
|---|---|---|---|---|---|---|
| **Multinomial Logistic Regression** | **0.6766 ± 0.0114** | **64.33%** | **0.6538** | **0.8847** | **0.7620** | 🏆 **Winner** |
| Gradient Boosting Classifier | 0.6356 ± 0.0163 | 62.42% | 0.6209 | 0.8541 | 0.8285 | Benchmarked |
| Random Forest Classifier | 0.6050 ± 0.0170 | 56.17% | 0.5743 | 0.8342 | 1.0123 | Benchmarked |
| Majority Class Baseline | 0.1429 | 40.00% | 0.1429 | 0.5000 | N/A | Baseline |

### Test Set Classification Report (N=1,200)

```
                              precision    recall  f1-score   support

           Needs Improvement       0.63      0.78      0.70       180
Partially Meets Expectations       0.72      0.59      0.65       480
          Meets Expectations       0.58      0.59      0.58       360
        Exceeds Expectations       0.63      0.76      0.69       180

                    accuracy                           0.64      1200
                   macro avg       0.64      0.68      0.65      1200
                weighted avg       0.65      0.64      0.64      1200
```

---

## 📦 Exported Artifacts (§4.4)

| Artifact | Location | Purpose |
|---|---|---|
| `performance_model.pkl` | `models/` | Production `ModelBundle` supporting standard `pickle.load()` with `.predict()`, `.predict_proba()`, `.predict_one()`, `.predict_many()` |
| `pipeline.pkl` | `models/` | Pure scikit-learn Pipeline object |
| `performance_model.joblib` | `models/` | High-performance joblib serialized model |
| `predict_example.py` | `./` | Verification script demonstrating single and batch inference via standard `pickle` |
| `confusion_matrix.png` | `reports/` | Test set confusion matrix heatmap (Figure 7 Left) |
| `feature_importance.png` | `reports/` | Top 15 permutation feature importances (Figure 7 Right) |
| `classification_metrics.png`| `reports/` | Per-class Precision, Recall, and F1-score chart (Figure 6) |
| `multiclass_roc_curve.png` | `reports/` | Multiclass One-vs-Rest ROC Curves |
| `leaderboard.json` | `reports/` | Machine-readable cross-validation benchmark table |
| `classification_report.json`| `reports/` | Detailed per-class precision, recall, and F1 metrics |

---

## 📡 API Reference

| Method | Endpoint | Auth | Role | Description |
|---|---|---|---|---|
| `POST` | `/api/login` | None | Public | Issues JWT access token with role claims |
| `GET` | `/api/auth/me` | JWT | All | Returns profile of current user |
| `GET` | `/api/departments` | JWT | All | Returns departments and employee counts |
| `GET` | `/api/employees` | JWT | All | List employee records (filter by `dept_id`) |
| `POST` | `/api/employees` | JWT | Admin, HR | Add new employee profile to 3NF database |
| `GET` | `/api/employees/<id>` | JWT | All | Retrieve employee profile + historical reviews |
| `DELETE`| `/api/employees/<id>` | JWT | Admin | Remove employee profile |
| `POST` | `/api/predict` | JWT | All | Predict single employee + SHAP + risk flags + DB persist |
| `POST` | `/api/predict_bulk` | JWT | All | Process cohort CSV upload and return predictions |
| `GET` | `/api/predictions/history`| JWT | All | Query recent predictions stored in DB |
| `GET` | `/api/analytics` | JWT | All | Aggregate KPI stats, risk counts, and distributions |
| `GET` | `/api/reports/export/csv` | JWT | All | Download CSV predictions dataset |
| `GET` | `/api/reports/export/pdf` | JWT | All | Download executive analytical PDF report |
| `GET` | `/api/model_info` | JWT | All | Leaderboard, classes, and diagnostic chart paths |
| `GET` | `/api/admin/users` | JWT | Admin | Manage user accounts |
| `GET` | `/api/admin/audit_logs` | JWT | Admin | View system operational audit trail |
| `POST` | `/api/admin/retrain` | JWT | Admin | Trigger live model retraining & registry update |

---

## 🧪 Automated Testing

Run the automated test suite with pytest:

```bash
pytest -v
```

**Test Suite Coverage (19 Passed):**
- `tests/test_pipeline.py`: Feature engineering derived attributes and predictor class checks.
- `tests/test_model_bundle.py`: Verifies pickle load, `.predict_one()`, and `.predict_many()` conforming to §4.4.
- `tests/test_database.py`: 3NF schema tables, foreign key cascades, PBKDF2 password hashing, and audit logs.
- `tests/test_explainability.py`: Automated detection of burnout risk, stagnation risk, and employee strengths.
- `tests/test_api.py`: JWT login, single scoring, bulk CSV scoring, RBAC protection, analytics, and PDF/CSV downloads.
