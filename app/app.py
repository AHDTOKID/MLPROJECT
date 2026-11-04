"""Flask REST API with JWT Auth, RBAC, CRUD, Predictions, SHAP, and Report Generation.

Conforms to the 4-Layer Architecture and Functional Modules specified in the Proposal:
- Module 1: Authentication & Authorization (Admin, HR Manager, Team Lead)
- Module 2: Data Management (Employee profiles, departments, CSV upload)
- Module 5: Prediction Service (Single & Bulk)
- Module 6: Explainability Engine (SHAP attribution, strengths, burnout/stagnation risks)
- Module 7: Dashboard & Reports (Analytics summary, CSV & PDF export)
- Module 8: System Administration (User management, audit logs, retraining triggers)
"""
from datetime import datetime
from functools import wraps
import io
import json
import os
from pathlib import Path
import sys
from typing import List

from flask import (
    Flask, Response, jsonify, make_response, render_template,
    request, send_file, send_from_directory,
)
from flask_jwt_extended import (
    JWTManager, create_access_token, get_jwt, get_jwt_identity,
    jwt_required,
)
import pandas as pd

# Ensure project root is in Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CLASS_NAMES, DATA_DIR, JWT_SECRET_KEY, MODEL_DIR, REPORT_DIR,
)
from src.database import (
    AuditLog, Department, Employee, MLModel, PerformanceRecord,
    Prediction, SessionLocal, User, hash_password, init_db, log_audit,
)
from src.explain import (
    build_explainer, explain_single, identify_strengths_and_risks,
)
from src.model_bundle import ModelBundle
from src.predict import PerformancePredictor
from src.report_generator import generate_csv_report, generate_pdf_report

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["JWT_SECRET_KEY"] = JWT_SECRET_KEY
jwt = JWTManager(app)

# Initialize database schema and baseline records
init_db(seed=True)

# Load production model & cache SHAP explainer
predictor = PerformancePredictor()
_bg_csv = DATA_DIR / "hr_data.csv"
if _bg_csv.exists():
    _bg = pd.read_csv(_bg_csv).drop(columns=["PerformanceRating"], errors="ignore").sample(min(80, len(pd.read_csv(_bg_csv))), random_state=42)
else:
    from data.generate_data import generate
    _df = generate()
    _bg = _df.drop(columns=["PerformanceRating"]).sample(80, random_state=42)

explainer, feat_names, _ = build_explainer(predictor.pipeline, _bg)


@app.teardown_appcontext
def shutdown_session(exception=None):
    """Scoped session cleanup after request context teardown."""
    SessionLocal.remove()


def roles_required(*allowed_roles):
    """Enforces Role-Based Access Control (RBAC) (§2.3 Module 1)."""
    def decorator(fn):
        @wraps(fn)
        @jwt_required()
        def wrapper(*args, **kwargs):
            claims = get_jwt()
            user_role = claims.get("role")
            if user_role not in allowed_roles:
                return jsonify({
                    "error": "Forbidden",
                    "msg": f"Access denied. Requires one of roles: {', '.join(allowed_roles)}",
                }), 403
            return fn(*args, **kwargs)
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# UI Root
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Module 1: Authentication & Authorization
# ---------------------------------------------------------------------------
@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", "")).strip()

    session = SessionLocal()
    user = session.query(User).filter_by(username=username).first()
    if not user or not user.check_password(password):
        return jsonify({"msg": "Invalid username or password"}), 401

    token = create_access_token(
        identity=str(user.user_id),
        additional_claims={
            "username": user.username,
            "name": user.name,
            "role": user.role,
            "email": user.email,
        },
    )
    user_dict = user.to_dict()
    log_audit(action="USER_LOGIN", details=f"User {user.username} ({user.role}) logged in", user_id=user.user_id)
    return jsonify({
        "access_token": token,
        "user": user_dict,
    })


@app.route("/api/auth/me", methods=["GET"])
@jwt_required()
def get_current_user():
    claims = get_jwt()
    return jsonify({
        "user_id": int(get_jwt_identity()),
        "username": claims.get("username"),
        "name": claims.get("name"),
        "role": claims.get("role"),
        "email": claims.get("email"),
    })


# ---------------------------------------------------------------------------
# Module 2: Data Management (Employees & Departments)
# ---------------------------------------------------------------------------
@app.route("/api/departments", methods=["GET"])
@jwt_required()
def list_departments():
    session = SessionLocal()
    depts = session.query(Department).all()
    return jsonify([d.to_dict() for d in depts])


@app.route("/api/employees", methods=["GET", "POST"])
@jwt_required()
def handle_employees():
    session = SessionLocal()
    if request.method == "GET":
        dept_id = request.args.get("dept_id")
        query = session.query(Employee)
        if dept_id:
            query = query.filter_by(dept_id=int(dept_id))
        employees = query.order_by(Employee.name).all()
        return jsonify([e.to_dict() for e in employees])

    # POST: Create employee
    claims = get_jwt()
    if claims.get("role") not in ["Admin", "HR Manager"]:
        return jsonify({"error": "Only Admin and HR Manager can add employee profiles"}), 403

    data = request.get_json(silent=True) or {}
    emp = Employee(
        dept_id=int(data["dept_id"]),
        name=data["name"],
        age=int(data["age"]),
        gender=data["gender"],
        job_role=data["job_role"],
        monthly_income=float(data["monthly_income"]),
        years_at_company=int(data["years_at_company"]),
        overtime=data["overtime"],
        education_level=data.get("education_level", "Bachelor's"),
    )
    session.add(emp)
    session.commit()
    emp_dict = emp.to_dict()
    log_audit("CREATE_EMPLOYEE", f"Created profile for {emp.name}", int(get_jwt_identity()))
    return jsonify(emp_dict), 201


@app.route("/api/employees/<int:emp_id>", methods=["GET", "DELETE"])
@jwt_required()
def single_employee_ops(emp_id):
    session = SessionLocal()
    emp = session.query(Employee).filter_by(emp_id=emp_id).first()
    if not emp:
        return jsonify({"error": "Employee not found"}), 404

    if request.method == "GET":
        emp_data = emp.to_dict()
        emp_data["history"] = [r.to_dict() for r in emp.performance_records]
        emp_data["predictions"] = [p.to_dict() for p in emp.predictions]
        return jsonify(emp_data)

    # DELETE: Admin only
    claims = get_jwt()
    if claims.get("role") != "Admin":
        return jsonify({"error": "Admin permission required to delete records"}), 403

    session.delete(emp)
    session.commit()
    log_audit("DELETE_EMPLOYEE", f"Deleted employee ID {emp_id}", int(get_jwt_identity()))
    return jsonify({"msg": "Employee deleted successfully"})


# ---------------------------------------------------------------------------
# Module 5 & 6: Prediction Service & Explainability Engine
# ---------------------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
@jwt_required()
def predict_single():
    payload = request.get_json(silent=True) or {}
    emp_id = payload.pop("emp_id", None)
    save_to_db = payload.pop("save_to_db", True)

    try:
        prediction = predictor.predict_one(payload)
        explanation = explain_single(payload, explainer, feat_names, predictor.pipeline)

        result = {
            **prediction,
            "top_contributions": explanation["top_contributions"],
            "strengths": explanation["strengths"],
            "risks": explanation["risks"],
            "features": payload,
        }

        # Persist to PREDICTION table (§2.6)
        if save_to_db:
            session = SessionLocal()
            active_model = session.query(MLModel).order_by(MLModel.model_id.desc()).first()
            user_id = int(get_jwt_identity())
            pred_record = Prediction(
                emp_id=int(emp_id) if emp_id else None,
                model_id=active_model.model_id if active_model else None,
                user_id=user_id,
                predicted_class=result["predicted_class"],
                probability=result["confidence"],
                features_json=json.dumps(payload),
                shap_json=json.dumps(result["top_contributions"]),
                risk_factors=json.dumps(result["risks"]),
            )
            session.add(pred_record)
            session.commit()
            result["pred_id"] = pred_record.pred_id

        log_audit("SINGLE_PREDICTION", f"Predicted {result['predicted_class']} ({result['confidence']*100:.1f}%)", int(get_jwt_identity()))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/predict_bulk", methods=["POST"])
@jwt_required()
def predict_bulk_cohort():
    if "file" not in request.files:
        return jsonify({"error": "Missing CSV file upload under field 'file'"}), 400

    uploaded_file = request.files["file"]
    if not uploaded_file or not uploaded_file.filename:
        return jsonify({"error": "No file selected"}), 400

    if not uploaded_file.filename.lower().endswith(".csv"):
        return jsonify({"error": "File must be in .csv format"}), 400

    try:
        content = uploaded_file.read()
        try:
            df = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(content), encoding="latin1")

        if df.empty:
            return jsonify({"error": "The uploaded CSV file is empty"}), 400

        predictions = predictor.predict_many(df)
        enriched = []

        name_col = None
        for col_candidate in ["Name", "name", "EmployeeName", "employee_name", "FullName", "Candidate"]:
            if col_candidate in df.columns:
                name_col = col_candidate
                break

        for i, pred in enumerate(predictions):
            row_dict = df.iloc[i].to_dict()
            analysis = identify_strengths_and_risks(row_dict)
            emp_name = str(row_dict[name_col]) if name_col and pd.notna(row_dict[name_col]) else f"Cohort #{i+1}"
            dept_name = str(row_dict.get("Department", row_dict.get("department", "Enterprise")))
            job_role = str(row_dict.get("JobRole", row_dict.get("job_role", "N/A")))

            enriched.append({
                "record_index": i + 1,
                "name": emp_name,
                "department": dept_name,
                "job_role": job_role,
                "predicted_class": pred["predicted_class"],
                "confidence": round(pred["confidence"], 4),
                "probabilities": pred["probabilities"],
                "strengths": analysis["strengths"],
                "risks": analysis["risks"],
                "features": {k: (v if pd.notna(v) else None) for k, v in row_dict.items()},
            })

        log_audit("BULK_PREDICTION", f"Processed bulk cohort of {len(enriched)} records", int(get_jwt_identity()))
        return jsonify({
            "total_records": len(enriched),
            "predictions": enriched,
        })
    except Exception as e:
        return jsonify({"error": f"CSV Processing Error: {str(e)}"}), 400


@app.route("/api/predictions/history", methods=["GET"])
@jwt_required()
def get_prediction_history():
    session = SessionLocal()
    limit = min(int(request.args.get("limit", 50)), 200)
    preds = session.query(Prediction).order_by(Prediction.pred_id.desc()).limit(limit).all()
    return jsonify([p.to_dict() for p in preds])


# ---------------------------------------------------------------------------
# Module 7: Dashboard Analytics & Report Generator
# ---------------------------------------------------------------------------
@app.route("/api/analytics", methods=["GET"])
@jwt_required()
def get_analytics():
    session = SessionLocal()
    total_employees = session.query(Employee).count()
    total_predictions = session.query(Prediction).count()

    # Distribution of predictions
    preds = session.query(Prediction).all()
    tier_counts = {name: 0 for name in CLASS_NAMES}
    for p in preds:
        if p.predicted_class in tier_counts:
            tier_counts[p.predicted_class] += 1

    # Department performance breakdown
    depts = session.query(Department).all()
    dept_data = []
    for d in depts:
        dept_data.append({
            "dept_id": d.dept_id,
            "dept_name": d.dept_name,
            "employee_count": len(d.employees),
        })

    # Risk distribution from stored predictions
    risk_counter = {"BURNOUT_RISK": 0, "CAREER_STAGNATION": 0, "ATTRITION_RISK": 0, "WORKLOAD_STRAIN": 0}
    for p in preds:
        if p.risk_factors:
            for r in json.loads(p.risk_factors):
                code = r.get("code")
                if code in risk_counter:
                    risk_counter[code] += 1

    # Leaderboard
    lb_file = REPORT_DIR / "leaderboard.json"
    leaderboard = json.loads(lb_file.read_text()) if lb_file.exists() else []

    return jsonify({
        "total_employees": total_employees,
        "total_predictions": total_predictions,
        "tier_distribution": tier_counts,
        "departments": dept_data,
        "risk_summary": risk_counter,
        "leaderboard": leaderboard,
    })


@app.route("/api/reports/export/csv", methods=["GET"])
@jwt_required()
def export_csv_report():
    session = SessionLocal()
    preds = session.query(Prediction).order_by(Prediction.pred_id.desc()).all()
    data = [p.to_dict() for p in preds]
    csv_content = generate_csv_report(data)
    return Response(
        csv_content,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=performance_predictions_report.csv"},
    )


@app.route("/api/reports/export/pdf", methods=["GET"])
@jwt_required()
def export_pdf_report():
    session = SessionLocal()
    claims = get_jwt()
    preds = session.query(Prediction).order_by(Prediction.pred_id.desc()).all()
    pred_dicts = [p.to_dict() for p in preds]

    # Model summary
    active_model = session.query(MLModel).order_by(MLModel.model_id.desc()).first()
    summary = {
        "model_name": active_model.algorithm if active_model else "Multinomial Logistic Regression",
        "operator_role": claims.get("role", "HR Manager"),
        "macro_f1": active_model.f1_score if active_model else 0.6606,
    }

    pdf_bytes = generate_pdf_report(summary, pred_dicts)
    response = make_response(pdf_bytes)
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = "attachment; filename=Employee_Performance_Analytical_Report.pdf"
    return response


@app.route("/api/reports/diagnostics/<path:filename>")
def serve_diagnostic_image(filename):
    """Serves generated diagnostic charts (Figure 6, Figure 7, ROC curves)."""
    return send_from_directory(REPORT_DIR, filename)


@app.route("/api/model_info", methods=["GET"])
@jwt_required()
def get_model_info():
    lb_file = REPORT_DIR / "leaderboard.json"
    rep_file = REPORT_DIR / "classification_report.json"
    lb = json.loads(lb_file.read_text()) if lb_file.exists() else []
    rep = json.loads(rep_file.read_text()) if rep_file.exists() else {}

    images = {
        "confusion_matrix": "/api/reports/diagnostics/confusion_matrix.png",
        "feature_importance": "/api/reports/diagnostics/feature_importance.png",
        "classification_metrics": "/api/reports/diagnostics/classification_metrics.png",
        "multiclass_roc_curve": "/api/reports/diagnostics/multiclass_roc_curve.png",
    }
    return jsonify({
        "classes": CLASS_NAMES,
        "leaderboard": lb,
        "classification_report": rep,
        "diagnostic_images": images,
    })


# ---------------------------------------------------------------------------
# Module 8: System Administration (Users, Logs, Retraining)
# ---------------------------------------------------------------------------
@app.route("/api/admin/users", methods=["GET", "POST"])
@roles_required("Admin")
def admin_users():
    session = SessionLocal()
    if request.method == "GET":
        users = session.query(User).all()
        return jsonify([u.to_dict() for u in users])

    # POST: Create user
    data = request.get_json(silent=True) or {}
    user = User(
        name=data["name"],
        username=data["username"],
        email=data["email"],
        password_hash=hash_password(data.get("password", "temp123")),
        role=data.get("role", "HR Manager"),
    )
    session.add(user)
    session.commit()
    user_dict = user.to_dict()
    log_audit("ADMIN_CREATE_USER", f"Created user {user.username} with role {user.role}", int(get_jwt_identity()))
    return jsonify(user_dict), 201


@app.route("/api/admin/audit_logs", methods=["GET"])
@roles_required("Admin")
def admin_audit_logs():
    session = SessionLocal()
    logs = session.query(AuditLog).order_by(AuditLog.log_id.desc()).limit(100).all()
    return jsonify([l.to_dict() for l in logs])


@app.route("/api/admin/retrain", methods=["POST"])
@roles_required("Admin")
def admin_trigger_retrain():
    """Triggers model retraining (§2.3 Module 8)."""
    try:
        from src.train import main as train_models
        bundle = train_models()

        # Update MLModel record
        session = SessionLocal()
        new_model = MLModel(
            algorithm=bundle.metadata.get("algorithm", "Multinomial Logistic Regression"),
            version=f"v{datetime.utcnow().strftime('%Y%m%d%H%M')}-retrained",
            accuracy=bundle.metadata.get("accuracy", 0.65),
            f1_score=bundle.metadata.get("macro_f1", 0.66),
            trained_at=datetime.utcnow(),
        )
        session.add(new_model)
        session.commit()

        # Reload in-memory predictor and explainer
        global predictor, explainer, feat_names
        predictor = PerformancePredictor()
        explainer, feat_names, _ = build_explainer(predictor.pipeline, _bg)

        log_audit("RETRAIN_MODEL", f"Retrained model: {bundle.metadata.get('algorithm')}", int(get_jwt_identity()))
        return jsonify({
            "status": "success",
            "msg": "Model retrained and production artifacts updated successfully",
            "metadata": bundle.metadata,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5003))
    app.run(host="0.0.0.0", port=port, debug=True)
