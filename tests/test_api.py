"""Tests for Flask REST API endpoints, JWT security, and Role-Based Access Control."""
import io
import json
import pytest

from app.app import app
from src.config import CLASS_NAMES


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def get_token(client, username="hr", password="hr123"):
    res = client.post("/api/login", json={"username": username, "password": password})
    assert res.status_code == 200
    return res.get_json()["access_token"]


def test_login_success_and_failure(client):
    # Success
    res = client.post("/api/login", json={"username": "hr", "password": "hr123"})
    assert res.status_code == 200
    data = res.get_json()
    assert "access_token" in data
    assert data["user"]["role"] == "HR Manager"

    # Invalid password
    res_fail = client.post("/api/login", json={"username": "hr", "password": "wrong"})
    assert res_fail.status_code == 401


def test_auth_me_endpoint(client):
    token = get_token(client, "admin", "admin123")
    res = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["username"] == "admin"
    assert data["role"] == "Admin"


def test_predict_single_endpoint(client):
    token = get_token(client, "hr", "hr123")
    payload = {
        "Age": 34, "Gender": "Female", "MaritalStatus": "Married",
        "DistanceFromHome": 6, "Education": 4, "Department": "Research & Development",
        "JobRole": "Developer", "YearsInCurrentRole": 3, "OverTime": "No",
        "MonthlyIncome": 8500.0, "TotalWorkingYears": 10, "YearsAtCompany": 5,
        "NumCompaniesWorked": 2, "PromotionsLast5Y": 1, "EngagementScore": 4,
        "WorkLifeBalance": 4, "PeerFeedbackScore": 4, "ManagerRatingPrev": 5,
        "ProjectsCompleted": 9, "DeadlineAdherence": 0.92,
        "AbsenteeismDays": 2, "Certifications": 4,
    }
    res = client.post(
        "/api/predict",
        headers={"Authorization": f"Bearer {token}"},
        json=payload,
    )
    assert res.status_code == 200
    data = res.get_json()
    assert "predicted_class" in data
    assert data["predicted_class"] in CLASS_NAMES
    assert "top_contributions" in data
    assert len(data["top_contributions"]) > 0
    assert "strengths" in data
    assert "risks" in data


def test_predict_bulk_endpoint(client):
    token = get_token(client, "lead", "lead123")
    csv_data = (
        "Age,Gender,MaritalStatus,DistanceFromHome,Education,Department,JobRole,"
        "YearsInCurrentRole,OverTime,MonthlyIncome,TotalWorkingYears,YearsAtCompany,"
        "NumCompaniesWorked,PromotionsLast5Y,EngagementScore,WorkLifeBalance,"
        "PeerFeedbackScore,ManagerRatingPrev,ProjectsCompleted,DeadlineAdherence,"
        "AbsenteeismDays,Certifications\n"
        "34,Female,Married,6,4,Research & Development,Developer,3,No,8500,10,5,2,1,4,4,4,5,9,0.92,2,4\n"
        "42,Male,Single,18,3,Sales,Sales Executive,6,Yes,9200,15,7,4,0,2,2,2,2,3,0.68,12,0\n"
    )
    data = {
        "file": (io.BytesIO(csv_data.encode("utf-8")), "test_cohort.csv")
    }
    res = client.post(
        "/api/predict_bulk",
        headers={"Authorization": f"Bearer {token}"},
        data=data,
        content_type="multipart/form-data",
    )
    assert res.status_code == 200
    res_json = res.get_json()
    assert res_json["total_records"] == 2
    assert len(res_json["predictions"]) == 2


def test_rbac_admin_protection(client):
    # HR Manager trying to access Admin Users list should receive 403 Forbidden
    hr_token = get_token(client, "hr", "hr123")
    res = client.get("/api/admin/users", headers={"Authorization": f"Bearer {hr_token}"})
    assert res.status_code == 403

    # Admin should receive 200 OK
    admin_token = get_token(client, "admin", "admin123")
    res_admin = client.get("/api/admin/users", headers={"Authorization": f"Bearer {admin_token}"})
    assert res_admin.status_code == 200


def test_analytics_and_model_info_endpoints(client):
    token = get_token(client, "hr", "hr123")
    res_an = client.get("/api/analytics", headers={"Authorization": f"Bearer {token}"})
    assert res_an.status_code == 200
    an_data = res_an.get_json()
    assert "total_employees" in an_data
    assert "tier_distribution" in an_data

    res_mi = client.get("/api/model_info", headers={"Authorization": f"Bearer {token}"})
    assert res_mi.status_code == 200
    mi_data = res_mi.get_json()
    assert "leaderboard" in mi_data
    assert len(mi_data["leaderboard"]) >= 3


def test_report_exports(client):
    token = get_token(client, "hr", "hr123")

    # CSV Export
    res_csv = client.get("/api/reports/export/csv", headers={"Authorization": f"Bearer {token}"})
    assert res_csv.status_code == 200
    assert "text/csv" in res_csv.content_type

    # PDF Export
    res_pdf = client.get("/api/reports/export/pdf", headers={"Authorization": f"Bearer {token}"})
    assert res_pdf.status_code == 200
    assert "application/pdf" in res_pdf.content_type
    assert len(res_pdf.data) > 1000  # Non-trivial PDF binary
