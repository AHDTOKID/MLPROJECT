"""Tests for 3NF Relational Database schema, ORM models, and CRUD operations (§2.6)."""
from datetime import date, datetime
import pytest

from src.database import (
    AuditLog, Department, Employee, MLModel, PerformanceRecord,
    Prediction, SessionLocal, User, init_db, log_audit,
)


@pytest.fixture(autouse=True)
def setup_db():
    init_db(seed=True)


def test_seed_departments_and_users():
    session = SessionLocal()
    try:
        depts = session.query(Department).all()
        assert len(depts) >= 3
        dept_names = [d.dept_name for d in depts]
        assert "Research & Development" in dept_names
        assert "Sales" in dept_names

        users = session.query(User).all()
        assert len(users) >= 3
        usernames = [u.username for u in users]
        assert "admin" in usernames
        assert "hr" in usernames
        assert "lead" in usernames
    finally:
        session.close()


def test_user_password_hashing():
    session = SessionLocal()
    try:
        hr_user = session.query(User).filter_by(username="hr").first()
        assert hr_user is not None
        assert hr_user.check_password("hr123") is True
        assert hr_user.check_password("wrongpassword") is False
    finally:
        session.close()


def test_employee_relationship_and_cascade():
    session = SessionLocal()
    try:
        rd_dept = session.query(Department).filter_by(dept_name="Research & Development").first()
        test_emp = Employee(
            dept_id=rd_dept.dept_id,
            name="Test Engineer",
            age=30,
            gender="Other",
            job_role="Developer",
            monthly_income=7200.0,
            years_at_company=2,
            overtime="No",
            education_level="Bachelor's",
        )
        session.add(test_emp)
        session.commit()

        # Add a performance review
        review = PerformanceRecord(
            emp_id=test_emp.emp_id,
            review_date=date.today(),
            performance_rating=3,
            job_satisfaction=4,
            work_life_balance=4,
        )
        session.add(review)
        session.commit()

        # Verify query and relation
        fetched = session.query(Employee).filter_by(emp_id=test_emp.emp_id).first()
        assert fetched is not None
        assert fetched.department.dept_name == "Research & Development"
        assert len(fetched.performance_records) == 1

        # Cleanup
        session.delete(fetched)
        session.commit()
    finally:
        session.close()


def test_audit_logging():
    session = SessionLocal()
    try:
        count_before = session.query(AuditLog).count()
        log_audit(action="TEST_ACTION", details="Testing audit functionality")
        count_after = session.query(AuditLog).count()
        assert count_after == count_before + 1
    finally:
        session.close()
