"""Normalized 3NF Relational Database schema via SQLAlchemy ORM (§2.6).

Models:
- Department (PK: dept_id)
- User (PK: user_id) [Roles: Admin, HR Manager, Team Lead]
- Employee (PK: emp_id, FK: dept_id)
- PerformanceRecord (PK: record_id, FK: emp_id)
- MLModel (PK: model_id)
- Prediction (PK: pred_id, FKs: emp_id, model_id, user_id)
- AuditLog (PK: log_id, FK: user_id)
"""
from datetime import date, datetime
import json
from typing import Any, Dict, List, Optional
from werkzeug.security import check_password_hash, generate_password_hash
from sqlalchemy import (
    Column, Date, DateTime, Float, ForeignKey, Integer,
    String, Text, create_engine, desc,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, scoped_session

from .config import DATABASE_URI

Base = declarative_base()

# Configure engine and thread-safe scoped session
engine = create_engine(
    DATABASE_URI,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URI else {},
    pool_pre_ping=True,
)
SessionLocal = scoped_session(
    sessionmaker(autocommit=False, autoflush=False, bind=engine, expire_on_commit=False)
)


class Department(Base):
    __tablename__ = "department"

    dept_id = Column(Integer, primary_key=True, autoincrement=True)
    dept_name = Column(String(100), unique=True, nullable=False)
    location = Column(String(100), nullable=False)

    employees = relationship("Employee", back_populates="department", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "dept_id": self.dept_id,
            "dept_name": self.dept_name,
            "location": self.location,
            "employee_count": len(self.employees) if self.employees else 0,
        }


class User(Base):
    __tablename__ = "user"

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False)  # Admin | HR Manager | Team Lead
    created_at = Column(DateTime, default=datetime.utcnow)

    predictions = relationship("Prediction", back_populates="user")
    audit_logs = relationship("AuditLog", back_populates="user")

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        return {
            "user_id": self.user_id,
            "name": self.name,
            "email": self.email,
            "username": self.username,
            "role": self.role,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Employee(Base):
    __tablename__ = "employee"

    emp_id = Column(Integer, primary_key=True, autoincrement=True)
    dept_id = Column(Integer, ForeignKey("department.dept_id"), nullable=False)
    name = Column(String(100), nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(String(20), nullable=False)
    job_role = Column(String(100), nullable=False)
    monthly_income = Column(Float, nullable=False)
    years_at_company = Column(Integer, nullable=False)
    overtime = Column(String(10), nullable=False)
    education_level = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    department = relationship("Department", back_populates="employees")
    performance_records = relationship(
        "PerformanceRecord", back_populates="employee", cascade="all, delete-orphan"
    )
    predictions = relationship(
        "Prediction", back_populates="employee", cascade="all, delete-orphan"
    )

    def to_dict(self):
        return {
            "emp_id": self.emp_id,
            "dept_id": self.dept_id,
            "dept_name": self.department.dept_name if self.department else None,
            "name": self.name,
            "age": self.age,
            "gender": self.gender,
            "job_role": self.job_role,
            "monthly_income": self.monthly_income,
            "years_at_company": self.years_at_company,
            "overtime": self.overtime,
            "education_level": self.education_level,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class PerformanceRecord(Base):
    __tablename__ = "performance_record"

    record_id = Column(Integer, primary_key=True, autoincrement=True)
    emp_id = Column(Integer, ForeignKey("employee.emp_id"), nullable=False)
    review_date = Column(Date, nullable=False, default=date.today)
    performance_rating = Column(Integer, nullable=False)  # 1 to 4
    job_satisfaction = Column(Integer, nullable=False)     # 1 to 5
    work_life_balance = Column(Integer, nullable=False)    # 1 to 5

    employee = relationship("Employee", back_populates="performance_records")

    def to_dict(self):
        return {
            "record_id": self.record_id,
            "emp_id": self.emp_id,
            "employee_name": self.employee.name if self.employee else None,
            "review_date": self.review_date.isoformat() if self.review_date else None,
            "performance_rating": self.performance_rating,
            "job_satisfaction": self.job_satisfaction,
            "work_life_balance": self.work_life_balance,
        }


class MLModel(Base):
    __tablename__ = "ml_model"

    model_id = Column(Integer, primary_key=True, autoincrement=True)
    algorithm = Column(String(100), nullable=False)
    version = Column(String(50), nullable=False)
    accuracy = Column(Float, nullable=False)
    f1_score = Column(Float, nullable=False)
    trained_at = Column(DateTime, default=datetime.utcnow)

    predictions = relationship("Prediction", back_populates="ml_model")

    def to_dict(self):
        return {
            "model_id": self.model_id,
            "algorithm": self.algorithm,
            "version": self.version,
            "accuracy": round(self.accuracy, 4),
            "f1_score": round(self.f1_score, 4),
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
        }


class Prediction(Base):
    __tablename__ = "prediction"

    pred_id = Column(Integer, primary_key=True, autoincrement=True)
    emp_id = Column(Integer, ForeignKey("employee.emp_id"), nullable=True)
    model_id = Column(Integer, ForeignKey("ml_model.model_id"), nullable=True)
    user_id = Column(Integer, ForeignKey("user.user_id"), nullable=True)
    predicted_class = Column(String(50), nullable=False)
    probability = Column(Float, nullable=False)
    features_json = Column(Text, nullable=True)
    shap_json = Column(Text, nullable=True)
    risk_factors = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    employee = relationship("Employee", back_populates="predictions")
    ml_model = relationship("MLModel", back_populates="predictions")
    user = relationship("User", back_populates="predictions")

    def to_dict(self):
        return {
            "pred_id": self.pred_id,
            "emp_id": self.emp_id,
            "employee_name": self.employee.name if self.employee else "Ad-hoc Prediction",
            "model_id": self.model_id,
            "algorithm": self.ml_model.algorithm if self.ml_model else "Logistic Regression",
            "user_id": self.user_id,
            "evaluated_by": self.user.name if self.user else "System",
            "predicted_class": self.predicted_class,
            "probability": round(self.probability, 4),
            "features": json.loads(self.features_json) if self.features_json else {},
            "shap_explanation": json.loads(self.shap_json) if self.shap_json else [],
            "risk_factors": json.loads(self.risk_factors) if self.risk_factors else [],
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class AuditLog(Base):
    __tablename__ = "audit_log"

    log_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("user.user_id"), nullable=True)
    action = Column(String(100), nullable=False)
    details = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="audit_logs")

    def to_dict(self):
        return {
            "log_id": self.log_id,
            "user_id": self.user_id,
            "user_name": self.user.name if self.user else "System",
            "action": self.action,
            "details": self.details,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


def hash_password(password: str) -> str:
    """Secure password hashing using PBKDF2:SHA256 (compatible with macOS / Python 3.9+)."""
    return generate_password_hash(password, method="pbkdf2:sha256")


def init_db(seed: bool = True):
    """Creates tables and populates baseline departments, users, and ML model records."""
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()

    try:
        # 1. Seed Departments if empty
        if session.query(Department).count() == 0:
            depts = [
                Department(dept_name="Sales", location="Building A - North Wing"),
                Department(dept_name="Research & Development", location="Innovation Hub - Floor 3"),
                Department(dept_name="Human Resources", location="Corporate HQ - Floor 2"),
            ]
            session.add_all(depts)
            session.commit()

        # 2. Seed Users (Admin, HR Manager, Team Lead) if empty
        if session.query(User).count() == 0:
            users = [
                User(
                    name="System Administrator",
                    username="admin",
                    email="admin@enterprise.org",
                    password_hash=hash_password("admin123"),
                    role="Admin",
                ),
                User(
                    name="HR Operations Manager",
                    username="hr",
                    email="hr.manager@enterprise.org",
                    password_hash=hash_password("hr123"),
                    role="HR Manager",
                ),
                User(
                    name="Engineering Team Lead",
                    username="lead",
                    email="team.lead@enterprise.org",
                    password_hash=hash_password("lead123"),
                    role="Team Lead",
                ),
            ]
            session.add_all(users)
            session.commit()

        # 3. Seed initial Employees if empty
        if session.query(Employee).count() == 0:
            sales_dept = session.query(Department).filter_by(dept_name="Sales").first()
            rd_dept = session.query(Department).filter_by(dept_name="Research & Development").first()
            hr_dept = session.query(Department).filter_by(dept_name="Human Resources").first()

            employees = [
                Employee(
                    name="Alice Chen",
                    dept_id=rd_dept.dept_id,
                    age=34,
                    gender="Female",
                    job_role="Developer",
                    monthly_income=8500.0,
                    years_at_company=5,
                    overtime="No",
                    education_level="Master's",
                ),
                Employee(
                    name="Bob Miller",
                    dept_id=sales_dept.dept_id,
                    age=42,
                    gender="Male",
                    job_role="Sales Executive",
                    monthly_income=9200.0,
                    years_at_company=7,
                    overtime="Yes",
                    education_level="Bachelor's",
                ),
                Employee(
                    name="Catherine Taylor",
                    dept_id=rd_dept.dept_id,
                    age=29,
                    gender="Female",
                    job_role="Research Scientist",
                    monthly_income=7800.0,
                    years_at_company=3,
                    overtime="No",
                    education_level="Doctorate",
                ),
                Employee(
                    name="David Kumar",
                    dept_id=hr_dept.dept_id,
                    age=38,
                    gender="Male",
                    job_role="HR Specialist",
                    monthly_income=6700.0,
                    years_at_company=4,
                    overtime="No",
                    education_level="Bachelor's",
                ),
            ]
            session.add_all(employees)
            session.commit()

            # Seed Performance Records for baseline employees
            records = [
                PerformanceRecord(
                    emp_id=employees[0].emp_id,
                    review_date=date(2025, 12, 15),
                    performance_rating=3,
                    job_satisfaction=4,
                    work_life_balance=4,
                ),
                PerformanceRecord(
                    emp_id=employees[1].emp_id,
                    review_date=date(2025, 12, 15),
                    performance_rating=1,
                    job_satisfaction=2,
                    work_life_balance=2,
                ),
                PerformanceRecord(
                    emp_id=employees[2].emp_id,
                    review_date=date(2025, 12, 15),
                    performance_rating=4,
                    job_satisfaction=5,
                    work_life_balance=4,
                ),
            ]
            session.add_all(records)
            session.commit()

        # 4. Seed ML Model entry if empty
        if session.query(MLModel).count() == 0:
            init_model = MLModel(
                algorithm="Multinomial Logistic Regression",
                version="v1.0.0-production",
                accuracy=0.6508,
                f1_score=0.6606,
                trained_at=datetime.utcnow(),
            )
            session.add(init_model)
            session.commit()

    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def log_audit(action: str, details: str = "", user_id: Optional[int] = None):
    """Records an operational audit event (§2.3 Module 8)."""
    session = SessionLocal()
    try:
        log = AuditLog(user_id=user_id, action=action, details=details, timestamp=datetime.utcnow())
        session.add(log)
        session.commit()
    except Exception:
        session.rollback()

