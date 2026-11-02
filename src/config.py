import os
from pathlib import Path

# Set writable cache dir for matplotlib to suppress warnings in sandbox
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path("/tmp/matplotlib").mkdir(parents=True, exist_ok=True)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
for d in (DATA_DIR, MODEL_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

DATABASE_URI = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR / 'employee_platform.db'}")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "employee-perf-platform-secure-key-2024")

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

TARGET_COL = "PerformanceRating"
CLASS_NAMES = [
    "Needs Improvement",
    "Partially Meets Expectations",
    "Meets Expectations",
    "Exceeds Expectations",
]

NUMERIC_FEATURES = [
    "Age", "DistanceFromHome", "Education", "YearsInCurrentRole",
    "MonthlyIncome", "TotalWorkingYears", "YearsAtCompany",
    "NumCompaniesWorked", "PromotionsLast5Y", "EngagementScore",
    "WorkLifeBalance", "PeerFeedbackScore", "ManagerRatingPrev",
    "ProjectsCompleted", "DeadlineAdherence", "AbsenteeismDays",
    "Certifications",
]

CATEGORICAL_FEATURES = [
    "Gender", "MaritalStatus", "Department", "JobRole", "OverTime",
]

ENGINEERED_FEATURES = [
    "TenureRatio", "PromotionRate", "TrainingPerYear",
    "EngagementIndex", "ManagerPeerGap",
]

ALL_NUMERIC = NUMERIC_FEATURES + ENGINEERED_FEATURES
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
