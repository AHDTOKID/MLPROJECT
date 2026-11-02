"""Generate a synthetic HR dataset matching the proposal schema (§2.5).

Calibrated to reproduce the empirical benchmark distribution and performance
levels detailed in Section 4 of the Project Proposal:
- Total rows: 6,000 records (4,800 train / 1,200 test at 80/20 stratified split)
- Stratified class distribution:
    * 0: Needs Improvement           (15%, support=180 in test)
    * 1: Partially Meets Expectations (40%, support=480 in test)
    * 2: Meets Expectations           (30%, support=360 in test)
    * 3: Exceeds Expectations         (15%, support=180 in test)
"""
from pathlib import Path
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 6000


def generate():
    # Demographics (§2.5)
    age = RNG.integers(22, 61, N)
    gender = RNG.choice(["Male", "Female", "Other"], N, p=[0.55, 0.43, 0.02])
    marital = RNG.choice(["Single", "Married", "Divorced"], N, p=[0.35, 0.55, 0.10])
    distance = RNG.integers(1, 31, N)
    education = RNG.integers(1, 6, N)

    # Job Details (§2.5)
    dept = RNG.choice(
        ["Sales", "Research & Development", "Human Resources"],
        N,
        p=[0.30, 0.55, 0.15],
    )
    role_map = {
        "Sales": ["Sales Executive", "Sales Representative", "Manager"],
        "Research & Development": [
            "Research Scientist",
            "Lab Technician",
            "Developer",
            "Manager",
        ],
        "Human Resources": ["HR Specialist", "HR Manager", "Recruiter"],
    }
    jobrole = np.array([RNG.choice(role_map[d]) for d in dept])

    years_role = np.clip(RNG.integers(0, 16, N), 0, age - 22)
    overtime = RNG.choice(["Yes", "No"], N, p=[0.28, 0.72])
    monthly_income = np.clip(
        RNG.normal(6500, 3000, N) + age * 100, 2000, 25000
    ).round(2)
    total_years = np.clip(RNG.integers(0, 41, N), 0, age - 18)
    years_company = np.clip(RNG.integers(0, total_years + 1, N), 0, 30)
    num_companies = RNG.integers(0, 10, N)
    promotions = RNG.integers(0, 4, N)

    # Engagement & Feedback (§2.5)
    engagement = RNG.integers(1, 6, N)
    wlb = RNG.integers(1, 6, N)
    peer_feedback = RNG.integers(1, 6, N)
    manager_rating = RNG.integers(1, 6, N)

    # Delivery & Performance (§2.5)
    projects = RNG.integers(0, 21, N)
    deadline = np.clip(RNG.normal(0.85, 0.12, N), 0.0, 1.0).round(3)
    absenteeism = np.clip(RNG.poisson(5, N), 0, 30)
    certifications = RNG.integers(0, 11, N)

    # Standardize signals for latent score computation
    def z(x):
        return (x - np.mean(x)) / (np.std(x) + 1e-9)

    # Predictive latent performance signal (matches audited feature weights)
    latent = (
        0.28 * z(engagement)
        + 0.25 * z(deadline)
        + 0.22 * z(projects)
        + 0.18 * z(manager_rating)
        + 0.16 * z(peer_feedback)
        + 0.14 * z(certifications)
        + 0.08 * z(wlb)
        - 0.18 * z(absenteeism)
        - 0.15 * z(overtime == "Yes").astype(float)
        + 0.08 * z(promotions)
        + 0.04 * z(years_company)
        + RNG.normal(0, 0.28, N)
    )

    # Stratified target splits matching §4.2 & §4.3:
    # 0: Needs Improvement (15%)
    # 1: Partially Meets Expectations (40%)
    # 2: Meets Expectations (30%)
    # 3: Exceeds Expectations (15%)
    q = np.quantile(latent, [0.15, 0.55, 0.85])
    target = np.digitize(latent, q)

    df = pd.DataFrame(
        {
            "Age": age,
            "Gender": gender,
            "MaritalStatus": marital,
            "DistanceFromHome": distance,
            "Education": education,
            "Department": dept,
            "JobRole": jobrole,
            "YearsInCurrentRole": years_role,
            "OverTime": overtime,
            "MonthlyIncome": monthly_income,
            "TotalWorkingYears": total_years,
            "YearsAtCompany": years_company,
            "NumCompaniesWorked": num_companies,
            "PromotionsLast5Y": promotions,
            "EngagementScore": engagement,
            "WorkLifeBalance": wlb,
            "PeerFeedbackScore": peer_feedback,
            "ManagerRatingPrev": manager_rating,
            "ProjectsCompleted": projects,
            "DeadlineAdherence": deadline,
            "AbsenteeismDays": absenteeism,
            "Certifications": certifications,
            "PerformanceRating": target,
        }
    )

    out = Path(__file__).resolve().parent / "hr_data.csv"
    df.to_csv(out, index=False)
    print(f"Generated {len(df)} records → {out}")
    print("Class distribution:\n", df["PerformanceRating"].value_counts(normalize=True).round(3))
    return df


if __name__ == "__main__":
    generate()
