"""Analytical Report Generator for PDF and CSV exports (§2.1, §2.3 Module 7).

Generates:
1. Formatted CSV datasets for downstream spreadsheet / BI integration.
2. Executive PDF reports using FPDF2 with summary metrics, model diagnostics,
   risk assessments, and employee cohort tables.
"""
from datetime import datetime
import io
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from fpdf import FPDF

from .config import REPORT_DIR


class PlatformPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(30, 58, 138)  # Deep blue #1e3a8a
        self.cell(0, 8, "Employee Performance Prediction Platform", align="L", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(100, 116, 139)  # Slate
        self.cell(0, 5, "Executive Analytical Report & Machine Learning Diagnostic Insights", align="L", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(37, 99, 235)  # Blue line
        self.set_line_width(0.6)
        self.line(self.get_x(), self.get_y() + 2, self.get_x() + 190, self.get_y() + 2)
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, f"Page {self.page_no()} | Employee Performance Platform - Confidential Human Capital Analytics", align="C")


def generate_csv_report(predictions: List[Dict[str, Any]]) -> str:
    """Converts a list of prediction records to a CSV string."""
    rows = []
    for p in predictions:
        row = {
            "Prediction_ID": p.get("pred_id", ""),
            "Employee_Name": p.get("employee_name", "Ad-hoc"),
            "Predicted_Tier": p.get("predicted_class", ""),
            "Confidence": f"{p.get('probability', 0)*100:.1f}%",
            "Evaluated_By": p.get("evaluated_by", "System"),
            "Date": p.get("created_at", datetime.utcnow().strftime("%Y-%m-%d")),
            "Risk_Alerts": "; ".join([r.get("title", "") for r in p.get("risk_factors", [])]),
        }
        features = p.get("features", {})
        for k in ["Department", "JobRole", "MonthlyIncome", "YearsAtCompany"]:
            if k in features:
                row[k] = features[k]
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


def generate_pdf_report(
    summary: Dict[str, Any],
    predictions: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
) -> bytes:
    """Generates an executive-grade PDF analytical report using FPDF2."""
    pdf = PlatformPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Meta Info block
    now_str = datetime.utcnow().strftime("%B %d, %Y - %H:%M UTC")
    pdf.set_fill_color(248, 250, 252)
    pdf.set_draw_color(203, 213, 225)
    pdf.rect(10, pdf.get_y(), 190, 18, style="DF")

    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(51, 65, 85)
    pdf.set_xy(14, pdf.get_y() + 2)
    pdf.cell(90, 6, f"Generated: {now_str}")
    pdf.cell(90, 6, f"Active Model: {summary.get('model_name', 'Multinomial Logistic Regression')}", new_x="LMARGIN", new_y="NEXT")

    pdf.set_x(14)
    pdf.cell(90, 6, f"Operator Persona: {summary.get('operator_role', 'HR Management')}")
    pdf.cell(90, 6, f"Benchmark Macro-F1: {summary.get('macro_f1', 0.66):.4f}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # 1. Executive Summary KPIs
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 7, "1. Executive Workforce Summary", new_x="LMARGIN", new_y="NEXT")

    total_eval = len(predictions)
    tier_counts = {}
    high_risks = 0
    for p in predictions:
        tier = p.get("predicted_class", "Unknown")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        if p.get("risk_factors"):
            high_risks += len(p["risk_factors"])

    # KPI table
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(30, 58, 138)
    pdf.set_text_color(255, 255, 255)
    cols = [47.5, 47.5, 47.5, 47.5]
    pdf.cell(cols[0], 7, "Total Evaluated", border=1, align="C", fill=True)
    pdf.cell(cols[1], 7, "Top Performers (Exceeds)", border=1, align="C", fill=True)
    pdf.cell(cols[2], 7, "At-Risk Cohorts", border=1, align="C", fill=True)
    pdf.cell(cols[3], 7, "Flagged Risk Alerts", border=1, align="C", fill=True, new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_fill_color(239, 246, 255)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(cols[0], 9, str(total_eval), border=1, align="C", fill=True)
    pdf.cell(cols[1], 9, str(tier_counts.get("Exceeds Expectations", 0)), border=1, align="C", fill=True)
    pdf.cell(cols[2], 9, str(tier_counts.get("Needs Improvement", 0)), border=1, align="C", fill=True)
    pdf.cell(cols[3], 9, str(high_risks), border=1, align="C", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # 2. Performance Tier Distribution
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 7, "2. Performance Tier Distribution & Recommendations", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(51, 65, 85)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(50, 6, "Performance Tier", border=1, fill=True)
    pdf.cell(25, 6, "Headcount", border=1, align="C", fill=True)
    pdf.cell(25, 6, "Proportion", border=1, align="C", fill=True)
    pdf.cell(90, 6, "Strategic Action Recommendation", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(51, 65, 85)
    recs = {
        "Needs Improvement": "Initiate targeted coaching & workload review; address burnout",
        "Partially Meets Expectations": "Targeted skills development and clear quarterly OKR alignment",
        "Meets Expectations": "Sustained recognition and retention incentives for steady value",
        "Exceeds Expectations": "Leadership pipeline nomination and high-impact project assignments",
    }
    for tier in [
        "Needs Improvement",
        "Partially Meets Expectations",
        "Meets Expectations",
        "Exceeds Expectations",
    ]:
        cnt = tier_counts.get(tier, 0)
        pct = f"{(cnt / total_eval * 100):.1f}%" if total_eval > 0 else "0.0%"
        pdf.cell(50, 6, tier, border=1)
        pdf.cell(25, 6, str(cnt), border=1, align="C")
        pdf.cell(25, 6, pct, border=1, align="C")
        pdf.cell(90, 6, recs.get(tier, "Standard ongoing review"), border=1, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # 3. Diagnostic Charts (if available)
    cm_path = REPORT_DIR / "confusion_matrix.png"
    fi_path = REPORT_DIR / "feature_importance.png"
    if cm_path.exists() and fi_path.exists():
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 7, "3. Empirical Model Verification & Interpretability (§4.3)", new_x="LMARGIN", new_y="NEXT")
        curr_y = pdf.get_y()
        pdf.image(str(cm_path), x=12, y=curr_y, w=88)
        pdf.image(str(fi_path), x=105, y=curr_y, w=93)
        pdf.set_y(curr_y + 68)
        pdf.set_font("Helvetica", "I", 7)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(90, 4, "Figure A: Confusion Matrix (Test Set N=1,200)", align="C")
        pdf.cell(95, 4, "Figure B: Permutation Feature Importances", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

    # 4. Sample Evaluated Employees
    if predictions:
        if pdf.get_y() > 220:
            pdf.add_page()
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 7, "4. Employee Assessment Roster (Latest Sample)", new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(30, 41, 59)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(40, 6, "Name / ID", border=1, fill=True)
        pdf.cell(35, 6, "Department", border=1, fill=True)
        pdf.cell(45, 6, "Predicted Tier", border=1, fill=True)
        pdf.cell(25, 6, "Confidence", border=1, align="C", fill=True)
        pdf.cell(45, 6, "Key Operational Note", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(51, 65, 85)
        for p in predictions[:8]:
            name = str(p.get("employee_name", f"Emp #{p.get('emp_id', 'N/A')}"))[:20]
            dept = str(p.get("features", {}).get("Department", "Enterprise"))[:18]
            tier = str(p.get("predicted_class", "N/A"))
            conf = f"{p.get('probability', 0)*100:.1f}%"
            risks = p.get("risk_factors", [])
            note = str(risks[0]["title"])[:22] if risks else "Stable profile"

            pdf.cell(40, 5.5, name, border=1)
            pdf.cell(35, 5.5, dept, border=1)
            pdf.cell(45, 5.5, tier, border=1)
            pdf.cell(25, 5.5, conf, border=1, align="C")
            pdf.cell(45, 5.5, note, border=1, new_x="LMARGIN", new_y="NEXT")

    pdf_bytes = bytes(pdf.output())
    if output_path:
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)
    return pdf_bytes
