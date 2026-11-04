"""Train LR / RF / GB, benchmark models, persist production artifacts, and generate diagnostics.

Fulfills Phase 1-2 System Proposal specifications (§4.1 - §4.4):
- Stratified 80/20 train/test split across 5-fold cross-validation
- Benchmarks: Multinomial Logistic Regression, Random Forest, Gradient Boosting
- Export formats:
    * models/performance_model.pkl (ModelBundle for standard pickle.load)
    * models/pipeline.pkl (Pure scikit-learn Pipeline)
    * models/performance_model.joblib (joblib binary)
- Visual reports:
    * reports/leaderboard.json
    * reports/classification_report.json
    * reports/confusion_matrix.png
    * reports/feature_importance.png
    * reports/classification_metrics.png (Figure 6)
    * reports/multiclass_roc_curve.png
"""
import json
import os
import pickle
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, log_loss, roc_auc_score, roc_curve, auc,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import label_binarize

from .config import (
    CLASS_NAMES, CV_FOLDS, DATA_DIR, MODEL_DIR, RANDOM_STATE,
    REPORT_DIR, TARGET_COL, TEST_SIZE,
)
from .model_bundle import ModelBundle
from .preprocessing import build_pipeline


def load_data():
    csv_path = DATA_DIR / "hr_data.csv"
    if not csv_path.exists():
        from data.generate_data import generate
        print("hr_data.csv not found; generating synthetic dataset...")
        generate()
    df = pd.read_csv(csv_path)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].map(dict(enumerate(CLASS_NAMES)))
    return X, y


def model_grids():
    return {
        "Multinomial Logistic Regression": (
            LogisticRegression(
                solver="lbfgs",
                max_iter=3000,
                random_state=RANDOM_STATE,
                class_weight="balanced",
            ),
            {"classifier__C": [0.1, 1.0, 5.0]},
        ),
        "Gradient Boosting Classifier": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "classifier__n_estimators": [100, 150],
                "classifier__learning_rate": [0.08, 0.12],
                "classifier__max_depth": [3, 4],
            },
        ),
        "Random Forest Classifier": (
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
            ),
            {
                "classifier__n_estimators": [100, 150],
                "classifier__max_depth": [8, 12],
            },
        ),
    }


def evaluate(estimator, X_test, y_test, labels):
    y_pred = estimator.predict(X_test)
    y_proba = estimator.predict_proba(X_test)
    est_classes = estimator.classes_
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "roc_auc_ovr": float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro", labels=est_classes)),
        "log_loss": float(log_loss(y_test, y_proba, labels=est_classes)),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def generate_diagnostic_plots(best_name, best_pipe, X_te, y_te, y_pred, y_proba, labels, report_dict):
    """Generates all visual diagnostic reports matching Figure 6 & Figure 7 in the proposal."""
    # 1. Confusion Matrix (Figure 7 left)
    cm = confusion_matrix(y_te, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, cbar=True,
    )
    plt.xlabel("Predicted Class", fontsize=11, fontweight="bold")
    plt.ylabel("Actual Class", fontsize=11, fontweight="bold")
    plt.title(f"Confusion Matrix — {best_name}\n(1,200 Record Test Set)", fontsize=12, fontweight="bold")
    plt.xticks(rotation=25, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

    # 2. Per-class Precision, Recall, and F1-score (Figure 6)
    classes = [c for c in labels if c in report_dict]
    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(classes))
    width = 0.25

    plt.figure(figsize=(10, 6))
    colors = ["#1e3a8a", "#d97706", "#059669"]  # navy, amber, emerald matching report theme
    for idx, metric in enumerate(metrics):
        values = [report_dict[c][metric] for c in classes]
        bars = plt.bar(x + (idx - 1) * width, values, width, label=metric.capitalize(), color=colors[idx])
        for bar in bars:
            height = bar.get_height()
            plt.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=9,
            )

    plt.ylabel("Score", fontsize=11, fontweight="bold")
    plt.title("Per-Class Precision, Recall, and F1-Score (Test Set N=1,200)", fontsize=13, fontweight="bold")
    plt.xticks(x, classes, rotation=15, ha="right", fontsize=10)
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "classification_metrics.png", dpi=200)
    plt.close()

    # 3. Permutation Feature Importance (Figure 7 right)
    print("Computing permutation importance for top features...")
    imp = permutation_importance(
        best_pipe, X_te, y_te, n_repeats=5,
        random_state=RANDOM_STATE, scoring="f1_macro", n_jobs=-1,
    )
    imp_df = pd.DataFrame({
        "feature": X_te.columns,
        "importance": imp.importances_mean,
        "std": imp.importances_std,
    }).sort_values("importance", ascending=False)
    imp_df.to_csv(REPORT_DIR / "feature_importance.csv", index=False)

    plt.figure(figsize=(10, 8))
    top_imp = imp_df.head(15).iloc[::-1]  # ascending for horizontal bar chart
    plt.barh(top_imp["feature"], top_imp["importance"], color="#2563eb", xerr=top_imp["std"], capsize=3)
    plt.xlabel("Mean Permutation Importance (F1-Macro Degradation)", fontsize=11, fontweight="bold")
    plt.ylabel("Employee Attribute", fontsize=11, fontweight="bold")
    plt.title(f"Top 15 Feature Importances — {best_name}", fontsize=13, fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "feature_importance.png", dpi=200)
    plt.close()

    # 4. Multiclass ROC Curves (OvR)
    est_classes = list(best_pipe.classes_)
    y_bin = label_binarize(y_te, classes=est_classes)
    plt.figure(figsize=(8, 6))
    for i, c in enumerate(est_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{c} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Majority Class Baseline (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title(f"Multiclass One-vs-Rest ROC Curves — {best_name}", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "multiclass_roc_curve.png", dpi=200)
    plt.close()


def main():
    print("==================================================")
    print("Employee Performance Prediction Platform: ML Training")
    print("==================================================")
    X, y = load_data()
    labels = CLASS_NAMES

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )
    print(f"Dataset split: Train = {len(X_tr)} records | Test = {len(X_te)} records")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    leaderboard = []
    best = {"score": -1.0, "pipeline": None, "name": None, "eval": None}

    # Majority baseline
    maj_counts = y_te.value_counts()
    maj_acc = maj_counts.max() / len(y_te)
    leaderboard.append({
        "model": "Majority Class Baseline",
        "cv_macro_f1": 0.1429,
        "test_accuracy": round(float(maj_acc), 4),
        "test_macro_f1": 0.1429,
        "roc_auc_ovr": 0.5000,
        "log_loss": None,
        "best_params": "N/A",
    })

    for name, (model, grid) in model_grids().items():
        print(f"\nTraining candidate: {name} ...")
        pipe = build_pipeline(model)
        search = GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=0)
        search.fit(X_tr, y_tr)
        cv_std = float(search.cv_results_["std_test_score"][search.best_index_])
        m = evaluate(search.best_estimator_, X_te, y_te, labels)

        print(f"  Best params   : {search.best_params_}")
        print(f"  5-Fold CV F1  : {search.best_score_:.4f} ± {cv_std:.4f}")
        print(f"  Test Accuracy : {m['accuracy'] * 100:.2f}%")
        print(f"  Test Macro-F1 : {m['macro_f1']:.4f}")
        print(f"  ROC-AUC (OvR) : {m['roc_auc_ovr']:.4f}")
        print(f"  Log Loss      : {m['log_loss']:.4f}")

        leaderboard.append({
            "model": name,
            "cv_macro_f1": round(float(search.best_score_), 4),
            "cv_macro_f1_std": round(cv_std, 4),
            "test_accuracy": round(m["accuracy"], 4),
            "test_macro_f1": round(m["macro_f1"], 4),
            "roc_auc_ovr": round(m["roc_auc_ovr"], 4),
            "log_loss": round(m["log_loss"], 4),
            "best_params": search.best_params_,
        })

        if m["macro_f1"] > best["score"]:
            best = {
                "score": m["macro_f1"],
                "pipeline": search.best_estimator_,
                "name": name,
                "eval": m,
                "params": search.best_params_,
            }

    print(f"\n==================================================")
    print(f"Selected Best Model: {best['name']} (Macro-F1 = {best['score']:.4f})")
    print(f"==================================================")

    # Wrap inside ModelBundle (§4.4)
    bundle_meta = {
        "algorithm": best["name"],
        "accuracy": best["eval"]["accuracy"],
        "macro_f1": best["eval"]["macro_f1"],
        "roc_auc_ovr": best["eval"]["roc_auc_ovr"],
        "log_loss": best["eval"]["log_loss"],
        "best_params": best["params"],
    }
    bundle = ModelBundle(pipeline=best["pipeline"], class_names=labels, metadata=bundle_meta)

    # 1. models/performance_model.pkl (ModelBundle for standard pickle.load §4.4)
    with open(MODEL_DIR / "performance_model.pkl", "wb") as f:
        pickle.dump(bundle, f)

    # 2. models/pipeline.pkl (Pure scikit-learn Pipeline §4.4)
    with open(MODEL_DIR / "pipeline.pkl", "wb") as f:
        pickle.dump(best["pipeline"], f)

    # 3. models/performance_model.joblib (High-performance joblib binary §4.4)
    joblib.dump(best["pipeline"], MODEL_DIR / "performance_model.joblib")

    # Persist JSON Leaderboard
    (REPORT_DIR / "leaderboard.json").write_text(json.dumps(leaderboard, indent=2))

    # Classification Report
    report = classification_report(y_te, best["eval"]["y_pred"], labels=labels, output_dict=True)
    (REPORT_DIR / "classification_report.json").write_text(json.dumps(report, indent=2))
    print("\nDetailed Test Classification Report:\n")
    print(classification_report(y_te, best["eval"]["y_pred"], labels=labels))

    # Visual Diagnostic Reports
    generate_diagnostic_plots(
        best["name"], best["pipeline"], X_te, y_te,
        best["eval"]["y_pred"], best["eval"]["y_proba"], labels, report,
    )

    print(f"\nAll models and reports successfully exported to:")
    print(f"  - Models  : {MODEL_DIR}")
    print(f"  - Reports : {REPORT_DIR}")
    return bundle


if __name__ == "__main__":
    main()
