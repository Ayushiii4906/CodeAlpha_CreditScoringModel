"""
Credit Scoring Model — educational demo.
Creates synthetic financial records, engineers features, trains Logistic Regression
and Random Forest, and reports classification metrics + ROC-AUC.
Do not use synthetic data or this demo for real lending decisions.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    RocCurveDisplay, accuracy_score, precision_score, recall_score, f1_score
)

def make_demo_data(n=1500, seed=42):
    rng = np.random.default_rng(seed)
    income = rng.normal(65000, 22000, n).clip(12000, 180000)
    debt = rng.normal(18000, 13000, n).clip(0, 100000)
    payment_history = rng.integers(0, 13, n)  # on-time payments out of 12
    age = rng.integers(18, 70, n)
    utilization = (debt / income).clip(0, 2)
    # Transparent synthetic label; higher means more likely creditworthy.
    score = (0.045 * payment_history - 2.2 * utilization
             + 0.000012 * income - 0.000008 * debt
             + rng.normal(0, 0.45, n))
    creditworthy = (score > np.quantile(score, 0.42)).astype(int)
    return pd.DataFrame({
        "income": income.round(0),
        "debt": debt.round(0),
        "payment_history": payment_history,
        "age": age,
        "creditworthy": creditworthy
    })

def main():
    df = make_demo_data()
    # Feature engineering: debt-to-income ratio and payment-history rate.
    df["debt_to_income"] = df["debt"] / df["income"].clip(lower=1)
    df["payment_rate"] = df["payment_history"] / 12.0
    X = df[["income", "debt", "payment_history", "age",
             "debt_to_income", "payment_rate"]]
    y = df["creditworthy"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    models = {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced",
            random_state=42
        )
    }

    plt.figure()
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        print(f"\n=== {name} ===")
        print("Accuracy:", round(accuracy_score(y_test, pred), 4))
        print("Precision:", round(precision_score(y_test, pred), 4))
        print("Recall:", round(recall_score(y_test, pred), 4))
        print("F1-score:", round(f1_score(y_test, pred), 4))
        print("ROC-AUC:", round(roc_auc_score(y_test, prob), 4))
        print("Confusion matrix:\n", confusion_matrix(y_test, pred))
        print(classification_report(y_test, pred, target_names=["Not creditworthy", "Creditworthy"]))
        RocCurveDisplay.from_predictions(y_test, prob, name=name)

    plt.title("Credit Scoring — ROC Curves")
    plt.tight_layout()
    plt.savefig("credit_scoring_roc.png", dpi=150)
    print("\nSaved ROC plot to credit_scoring_roc.png")

if __name__ == "__main__":
    main()
