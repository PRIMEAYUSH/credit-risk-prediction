"""
Credit Risk / Loan Default Prediction
Dataset: German Credit (1,000 applicants)

Run:
    python credit_risk_project.py

The script downloads the public dataset, performs EDA, trains multiple
classification models, evaluates them, creates plots, and saves the best
model pipeline for the Streamlit app.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_URL = "https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv"
OUTPUT_DIR = Path("outputs")
MODEL_DIR = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# 1. Load data
df = pd.read_csv(DATA_URL)

# Original label: 1 = good credit, 0 = bad credit.
# We model DEFAULT RISK, so flip it.
df["default"] = (df["credit_risk"] == 0).astype(int)
df = df.drop(columns="credit_risk")

print("=" * 70)
print("CREDIT RISK / LOAN DEFAULT PREDICTION")
print("=" * 70)
print("Dataset shape:", df.shape)
print("Overall default rate: {:.1%}".format(df["default"].mean()))

# 2. EDA
for col in ["status", "credit_history", "purpose", "savings"]:
    print(f"\nDefault rate by {col}")
    print((df.groupby(col)["default"].mean().sort_values(ascending=False) * 100).round(1))

df["amount_band"] = pd.qcut(
    df["amount"], 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
)

print("\nDefault rate by loan amount quartile")
print((df.groupby("amount_band", observed=True)["default"].mean() * 100).round(1))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
(
    df.groupby("status")["default"].mean() * 100
).sort_values().plot.barh(ax=ax[0])
ax[0].set_title("Default rate (%) by checking account status")
ax[0].set_xlabel("Default rate (%)")

(
    df.groupby("amount_band", observed=True)["default"].mean() * 100
).plot.bar(ax=ax[1])
ax[1].set_title("Default rate (%) by loan amount quartile")
ax[1].set_ylabel("Default rate (%)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "eda_default_drivers.png", dpi=180)
plt.close()

# 3. Prepare data
X = df.drop(columns=["default", "amount_band"])
y = df["default"]

num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(include="object").columns.tolist()

preprocessor = ColumnTransformer(
    [
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# 4. Train and compare models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}

results = []
fitted = {}

plt.figure(figsize=(7, 5))

for name, model in models.items():
    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipe.fit(X_train, y_train)
    fitted[name] = pipe

    probabilities = pipe.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
    }
    results.append(metrics)

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC={metrics['ROC-AUC']:.2f})",
    )

results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

plt.plot([0, 1], [0, 1], "k--", label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_curves.png", dpi=180)
plt.close()

print("\nModel comparison:")
print(results_df.round(3).to_string(index=False))

# 5. Select best model by ROC-AUC
best_model_name = results_df.iloc[0]["Model"]
best_model = fitted[best_model_name]
best_probabilities = best_model.predict_proba(X_test)[:, 1]
best_predictions = (best_probabilities >= 0.50).astype(int)

print(f"\nBest model: {best_model_name}")
print("\nClassification report:")
print(classification_report(y_test, best_predictions, target_names=["Low risk", "Default risk"]))

# 6. Confusion matrix
cm = confusion_matrix(y_test, best_predictions)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Low risk", "Default risk"],
)
disp.plot(values_format="d")
plt.title(f"Confusion Matrix — {best_model_name}")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=180)
plt.close()

# 7. Feature importance for Random Forest
rf = fitted["Random Forest"]
feature_names = rf.named_steps["preprocessor"].get_feature_names_out()
raw_importance = pd.Series(
    rf.named_steps["model"].feature_importances_,
    index=feature_names,
)

# Aggregate one-hot encoded features back to their original columns.
importance = {}
for transformed_name, value in raw_importance.items():
    original = transformed_name.split("__", 1)[-1]
    original = original.split("_", 1)[0] if original in num_cols else original
    # For categorical columns, find the matching original column.
    matched = next(
        (col for col in cat_cols if original.startswith(col + "_")),
        original,
    )
    importance[matched] = importance.get(matched, 0) + value

importance = (
    pd.Series(importance)
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop risk drivers:")
print(importance.round(4).to_string())

importance.sort_values().plot.barh(figsize=(8, 5))
plt.title("Top Credit Risk Drivers — Random Forest")
plt.xlabel("Aggregated feature importance")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=180)
plt.close()

# 8. Save best model + metadata
joblib.dump(best_model, MODEL_DIR / "credit_risk_model.joblib")
pd.Series(X.columns).to_csv(MODEL_DIR / "feature_columns.csv", index=False)

with open(MODEL_DIR / "model_info.txt", "w", encoding="utf-8") as f:
    f.write(f"Best model: {best_model_name}\n")
    f.write(f"ROC-AUC: {results_df.iloc[0]['ROC-AUC']:.4f}\n")
    f.write("Prediction threshold: 0.50\n")

print("\nSaved:")
print("- outputs/eda_default_drivers.png")
print("- outputs/roc_curves.png")
print("- outputs/confusion_matrix.png")
print("- outputs/feature_importance.png")
print("- outputs/model_comparison.csv")
print("- models/credit_risk_model.joblib")
print("\nProject completed successfully.")
