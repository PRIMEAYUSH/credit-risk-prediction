# Credit Risk / Loan Default Prediction - German Credit dataset (1,000 applicants)
# Run top to bottom in Google Colab (paste each "# %%" block as a cell) or: python credit_risk_project.py
# Tools: Python, Pandas, Scikit-learn, Matplotlib

# %% 1. Load data
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve

URL = "https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv"
df = pd.read_csv(URL)

# Original label: 1 = good credit, 0 = bad credit. We model DEFAULT RISK, so flip it.
df["default"] = (df["credit_risk"] == 0).astype(int)
df = df.drop(columns="credit_risk")
print("Rows, columns:", df.shape)
print("Overall default rate: {:.1%}".format(df["default"].mean()))

# %% 2. EDA - default rate by key drivers
for col in ["status", "credit_history", "purpose", "savings"]:
    print("\nDefault rate by", col)
    print((df.groupby(col)["default"].mean().sort_values(ascending=False) * 100).round(1))

df["amount_band"] = pd.qcut(df["amount"], 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
print("\nDefault rate by loan amount quartile")
print((df.groupby("amount_band", observed=True)["default"].mean() * 100).round(1))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
(df.groupby("status")["default"].mean() * 100).sort_values().plot.barh(ax=ax[0], color="#c0392b")
ax[0].set_title("Default rate (%) by checking account status")
(df.groupby("amount_band", observed=True)["default"].mean() * 100).plot.bar(ax=ax[1], color="#2c3e50")
ax[1].set_title("Default rate (%) by loan amount quartile")
plt.tight_layout(); plt.savefig("eda_default_drivers.png", dpi=150)

# %% 3. Preprocess + train/test split
X = df.drop(columns=["default", "amount_band"])
y = df["default"]
num_cols = X.select_dtypes("number").columns.tolist()
cat_cols = X.select_dtypes("object").columns.tolist()

pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# %% 4. Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
}
rows, fitted = [], {}
plt.figure(figsize=(5, 4))
for name, m in models.items():
    pipe = Pipeline([("pre", pre), ("model", m)]).fit(X_tr, y_tr)
    fitted[name] = pipe
    p = pipe.predict_proba(X_te)[:, 1]
    pred = (p >= 0.5).astype(int)
    rows.append({"Model": name,
                 "Accuracy": accuracy_score(y_te, pred),
                 "Precision": precision_score(y_te, pred),
                 "Recall": recall_score(y_te, pred),
                 "ROC-AUC": roc_auc_score(y_te, p)})
    fpr, tpr, _ = roc_curve(y_te, p)
    plt.plot(fpr, tpr, label="{} (AUC={:.2f})".format(name, rows[-1]["ROC-AUC"]))
plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC curves"); plt.legend(); plt.tight_layout(); plt.savefig("roc_curves.png", dpi=150)
print("\n", pd.DataFrame(rows).round(3).to_string(index=False))

# %% 5. Top risk drivers (Random Forest feature importance)
rf = fitted["Random Forest"]
names = rf.named_steps["pre"].get_feature_names_out()
imp = pd.Series(rf.named_steps["model"].feature_importances_, index=names).sort_values(ascending=False)
print("\nTop 8 risk drivers:\n", imp.head(8).round(3))
imp.head(8)[::-1].plot.barh(figsize=(7, 4), color="#c0392b")
plt.title("Top 8 default risk drivers"); plt.tight_layout(); plt.savefig("feature_importance.png", dpi=150)
