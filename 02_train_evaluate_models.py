"""
02_train_evaluate_models.py
Comparative training & evaluation of 5 ML algorithms for alternative credit
scoring: Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost.
Produces the metrics table, ROC curves, confusion matrices, and feature
importance / SHAP outputs used in Chapter 4 (Implementation) and Chapter 5
(Testing, Results & Evaluation).
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)

RANDOM_SEED = 42
sns.set_theme(style="whitegrid")

DEEP_BLUE = "#0B3D91"
ORANGE = "#F5821F"
PALETTE = ["#0B3D91", "#F5821F", "#1F6FB2", "#F2A65A", "#5B8FCB"]

# ----------------------------------------------------------------------------------
# 1. Load data & split
# ----------------------------------------------------------------------------------
df = pd.read_csv("credit_scoring_dataset.csv")

X = df.drop(columns=["default"])
y = df["default"]

categorical_features = ["business_type"]
numeric_features = [c for c in X.columns if c not in categorical_features]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
)

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
])

# ----------------------------------------------------------------------------------
# 2. Define models
# ----------------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_SEED),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=RANDOM_SEED),
    "Random Forest": RandomForestClassifier(
        n_estimators=400, max_depth=8, min_samples_leaf=5, random_state=RANDOM_SEED, n_jobs=-1
    ),
    "SVM (RBF)": SVC(kernel="rbf", C=2.0, gamma="scale", probability=True, random_state=RANDOM_SEED),
    "XGBoost": XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.85,
        colsample_bytree=0.85, eval_metric="logloss", random_state=RANDOM_SEED
    ),
}

results = []
roc_data = {}
fitted_pipelines = {}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

for name, clf in models.items():
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])
    pipe.fit(X_train, y_train)
    fitted_pipelines[name] = pipe

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    # 5-fold CV AUC for robustness check
    cv_auc = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1)

    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC (Test)": round(auc, 4),
        "ROC-AUC (5-fold CV mean)": round(cv_auc.mean(), 4),
        "ROC-AUC (5-fold CV std)": round(cv_auc.std(), 4),
    })

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_data[name] = (fpr, tpr, auc)

    # Confusion matrix figure
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4.2, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Default", "Default"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Confusion Matrix — {name}", fontsize=11, color=DEEP_BLUE, fontweight="bold")
    plt.tight_layout()
    fname = f"cm_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(fname, dpi=200)
    plt.close()

results_df = pd.DataFrame(results).sort_values("ROC-AUC (Test)", ascending=False).reset_index(drop=True)
print(results_df.to_string(index=False))
results_df.to_csv("model_results.csv", index=False)

# ----------------------------------------------------------------------------------
# 3. ROC Curve comparison chart
# ----------------------------------------------------------------------------------
plt.figure(figsize=(7, 6))
for i, (name, (fpr, tpr, auc)) in enumerate(roc_data.items()):
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=PALETTE[i % len(PALETTE)], linewidth=2)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Random Classifier")
plt.xlabel("False Positive Rate", fontsize=11)
plt.ylabel("True Positive Rate", fontsize=11)
plt.title("ROC Curve Comparison — Alternative Credit Scoring Models", fontsize=12, color=DEEP_BLUE, fontweight="bold")
plt.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("roc_curve_comparison.png", dpi=220)
plt.close()

# ----------------------------------------------------------------------------------
# 4. Grouped bar chart — metric comparison across models
# ----------------------------------------------------------------------------------
metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC (Test)"]
plot_df = results_df.set_index("Model")[metric_cols]

fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(plot_df.index))
width = 0.15
for i, metric in enumerate(metric_cols):
    ax.bar(x + i * width, plot_df[metric], width=width,
           label=metric, color=[DEEP_BLUE, ORANGE, "#1F6FB2", "#F2A65A", "#5B8FCB"][i])
ax.set_xticks(x + width * 2)
ax.set_xticklabels(plot_df.index, rotation=10)
ax.set_ylim(0, 1.0)
ax.set_title("Performance Metric Comparison Across Models", fontsize=13, color=DEEP_BLUE, fontweight="bold")
ax.legend(loc="lower right", ncol=3, fontsize=8.5)
plt.tight_layout()
plt.savefig("metric_comparison_bar.png", dpi=220)
plt.close()

# ----------------------------------------------------------------------------------
# 5. Feature importance — Random Forest & XGBoost
# ----------------------------------------------------------------------------------
def get_feature_names(preprocessor):
    num_names = numeric_features
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(categorical_features))
    return num_names + cat_names

for name in ["Random Forest", "XGBoost"]:
    pipe = fitted_pipelines[name]
    model = pipe.named_steps["model"]
    feat_names = get_feature_names(pipe.named_steps["preprocess"])
    importances = model.feature_importances_
    imp_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values(
        "importance", ascending=True
    ).tail(12)

    plt.figure(figsize=(8, 6))
    plt.barh(imp_df["feature"], imp_df["importance"], color=DEEP_BLUE if name == "Random Forest" else ORANGE)
    plt.title(f"Top 12 Feature Importances — {name}", fontsize=12, color=DEEP_BLUE, fontweight="bold")
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.savefig(f"feature_importance_{name.replace(' ', '_')}.png", dpi=220)
    plt.close()

# ----------------------------------------------------------------------------------
# 6. Save summary JSON for downstream report generation
# ----------------------------------------------------------------------------------
summary = {
    "n_samples": int(len(df)),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "default_rate": round(float(y.mean()), 4),
    "best_model": results_df.iloc[0]["Model"],
    "best_auc": float(results_df.iloc[0]["ROC-AUC (Test)"]),
}
with open("run_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nSummary:", summary)
print("\nAll charts and result tables saved.")
