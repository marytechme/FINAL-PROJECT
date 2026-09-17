"""
03_shap_interpretability.py
Explainable AI (XAI) analysis using SHAP for the recommended alternative
credit scoring model (XGBoost), addressing the interpretability requirement
identified in Chapter 2.7 (Bias and Interpretability Issues).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

RANDOM_SEED = 42

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

model = XGBClassifier(
    n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.85,
    colsample_bytree=0.85, eval_metric="logloss", random_state=RANDOM_SEED
)

pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
pipe.fit(X_train, y_train)

# Transform test data through the preprocessing step to get final feature matrix
X_test_transformed = pipe.named_steps["preprocess"].transform(X_test)
cat_encoder = pipe.named_steps["preprocess"].named_transformers_["cat"]
cat_names = list(cat_encoder.get_feature_names_out(categorical_features))
feature_names = numeric_features + cat_names

X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)

explainer = shap.TreeExplainer(pipe.named_steps["model"])
shap_values = explainer(X_test_df)

# Global feature importance (summary/beeswarm plot)
plt.figure()
shap.summary_plot(shap_values, X_test_df, show=False, plot_size=(9, 6))
plt.title("SHAP Summary Plot — XGBoost Alternative Credit Scoring Model", fontsize=12)
plt.tight_layout()
plt.savefig("shap_summary_plot.png", dpi=220, bbox_inches="tight")
plt.close()

# Mean |SHAP value| bar chart
plt.figure()
shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False, plot_size=(8, 6))
plt.title("Mean |SHAP Value| by Feature — Global Importance", fontsize=12)
plt.tight_layout()
plt.savefig("shap_bar_plot.png", dpi=220, bbox_inches="tight")
plt.close()

# Individual borrower explanation (waterfall) — a representative high-risk case
proba = pipe.predict_proba(X_test)[:, 1]
idx_high_risk = int(np.argmax(proba))

plt.figure()
shap.plots.waterfall(shap_values[idx_high_risk], show=False, max_display=10)
plt.title("SHAP Waterfall — Example High-Risk Applicant Explanation", fontsize=11)
plt.tight_layout()
plt.savefig("shap_waterfall_example.png", dpi=220, bbox_inches="tight")
plt.close()

# Save top mean-abs shap features to a small table for the report
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
shap_importance_df = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

shap_importance_df.to_csv("shap_feature_importance.csv", index=False)
print(shap_importance_df.head(10).to_string(index=False))
print("\nSHAP interpretability artefacts saved.")
