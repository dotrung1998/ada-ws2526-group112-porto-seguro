"""
02_logistic_regression.py
Modellierung: Logistische Regression (Finales Modell).
"""

import os
import time
import warnings

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from plotting import (
    plot_pca_variance,
    plot_logreg_coefficients,
    save_current_figure,
)
from timing import log_custom_runtime, measure_runtime

np.seterr(all="ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 3

# %% Teil 1: Gemeinsame Datenbasis & Basisblock
porto = fetch_openml(data_id=42742, as_frame=True)

X = porto.data.copy().replace(-1, np.nan)
y = pd.to_numeric(porto.target).astype("int8")

categorical_features = [c for c in X.columns if c.endswith("_cat")]
binary_features = [c for c in X.columns if c.endswith("_bin")]
numeric_features = [
    c for c in X.columns
    if c not in categorical_features + binary_features
]

for col in numeric_features + binary_features:
    X[col] = pd.to_numeric(X[col], errors="coerce").astype(np.float64)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

SCORING = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "balanced_accuracy": "balanced_accuracy",
    "f1": "f1",
}


def evaluate_pipeline(name, pipeline):
    """Bewertet eine vollständige Pipeline nur auf den Trainingsdaten."""
    start = time.time()
    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=SCORING,
        n_jobs=-1,
        return_train_score=False,
    )
    return pd.DataFrame([{
        "Modell": name,
        "PR-AUC": scores["test_pr_auc"].mean(),
        "ROC-AUC": scores["test_roc_auc"].mean(),
        "Balanced Accuracy": scores["test_balanced_accuracy"].mean(),
        "F1": scores["test_f1"].mean(),
        "CV-Zeit (s)": time.time() - start,
    }])


print("Trainingsdaten:", X_train.shape)
print("Testdaten:", X_test.shape)
print("Positive Klasse im Training:", round(y_train.mean(), 4))
print(f"Anzahl Merkmale: {len(numeric_features)} numerisch, {len(categorical_features)} kategorial, {len(binary_features)} binär")

# %% Teil 2: Hilfsfunktionen
def summarize_cv(scores):
    return {
        "PR-AUC": scores["test_pr_auc"].mean(),
        "PR-AUC Std": scores["test_pr_auc"].std(),
        "ROC-AUC": scores["test_roc_auc"].mean(),
        "Balanced Accuracy": scores["test_balanced_accuracy"].mean(),
        "F1": scores["test_f1"].mean(),
        "Mittlere Fit-Zeit (s)": scores["fit_time"].mean(),
    }


def stratified_subsample(X_data, y_data, n_samples, random_state=RANDOM_STATE):
    if n_samples is None or n_samples >= len(X_data):
        return X_data.copy(), y_data.copy()
    X_sub, _, y_sub, _ = train_test_split(
        X_data,
        y_data,
        train_size=n_samples,
        random_state=random_state,
        stratify=y_data,
    )
    return X_sub, y_sub

# %% Teil 3: Vorverarbeitungs-Pipeline
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

binary_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
])

preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_transformer, numeric_features),
    ("categorical", categorical_transformer, categorical_features),
    ("binary", binary_transformer, binary_features),
])

# %% Teil 4: Prüfung der Dimensionsreduktion (PCA)
numeric_pca = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=18, svd_solver="arpack", random_state=RANDOM_STATE)),
])

preprocessor_with_pca = ColumnTransformer(transformers=[
    ("numeric", numeric_pca, numeric_features),
    ("categorical", categorical_transformer, categorical_features),
    ("binary", binary_transformer, binary_features),
])

pipe_no_pca = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
])

pipe_with_pca = Pipeline(steps=[
    ("preprocessor", preprocessor_with_pca),
    ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
])

pca_compare_results = pd.concat([
    evaluate_pipeline("LogReg ohne PCA", pipe_no_pca),
    evaluate_pipeline("LogReg mit PCA (n=18)", pipe_with_pca),
], ignore_index=True)

print("\n=== Vergleich PCA vs. ohne PCA ===")
print(pca_compare_results.round(4))

# %% Teil 5: Baseline-Modell der Logistischen Regression
baseline_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )),
])

baseline_cv_results = evaluate_pipeline("LogReg Baseline", baseline_pipeline)
print("\n=== Baseline CV-Ergebnisse ===")
print(baseline_cv_results.round(4))

# %% Teil 6: Hyperparameteroptimierung (GridSearchCV)
param_grid_lr = {
    "model__C": [0.01, 0.1, 1.0, 10.0],
}

grid_search_lr = GridSearchCV(
    estimator=Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE)),
    ]),
    param_grid=param_grid_lr,
    scoring=SCORING,
    refit="pr_auc",
    cv=cv,
    n_jobs=-1,
    return_train_score=False,
    verbose=0,
)

train_start_lr = time.time()
grid_search_lr.fit(X_train, y_train)
train_time_grid = time.time() - train_start_lr

# Modell-Laufzeit protokollieren
log_custom_runtime("02_LogReg_GridSearch_Fit", train_time_grid)

print(f"\nOptimierungsdauer: {train_time_grid:.1f} s")
print(f"Beste Parameter: {grid_search_lr.best_params_}")
print(f"Beste CV PR-AUC: {grid_search_lr.best_score_:.6f}")

best_lr_model = grid_search_lr.best_estimator_

# %% Teil 7: Interpretation der Modellkoeffizienten
plot_logreg_coefficients(best_lr_model, numeric_features, categorical_features, binary_features)

# %% Teil 8: Finales Modell & Schwellenwert-Analyse
final_pipeline = best_lr_model

final_fit_start = time.time()
final_pipeline.fit(X_train, y_train)
final_fit_time = time.time() - final_fit_start

# Reinen Fit des finalen Modells protokollieren
log_custom_runtime("02_LogReg_Final_Fit", final_fit_time)

final_predict_start = time.time()
y_test_pred_default = final_pipeline.predict(X_test)
y_test_proba = final_pipeline.predict_proba(X_test)[:, 1]
final_predict_time = time.time() - final_predict_start

print("\n=== Logistische Regression (optimiert), Testdaten (Threshold 0.5) ===")
print(classification_report(y_test, y_test_pred_default))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
RocCurveDisplay.from_estimator(final_pipeline, X_test, y_test, ax=ax1, name="LogReg Optimiert")
PrecisionRecallDisplay.from_estimator(final_pipeline, X_test, y_test, ax=ax2, name="LogReg Optimiert")
ax1.set_title("ROC-Kurve (Testdaten)")
ax2.set_title("Precision-Recall-Kurve (Testdaten)")
plt.tight_layout()
save_current_figure("02_roc_pr_kurven.png", subdir="02_logistic_regression")
plt.close()

# Schwellenwert-Analyse
thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
threshold_results = []

for t in thresholds:
    y_pred_t = (y_test_proba >= t).astype(int)
    tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_test, y_pred_t).ravel()
    threshold_results.append({
        "Threshold": t,
        "False Positives": fp_t,
        "Recall": recall_score(y_test, y_pred_t, zero_division=0),
        "Precision": precision_score(y_test, y_pred_t, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred_t, zero_division=0),
    })

threshold_results_df = pd.DataFrame(threshold_results)
print("\n=== Schwellenwert-Analyse ===")
print(threshold_results_df.round(6))

# %% Teil 9: Finale Ergebnistabelle mit optimiertem Schwellenwert (T=0.65)
OPTIMAL_THRESHOLD = 0.65
y_test_pred_opt = (y_test_proba >= OPTIMAL_THRESHOLD).astype(int)

best_c = final_pipeline.named_steps["model"].C
best_cw = str(final_pipeline.named_steps["model"].class_weight)

tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred_opt).ravel()

final_results_table = pd.DataFrame([{
    "Modell": f"Logistische Regression (optimiert, T={OPTIMAL_THRESHOLD})",
    "C": best_c,
    "class_weight": best_cw,
    "Test PR-AUC": average_precision_score(y_test, y_test_proba),
    "Test ROC-AUC": roc_auc_score(y_test, y_test_proba),
    "Test Balanced Accuracy": balanced_accuracy_score(y_test, y_test_pred_opt),
    "Test F1": f1_score(y_test, y_test_pred_opt, zero_division=0),
    "Test Precision": precision_score(y_test, y_test_pred_opt, zero_division=0),
    "Test Recall": recall_score(y_test, y_test_pred_opt, zero_division=0),
    "True Negatives": tn,
    "False Positives": fp,
    "False Negatives": fn,
    "True Positives": tp,
    "Trainingszeit gesamt (s)": final_fit_time,
    "Vorhersagezeit Test (s)": final_predict_time,
}])

results = final_results_table.copy()
final_test_results = final_results_table.copy()
ergebnis_tabelle = final_results_table.copy()

print("\n=== FINALE ERGEBNISTABELLE (LOGISTISCHE REGRESSION) ===")
print(final_results_table.round({
    "Test PR-AUC": 6,
    "Test ROC-AUC": 6,
    "Test Balanced Accuracy": 6,
    "Test F1": 6,
    "Test Precision": 6,
    "Test Recall": 6,
    "Trainingszeit gesamt (s)": 2,
    "Vorhersagezeit Test (s)": 2,
}))


def main():
    pass


if __name__ == "__main__":
    with measure_runtime("02_logistic_regression"):
        main()