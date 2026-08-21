"""
04_xgboost_histgradientboosting.py
Vergleicht XGBoost und HistGradientBoosting auf dem Porto-Seguro-Datensatz.
"""

import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
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
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from plotting import save_current_figure
from timing import log_custom_runtime, measure_runtime

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception as exc:
    XGBClassifier = None
    HAS_XGBOOST = False
    print(f"XGBoost ist in dieser Umgebung nicht verfuegbar und wird uebersprungen: {exc}")

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 3

# %% Teil 1: Gemeinsame Datenbasis
porto = fetch_openml(data_id=42742, as_frame=True)

X = porto.data.copy().replace(-1, np.nan)
y = pd.to_numeric(porto.target).astype("int8")

categorical_features = [c for c in X.columns if c.endswith("_cat")]
binary_features = [c for c in X.columns if c.endswith("_bin")]
numeric_features = [
    c for c in X.columns
    if c not in categorical_features + binary_features
]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

SCORING = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "balanced_accuracy": "balanced_accuracy",
    "f1": "f1",
}


def evaluate_pipeline(name, pipeline):
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
print("Positive Klasse im Training:", y_train.mean().round(4))

# %% Teil 3: Preprocessing-Pipelines
num_transformer_xgb = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

cat_transformer_xgb = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

bin_transformer_xgb = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
])

preprocessor_xgb = ColumnTransformer(transformers=[
    ("num", num_transformer_xgb, numeric_features),
    ("cat", cat_transformer_xgb, categorical_features),
    ("bin", bin_transformer_xgb, binary_features),
])

cat_transformer_hgb = OrdinalEncoder(
    handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2
)

preprocessor_hgb = ColumnTransformer(transformers=[
    ("num", "passthrough", numeric_features),
    ("cat", cat_transformer_hgb, categorical_features),
    ("bin", "passthrough", binary_features),
])

categorical_mask = (
    [False] * len(numeric_features)
    + [True] * len(categorical_features)
    + [False] * len(binary_features)
)

# %% Teil 4: XGBoost
xgb_available = HAS_XGBOOST
best_xgb_params = {}

if xgb_available:
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    param_dist_xgb = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [3, 4, 5, 6],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__subsample": [0.7, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.9, 1.0],
    }

    random_search_xgb = RandomizedSearchCV(
        estimator=Pipeline(steps=[
            ("preprocessor", preprocessor_xgb),
            ("model", XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="auc",
                scale_pos_weight=scale_pos_weight,
            )),
        ]),
        param_distributions=param_dist_xgb,
        n_iter=10,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    train_start_xgb = time.time()
    random_search_xgb.fit(X_train, y_train)
    train_duration_xgb = time.time() - train_start_xgb

    # XGBoost Tuning-Laufzeit protokollieren
    log_custom_runtime("04_XGBoost_RandomSearch_Fit", train_duration_xgb)

    best_xgb = random_search_xgb.best_estimator_
    best_xgb_params = random_search_xgb.best_params_

    predict_start_xgb = time.time()
    y_pred_xgb = best_xgb.predict(X_test)
    y_proba_xgb = best_xgb.predict_proba(X_test)[:, 1]
    predict_duration_xgb = time.time() - predict_start_xgb

    roc_auc_xgb = roc_auc_score(y_test, y_proba_xgb)
    avg_prec_xgb = average_precision_score(y_test, y_proba_xgb)
    f1_xgb = f1_score(y_test, y_pred_xgb)
    bal_acc_xgb = balanced_accuracy_score(y_test, y_pred_xgb)
    prec_xgb = precision_score(y_test, y_pred_xgb, zero_division=0)
    rec_xgb = recall_score(y_test, y_pred_xgb, zero_division=0)
    tn_xgb, fp_xgb, fn_xgb, tp_xgb = confusion_matrix(y_test, y_pred_xgb).ravel()
else:
    best_xgb = None
    y_pred_xgb, y_proba_xgb = None, None
    train_duration_xgb, predict_duration_xgb = np.nan, np.nan
    roc_auc_xgb, avg_prec_xgb, f1_xgb = np.nan, np.nan, np.nan

# %% Teil 5: HistGradientBoosting
param_dist_hgb = {
    "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "model__max_iter": [100, 200, 300],
    "model__max_leaf_nodes": [15, 31, 63],
    "model__min_samples_leaf": [20, 50, 100],
    "model__l2_regularization": [0.0, 0.1, 1.0],
}

random_search_hgb = RandomizedSearchCV(
    estimator=Pipeline(steps=[
        ("preprocessor", preprocessor_hgb),
        ("model", HistGradientBoostingClassifier(
            categorical_features=categorical_mask,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ]),
    param_distributions=param_dist_hgb,
    n_iter=10,
    scoring="roc_auc",
    cv=cv,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1,
)

train_start_hgb = time.time()
random_search_hgb.fit(X_train, y_train)
train_duration_hgb = time.time() - train_start_hgb

# HistGradientBoosting Tuning-Laufzeit protokollieren
log_custom_runtime("04_HistGradientBoosting_Fit", train_duration_hgb)

best_hgb = random_search_hgb.best_estimator_
best_hgb_params = random_search_hgb.best_params_

predict_start_hgb = time.time()
y_pred_hgb = best_hgb.predict(X_test)
y_proba_hgb = best_hgb.predict_proba(X_test)[:, 1]
predict_duration_hgb = time.time() - predict_start_hgb

roc_auc_hgb = roc_auc_score(y_test, y_proba_hgb)
avg_prec_hgb = average_precision_score(y_test, y_proba_hgb)
f1_hgb = f1_score(y_test, y_pred_hgb)
bal_acc_hgb = balanced_accuracy_score(y_test, y_pred_hgb)
prec_hgb = precision_score(y_test, y_pred_hgb, zero_division=0)
rec_hgb = recall_score(y_test, y_pred_hgb, zero_division=0)
tn_hgb, fp_hgb, fn_hgb, tp_hgb = confusion_matrix(y_test, y_pred_hgb).ravel()

# %% Teil 6: Vergleich & Tabellen-Export
hgb_row = {
    "Modell": "HistGradientBoosting (optimiert)",
    "class_weight": "balanced",
    "n_estimators": best_hgb_params.get("model__max_iter", 200),
    "max_leaf_nodes": best_hgb_params.get("model__max_leaf_nodes", 15),
    "min_samples_leaf": best_hgb_params.get("model__min_samples_leaf", 20),
    "learning_rate": best_hgb_params.get("model__learning_rate", 0.05),
    "l2_regularization": best_hgb_params.get("model__l2_regularization", 0.1),
    "Test PR-AUC": avg_prec_hgb,
    "Test ROC-AUC": roc_auc_hgb,
    "Test Balanced Accuracy": bal_acc_hgb,
    "Test F1": f1_hgb,
    "Test Precision": prec_hgb,
    "Test Recall": rec_hgb,
    "True Negatives": tn_hgb,
    "False Positives": fp_hgb,
    "False Negatives": fn_hgb,
    "True Positives": tp_hgb,
    "Trainingszeit gesamt (s)": train_duration_hgb,
    "Vorhersagezeit Test (s)": predict_duration_hgb,
}

if HAS_XGBOOST:
    xgb_row = {
        "Modell": "XGBoost (optimiert)",
        "class_weight": f"scale_pos_weight={scale_pos_weight:.1f}",
        "n_estimators": best_xgb_params.get("model__n_estimators", 300),
        "max_depth": best_xgb_params.get("model__max_depth", 4),
        "learning_rate": best_xgb_params.get("model__learning_rate", 0.05),
        "Test PR-AUC": avg_prec_xgb,
        "Test ROC-AUC": roc_auc_xgb,
        "Test Balanced Accuracy": bal_acc_xgb,
        "Test F1": f1_xgb,
        "Test Precision": prec_xgb,
        "Test Recall": rec_xgb,
        "True Negatives": tn_xgb,
        "False Positives": fp_xgb,
        "False Negatives": fn_xgb,
        "True Positives": tp_xgb,
        "Trainingszeit gesamt (s)": train_duration_xgb,
        "Vorhersagezeit Test (s)": predict_duration_xgb,
    }
    ergebnis_tabelle = pd.DataFrame([xgb_row, hgb_row])
else:
    ergebnis_tabelle = pd.DataFrame([hgb_row])

results = ergebnis_tabelle.copy()
final_test_results = ergebnis_tabelle.copy()

print("\n=== FINALE BOOSTING ERGEBNISSE ===")
print(ergebnis_tabelle.round(4))


def main():
    pass


if __name__ == "__main__":
    with measure_runtime("04_xgboost_histgradientboosting"):
        main()