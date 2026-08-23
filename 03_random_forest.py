"""
03_random_forest.py
Modellierung: Random Forest (Finales Modell).

Dieses Skript nutzt die gemeinsamen Projektmodule:
- data_loading: Einheitlicher Datenbezug
- preprocessing: Zentrale Feature-Gruppierung und Preprocessor-Pipeline
- config: Einheitliche Konfiguration fuer Reproduzierbarkeit (RANDOM_STATE, TEST_SIZE)
- plotting: Speichern von Abbildungen unter output/figures/
- timing: Laufzeitmessung und Protokollierung
"""

import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    learning_curve,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from config import RANDOM_STATE, TEST_SIZE
from data_loading import load_data
from plotting import save_current_figure
from preprocessing import build_preprocessor, get_feature_groups
from timing import log_custom_runtime, measure_runtime

warnings.filterwarnings("ignore")

N_SPLITS = 3

# %% Teil 1: Gemeinsame Datenbasis & Basisblock
# Datenbezug ueber zentrales Modul
df_raw = load_data()
df_cleaned = df_raw.copy()

X = df_cleaned.drop(columns=["target"])
y = pd.to_numeric(df_cleaned["target"]).astype("int8")

# Gemeinsame Ermittlung der Feature-Gruppen
num_features, cat_features, bin_features = get_feature_groups(X)

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
    """Bewertet eine vollstaendige Pipeline per Cross-Validation auf den Trainingsdaten."""
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

# %% Teil 2: Hilfsfunktionen
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


def summarize_cv(scores):
    return {
        "PR-AUC": scores["test_pr_auc"].mean(),
        "PR-AUC Std": scores["test_pr_auc"].std(),
        "ROC-AUC": scores["test_roc_auc"].mean(),
        "Balanced Accuracy": scores["test_balanced_accuracy"].mean(),
        "F1": scores["test_f1"].mean(),
        "Mittlere Fit-Zeit (s)": scores["fit_time"].mean(),
    }


# %% Teil 3: Vorverarbeitungs-Pipeline (nutzt zentrales preprocessing-Modul)
preprocessor = build_preprocessor(num_features, cat_features, bin_features, random_state=RANDOM_STATE)

base_model = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=1,
)

pipeline_baseline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", base_model),
])

# %% Teil 4: Learning Curve
max_cv_train_size = int(len(X_train) * (N_SPLITS - 1) / N_SPLITS)

train_sizes = np.array([
    10_000,
    25_000,
    50_000,
    100_000,
    150_000,
    200_000,
    max_cv_train_size,
])

lc_sizes, lc_train_scores, lc_valid_scores, lc_fit_times, _ = learning_curve(
    estimator=pipeline_baseline,
    X=X_train,
    y=y_train,
    train_sizes=train_sizes,
    cv=cv,
    scoring="average_precision",
    n_jobs=-1,
    shuffle=True,
    random_state=RANDOM_STATE,
    return_times=True,
)

plt.figure(figsize=(9, 5))
plt.plot(lc_sizes, lc_train_scores.mean(axis=1), marker="o", label="Training")
plt.plot(lc_sizes, lc_valid_scores.mean(axis=1), marker="o", label="Cross-Validation")
plt.fill_between(
    lc_sizes,
    lc_valid_scores.mean(axis=1) - lc_valid_scores.std(axis=1),
    lc_valid_scores.mean(axis=1) + lc_valid_scores.std(axis=1),
    alpha=0.2,
)
plt.xlabel("Anzahl Trainingsbeobachtungen")
plt.ylabel("PR-AUC")
plt.title("Learning Curve: Random Forest")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
save_current_figure("01_learning_curve_pr_auc.png", subdir="03_random_forest")
plt.close()

# %% Teil 5: Hyperparameteroptimierung auf 50.000 Tuning-Stichprobe
TUNING_SAMPLE_SIZE = 50_000
X_tuning, y_tuning = stratified_subsample(X_train, y_train, n_samples=TUNING_SAMPLE_SIZE, random_state=RANDOM_STATE)

rf_configurations = [
    {
        "Modell": "Baseline",
        "n_estimators": 50,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced",
    },
    {
        "Modell": "Variante 1",
        "n_estimators": 50,
        "max_depth": 15,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "class_weight": "balanced",
    },
    {
        "Modell": "Variante 2",
        "n_estimators": 50,
        "max_depth": 15,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
    },
    {
        "Modell": "Variante 3",
        "n_estimators": 50,
        "max_depth": 20,
        "min_samples_split": 20,
        "min_samples_leaf": 20,
        "max_features": 0.3,
        "class_weight": "balanced_subsample",
    },
]

manual_results = []
for config in rf_configurations:
    fold_pr_auc, fold_roc_auc, fold_balanced_accuracy, fold_f1 = [], [], [], []
    fold_precision, fold_recall = [], []
    all_true, all_pred = [], []
    total_fit_time = 0
    total_predict_time = 0

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X_tuning, y_tuning), start=1):
        X_fold_train = X_tuning.iloc[train_idx]
        X_fold_valid = X_tuning.iloc[valid_idx]
        y_fold_train = y_tuning.iloc[train_idx]
        y_fold_valid = y_tuning.iloc[valid_idx]

        rf_model = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_split=config["min_samples_split"],
            min_samples_leaf=config["min_samples_leaf"],
            max_features=config["max_features"],
            class_weight=config["class_weight"],
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        pipeline = Pipeline([
            ("preprocessing", clone(preprocessor)),
            ("model", rf_model),
        ])

        fit_start = time.time()
        pipeline.fit(X_fold_train, y_fold_train)
        total_fit_time += time.time() - fit_start

        predict_start = time.time()
        y_probability = pipeline.predict_proba(X_fold_valid)[:, 1]
        y_prediction = (y_probability >= 0.5).astype(int)
        total_predict_time += time.time() - predict_start

        fold_pr_auc.append(average_precision_score(y_fold_valid, y_probability))
        fold_roc_auc.append(roc_auc_score(y_fold_valid, y_probability))
        fold_balanced_accuracy.append(balanced_accuracy_score(y_fold_valid, y_prediction))
        fold_f1.append(f1_score(y_fold_valid, y_prediction, zero_division=0))
        fold_precision.append(precision_score(y_fold_valid, y_prediction, zero_division=0))
        fold_recall.append(recall_score(y_fold_valid, y_prediction, zero_division=0))
        all_true.extend(y_fold_valid.to_numpy())
        all_pred.extend(y_prediction)

    tn, fp, fn, tp = confusion_matrix(all_true, all_pred).ravel()

    manual_results.append({
        "Modell": config["Modell"],
        "n_estimators": config["n_estimators"],
        "max_depth": config["max_depth"],
        "min_samples_split": config["min_samples_split"],
        "min_samples_leaf": config["min_samples_leaf"],
        "max_features": config["max_features"],
        "class_weight": config["class_weight"],
        "PR-AUC": np.mean(fold_pr_auc),
        "PR-AUC Std": np.std(fold_pr_auc),
        "ROC-AUC": np.mean(fold_roc_auc),
        "Balanced Accuracy": np.mean(fold_balanced_accuracy),
        "F1": np.mean(fold_f1),
        "Precision": np.mean(fold_precision),
        "Recall": np.mean(fold_recall),
        "True Negatives": tn,
        "False Positives": fp,
        "False Negatives": fn,
        "True Positives": tp,
        "Trainingszeit gesamt (s)": total_fit_time,
        "CV-Vorhersagezeit gesamt (s)": total_predict_time,
    })

manual_results_df = pd.DataFrame(manual_results).sort_values("PR-AUC", ascending=False).reset_index(drop=True)
print("\n=== Hyperparameter-Vergleich auf 50k Tuning-Stichprobe ===")
print(manual_results_df.round(4))

# %% Teil 6: Validierung der besten Variante (Variante 3) auf 300k Stichprobe
SEARCH_SAMPLE_SIZE = 300_000
X_search, y_search = stratified_subsample(X_train, y_train, n_samples=SEARCH_SAMPLE_SIZE, random_state=RANDOM_STATE)

variant3_pipeline = Pipeline([
    ("preprocessing", clone(preprocessor)),
    ("model", RandomForestClassifier(
        n_estimators=50,
        max_depth=20,
        min_samples_split=20,
        min_samples_leaf=20,
        max_features=0.3,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )),
])

# %% Teil 7: Evaluation auf den Testdaten (Baseline vs. Variante 3)

# 7.1 Baseline Test-Evaluation
baseline_test_pipeline = Pipeline([
    ("preprocessing", clone(preprocessor)),
    ("model", RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )),
])

fit_start_base = time.time()
baseline_test_pipeline.fit(X_train, y_train)
baseline_fit_time = time.time() - fit_start_base

# Baseline Fit-Zeit loggen
log_custom_runtime("03_RandomForest_Baseline_Fit", baseline_fit_time)

predict_start_base = time.time()
baseline_test_proba = baseline_test_pipeline.predict_proba(X_test)[:, 1]
baseline_test_pred = (baseline_test_proba >= 0.5).astype(int)
baseline_predict_time = time.time() - predict_start_base

base_tn, base_fp, base_fn, base_tp = confusion_matrix(y_test, baseline_test_pred).ravel()

baseline_test_results = pd.DataFrame([{
    "Modell": "Random Forest (Baseline/Optimiert)",
    "Datengrundlage": "Testdaten",
    "Threshold": 0.50,
    "n_estimators": 50,
    "max_depth": 10,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "Test PR-AUC": average_precision_score(y_test, baseline_test_proba),
    "Test ROC-AUC": roc_auc_score(y_test, baseline_test_proba),
    "Test Balanced Accuracy": balanced_accuracy_score(y_test, baseline_test_pred),
    "Test F1": f1_score(y_test, baseline_test_pred, zero_division=0),
    "Test Precision": precision_score(y_test, baseline_test_pred, zero_division=0),
    "Test Recall": recall_score(y_test, baseline_test_pred, zero_division=0),
    "True Negatives": base_tn,
    "False Positives": base_fp,
    "False Negatives": base_fn,
    "True Positives": base_tp,
    "Trainingszeit gesamt (s)": baseline_fit_time,
    "Vorhersagezeit Test (s)": baseline_predict_time,
}])

# 7.2 Variante 3 Test-Evaluation
variant3_test_pipeline = clone(variant3_pipeline)

fit_start_v3 = time.time()
variant3_test_pipeline.fit(X_train, y_train)
variant3_fit_time = time.time() - fit_start_v3

# Variante 3 Fit-Zeit loggen
log_custom_runtime("03_RandomForest_Variante3_Fit", variant3_fit_time)

predict_start_v3 = time.time()
variant3_test_proba = variant3_test_pipeline.predict_proba(X_test)[:, 1]
variant3_test_pred = (variant3_test_proba >= 0.5).astype(int)
variant3_predict_time = time.time() - predict_start_v3

v3_tn, v3_fp, v3_fn, v3_tp = confusion_matrix(y_test, variant3_test_pred).ravel()

variant3_test_results = pd.DataFrame([{
    "Modell": "Variante 3",
    "Datengrundlage": "Testdaten",
    "Threshold": 0.50,
    "n_estimators": 50,
    "max_depth": 20,
    "min_samples_split": 20,
    "min_samples_leaf": 20,
    "max_features": 0.3,
    "class_weight": "balanced_subsample",
    "Test PR-AUC": average_precision_score(y_test, variant3_test_proba),
    "Test ROC-AUC": roc_auc_score(y_test, variant3_test_proba),
    "Test Balanced Accuracy": balanced_accuracy_score(y_test, variant3_test_pred),
    "Test F1": f1_score(y_test, variant3_test_pred, zero_division=0),
    "Test Precision": precision_score(y_test, variant3_test_pred, zero_division=0),
    "Test Recall": recall_score(y_test, variant3_test_pred, zero_division=0),
    "True Negatives": v3_tn,
    "False Positives": v3_fp,
    "False Negatives": v3_fn,
    "True Positives": v3_tp,
    "Trainingszeit gesamt (s)": variant3_fit_time,
    "Vorhersagezeit Test (s)": variant3_predict_time,
}])

model_comparison = pd.concat([baseline_test_results, variant3_test_results], ignore_index=True)
print("\n=== Testdaten-Vergleich: Baseline vs. Variante 3 ===")
print(model_comparison.round(4))

# %% Teil 8: Finale Ergebnistabelle für Gesamtauswertung (06_)
final_results = baseline_test_results.copy()
results = final_results.copy()
final_test_results = final_results.copy()
ergebnis_tabelle = final_results.copy()

print("\n=== FINAL RANDOM FOREST ERGEBNIS ===")
print(final_results.round(4))


def main():
    pass


if __name__ == "__main__":
    with measure_runtime("03_random_forest"):
        main()
