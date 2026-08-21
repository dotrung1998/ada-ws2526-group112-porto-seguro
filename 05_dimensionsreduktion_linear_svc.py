# %%
# GEMEINSAMER BASISBLOCK

import time
import warnings
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
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
    learning_curve,
    train_test_split,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import LinearSVC

from plotting import (
    plot_svc_learning_curve,
    plot_selectkbest_results,
    plot_numeric_pca_variance,
    plot_pca_pipeline_scores,
    plot_truncatedsvd_pipeline_scores,
)
from timing import log_custom_runtime, measure_runtime

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 3

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

# %%
warnings.filterwarnings(
    "ignore",
    message=".*invalid value encountered in matmul.*",
    category=RuntimeWarning,
    module="sklearn.decomposition._base",
)
warnings.filterwarnings(
    "ignore",
    message=".*divide by zero encountered in matmul.*",
    category=RuntimeWarning,
    module="sklearn.decomposition._base",
)
warnings.filterwarnings(
    "ignore",
    message=".*overflow encountered in matmul.*",
    category=RuntimeWarning,
    module="sklearn.decomposition._base",
)


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


# %%
numeric_base = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_base = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=20,
        sparse_output=True,
    )),
])

binary_base = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
])

preprocessor_full = ColumnTransformer([
    ("numeric", numeric_base, numeric_features),
    ("categorical", categorical_base, categorical_features),
    ("binary", binary_base, binary_features),
])

base_model = LinearSVC(
    C=1.0,
    class_weight="balanced",
    dual="auto",
    random_state=RANDOM_STATE,
)

pipeline_baseline = Pipeline([
    ("preprocessing", preprocessor_full),
    ("model", base_model),
])

# %%
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

learning_curve_results = pd.DataFrame({
    "Trainingsgröße": lc_sizes,
    "Train PR-AUC": lc_train_scores.mean(axis=1),
    "Train PR-AUC Std": lc_train_scores.std(axis=1),
    "CV PR-AUC": lc_valid_scores.mean(axis=1),
    "CV PR-AUC Std": lc_valid_scores.std(axis=1),
    "Mittlere Fit-Zeit (s)": lc_fit_times.mean(axis=1),
})

plot_svc_learning_curve(lc_sizes, lc_train_scores, lc_valid_scores, lc_fit_times)

# %%
X_experiment, y_experiment = stratified_subsample(
    X_train,
    y_train,
    n_samples=225_000,
)

print("Experimentelle Trainingsmenge:", X_experiment.shape)
print("Anteil positive Klasse:", round(y_experiment.mean(), 4))

# %%
preprocessor_probe = clone(preprocessor_full)
X_transformed = preprocessor_probe.fit_transform(X_experiment, y_experiment)
n_transformed_features = X_transformed.shape[1]

print("Anzahl Features vor Preprocessing:", X_experiment.shape[1])
print("Anzahl Features nach Preprocessing:", n_transformed_features)

# %%
select_k_results = []
candidate_k = [10, 20, 40, 80, 120, n_transformed_features]

for k in candidate_k:
    pipeline_k = Pipeline([
        ("preprocessing", clone(preprocessor_full)),
        ("feature_selection", SelectKBest(score_func=f_classif, k=k)),
        ("model", clone(base_model)),
    ])

    start = time.time()
    scores = cross_validate(
        pipeline_k,
        X_experiment,
        y_experiment,
        cv=cv,
        scoring=SCORING,
        n_jobs=-1,
        return_train_score=False,
    )

    select_k_results.append({
        "k": k,
        **summarize_cv(scores),
        "Gesamtzeit (s)": time.time() - start,
    })

select_k_results = pd.DataFrame(select_k_results)
plot_selectkbest_results(select_k_results, n_transformed_features)

# %%
numeric_probe = clone(numeric_base)
X_experiment_numeric = X_experiment[numeric_features]

X_numeric_prepared = numeric_probe.fit_transform(
    X_experiment_numeric,
    y_experiment,
)

pca_probe = PCA(random_state=RANDOM_STATE)
pca_probe.fit(X_numeric_prepared)

cumulative_variance = np.cumsum(pca_probe.explained_variance_ratio_)
pca_variance_results = pd.DataFrame({
    "Komponenten": np.arange(1, len(cumulative_variance) + 1),
    "Kumulierte erklärte Varianz": cumulative_variance,
})

for threshold in [0.80, 0.90, 0.95, 0.99]:
    n_comp = np.searchsorted(cumulative_variance, threshold) + 1
    print(f"{threshold:.0%} erklärte Varianz: {n_comp} Komponenten")

plot_numeric_pca_variance(pca_variance_results)

# %%
pca_candidates = sorted(set(
    min(
        len(cumulative_variance),
        np.searchsorted(cumulative_variance, threshold) + 1,
    )
    for threshold in [0.80, 0.90, 0.95, 0.99]
))

pca_results = []

for n_components in pca_candidates:
    numeric_pca = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(
            n_components=n_components,
            random_state=RANDOM_STATE,
        )),
    ])

    preprocessor_pca = ColumnTransformer([
        ("numeric", numeric_pca, numeric_features),
        ("categorical", clone(categorical_base), categorical_features),
        ("binary", clone(binary_base), binary_features),
    ])

    pipeline_pca = Pipeline([
        ("preprocessing", preprocessor_pca),
        ("model", clone(base_model)),
    ])

    start = time.time()
    scores = cross_validate(
        pipeline_pca,
        X_experiment,
        y_experiment,
        cv=cv,
        scoring=SCORING,
        n_jobs=-1,
        return_train_score=False,
    )

    pca_results.append({
        "PCA-Komponenten": n_components,
        **summarize_cv(scores),
        "Gesamtzeit (s)": time.time() - start,
    })

pca_results = pd.DataFrame(pca_results)
plot_k = select_k_results.sort_values("k").copy()
baseline_pr_auc = plot_k.loc[plot_k["k"] == n_transformed_features, "PR-AUC"].iloc[0]

plot_pca_pipeline_scores(pca_results, baseline_pr_auc)

# %%
candidate_svd_components = [
    10,
    20,
    40,
    60,
    80,
    120,
    n_transformed_features,
]
svd_results = []

for n_components in candidate_svd_components:
    pipeline_svd = Pipeline([
        ("preprocessing", clone(preprocessor_full)),
        ("dimensionality_reduction", TruncatedSVD(
            n_components=n_components,
            random_state=RANDOM_STATE,
        )),
        ("model", clone(base_model)),
    ])

    start = time.time()
    scores = cross_validate(
        pipeline_svd,
        X_experiment,
        y_experiment,
        cv=cv,
        scoring=SCORING,
        n_jobs=-1,
        return_train_score=False,
    )

    svd_results.append({
        "SVD-Komponenten": n_components,
        **summarize_cv(scores),
        "Gesamtzeit (s)": time.time() - start,
    })

svd_results = pd.DataFrame(svd_results)
plot_truncatedsvd_pipeline_scores(svd_results, baseline_pr_auc)

# %%
best_k_row = select_k_results.loc[select_k_results["PR-AUC"].idxmax()]
BEST_K = int(best_k_row["k"])

best_pca_row = pca_results.loc[pca_results["PR-AUC"].idxmax()]
BEST_PCA_COMPONENTS = int(best_pca_row["PCA-Komponenten"])

print("Bestes SelectKBest-k:", BEST_K)
print("Beste PCA-Komponentenzahl:", BEST_PCA_COMPONENTS)

# %%
pipeline_select_k = Pipeline([
    ("preprocessing", clone(preprocessor_full)),
    ("feature_selection", SelectKBest(score_func=f_classif, k=BEST_K)),
    ("model", clone(base_model)),
])

numeric_pca_best = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("pca", PCA(
        n_components=BEST_PCA_COMPONENTS,
        random_state=RANDOM_STATE,
    )),
])

preprocessor_pca_best = ColumnTransformer([
    ("numeric", numeric_pca_best, numeric_features),
    ("categorical", clone(categorical_base), categorical_features),
    ("binary", clone(binary_base), binary_features),
])

pipeline_pca_best = Pipeline([
    ("preprocessing", preprocessor_pca_best),
    ("model", clone(base_model)),
])

# %%
MODEL_PARAM_GRID = {
    "model__C": [0.1, 1.0, 10.0],
    "model__class_weight": [
        None,
        {0: 1, 1: 5},
        {0: 1, 1: 10},
        "balanced",
    ],
}

GRID_SCORING = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "balanced_accuracy": "balanced_accuracy",
    "f1": "f1",
}

# %%
pipeline_baseline_tuning = Pipeline([
    ("preprocessing", clone(preprocessor_full)),
    ("model", LinearSVC(
        dual="auto",
        random_state=RANDOM_STATE,
    )),
])

pipeline_select_k_tuning = Pipeline([
    ("preprocessing", clone(preprocessor_full)),
    ("feature_selection", SelectKBest(
        score_func=f_classif,
        k=BEST_K,
    )),
    ("model", LinearSVC(
        dual="auto",
        random_state=RANDOM_STATE,
    )),
])

pipeline_pca_tuning = Pipeline([
    ("preprocessing", clone(preprocessor_pca_best)),
    ("model", LinearSVC(
        dual="auto",
        random_state=RANDOM_STATE,
    )),
])


# %%
def tune_pipeline(name, pipeline):
    start = time.time()

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=MODEL_PARAM_GRID,
        scoring=GRID_SCORING,
        refit="pr_auc",
        cv=cv,
        n_jobs=-1,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(X_experiment, y_experiment)

    best_idx = search.best_index_
    total_time = time.time() - start
    result = {
        "Variante": name,
        "Bestes C": search.best_params_["model__C"],
        "Beste Klassengewichtung": str(
            search.best_params_["model__class_weight"]
        ),
        "CV PR-AUC": search.cv_results_["mean_test_pr_auc"][best_idx],
        "CV PR-AUC Std": search.cv_results_["std_test_pr_auc"][best_idx],
        "CV ROC-AUC": search.cv_results_["mean_test_roc_auc"][best_idx],
        "CV Balanced Accuracy": search.cv_results_[
            "mean_test_balanced_accuracy"
        ][best_idx],
        "CV F1": search.cv_results_["mean_test_f1"][best_idx],
        "Gesamtzeit (s)": total_time,
    }
    return search, result


# %%
grid_baseline, result_baseline = tune_pipeline(
    "Ohne Dimensionsreduktion",
    pipeline_baseline_tuning,
)

grid_select_k, result_select_k = tune_pipeline(
    f"SelectKBest (k={BEST_K})",
    pipeline_select_k_tuning,
)

grid_pca, result_pca = tune_pipeline(
    f"PCA (n={BEST_PCA_COMPONENTS})",
    pipeline_pca_tuning,
)

tuning_results = pd.DataFrame([
    result_baseline,
    result_select_k,
    result_pca,
]).sort_values(
    "CV PR-AUC",
    ascending=False,
).reset_index(drop=True)

# %%
searches = {
    "Ohne Dimensionsreduktion": grid_baseline,
    f"SelectKBest (k={BEST_K})": grid_select_k,
    f"PCA (n={BEST_PCA_COMPONENTS})": grid_pca,
}

BEST_VARIANT_NAME = tuning_results.loc[0, "Variante"]
best_optimized_pipeline = searches[BEST_VARIANT_NAME].best_estimator_

print("Beste optimierte Variante:", BEST_VARIANT_NAME)
print("Beste Parameter:", searches[BEST_VARIANT_NAME].best_params_)
print(
    "Beste mittlere CV-PR-AUC:",
    round(searches[BEST_VARIANT_NAME].best_score_, 6),
)

# %%
final_pipeline = clone(best_optimized_pipeline)

final_fit_start = time.time()
final_pipeline.fit(X_train, y_train)
final_fit_time = time.time() - final_fit_start

# LinearSVC Fit-Zeit loggen
log_custom_runtime("05_LinearSVC_Final_Fit", final_fit_time)

print("Finale Pipeline:", BEST_VARIANT_NAME)
print("Verwendete Trainingsdaten:", X_train.shape)
print("Verwendete Testdaten:", X_test.shape)
print("Trainingszeit auf allen Trainingsdaten (s):", round(final_fit_time, 2))

# %%
final_predict_start = time.time()
y_test_pred = final_pipeline.predict(X_test)
y_test_score = final_pipeline.decision_function(X_test)
final_predict_time = time.time() - final_predict_start

tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()

final_test_results = pd.DataFrame([{
    "Modell": BEST_VARIANT_NAME,
    "SelectKBest k": BEST_K if "SelectKBest" in BEST_VARIANT_NAME else np.nan,
    "PCA-Komponenten": BEST_PCA_COMPONENTS if "PCA" in BEST_VARIANT_NAME else np.nan,
    "C": final_pipeline.named_steps["model"].C,
    "class_weight": str(final_pipeline.named_steps["model"].class_weight),
    "Test PR-AUC": average_precision_score(y_test, y_test_score),
    "Test ROC-AUC": roc_auc_score(y_test, y_test_score),
    "Test Balanced Accuracy": balanced_accuracy_score(y_test, y_test_pred),
    "Test F1": f1_score(y_test, y_test_pred, zero_division=0),
    "Test Precision": precision_score(y_test, y_test_pred, zero_division=0),
    "Test Recall": recall_score(y_test, y_test_pred, zero_division=0),
    "True Negatives": tn,
    "False Positives": fp,
    "False Negatives": fn,
    "True Positives": tp,
    "Trainingszeit gesamt (s)": final_fit_time,
    "Vorhersagezeit Test (s)": final_predict_time,
}])

print("\n=== FINALE ERGEBNISTABELLE (LINEAR SVC) ===")
print(final_test_results.round({
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
    with measure_runtime("05_dimensionsreduktion_linear_svc"):
        main()