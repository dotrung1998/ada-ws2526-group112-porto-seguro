# %% [markdown]
# # Teil 3 – LinearSVC, Learning Curve und Dimensionsreduktion
#
# Dieses Skript bearbeitet im Rahmen der Arbeitsteilung den Schwerpunkt
# **Datenvorverarbeitung und Dimensionsreduktion** mit einem linearen
# Support-Vector-Classifier (`LinearSVC`) auf dem Porto-Seguro-Datensatz.
# Untersucht werden:
#
# 1. die sinnvolle Trainingsmenge anhand einer Learning Curve,
# 2. PCA, SelectKBest und TruncatedSVD zur Dimensionsreduktion,
# 3. PR-AUC als Hauptmetrik sowie ROC-AUC, Balanced Accuracy und F1 als
#    ergänzende Metriken,
# 4. die Hyperparameter `C` und `class_weight`,
# 5. die einmalige Abschlussbewertung auf dem Hold-out-Testdatensatz.
#
# Die positive Zielklasse ist mit etwa 3,64 % stark unterrepräsentiert.
# Deshalb ist PR-AUC für die Modellauswahl aussagekräftiger als die gewöhnliche
# Accuracy: Die Metrik konzentriert sich auf die Erkennung der seltenen
# positiven Fälle und berücksichtigt dabei sowohl Precision als auch Recall.
#
# ## Gemeinsamer Basisblock
#
# Der Porto-Seguro-Datensatz wird geladen und der Missing-Value-Code `-1` durch
# `NaN` ersetzt. Danach werden kategoriale, binäre und numerische Merkmale anhand
# ihrer Namensendungen getrennt. Der stratifizierte 80/20-Split erzeugt einen
# Trainings- und einen bis zur finalen Bewertung unberührten Testdatensatz.
# Eine dreifache stratifizierte Cross-Validation sorgt auch in den Folds für
# eine vergleichbare Klassenverteilung.
#
# Die Einzelgrafiken werden über Funktionen aus `plotting.py` erzeugt.
# Die Laufzeitprotokollierung ist in `timing.py` ausgelagert.

# %%
# GEMEINSAMER BASISBLOCK

import time
import warnings
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
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
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    learning_curve,
    train_test_split,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

from config import RANDOM_STATE, TEST_SIZE
from data_loading import load_data
from plotting import (
    plot_svc_learning_curve,
    plot_selectkbest_results,
    plot_numeric_pca_variance,
    plot_pca_pipeline_scores,
    plot_truncatedsvd_pipeline_scores,
)
from preprocessing import get_feature_groups
from timing import log_custom_runtime, measure_runtime

N_SPLITS = 3

# Zentraler Datenbezug ueber data_loading.py
df_raw = load_data()
df_cleaned = df_raw.copy()

X = df_cleaned.drop(columns=["target"])
y = pd.to_numeric(df_cleaned["target"]).astype("int8")

# Gemeinsame Ermittlung der Feature-Gruppen aus preprocessing.py
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
    """Bewertet eine vollstaendige Pipeline via Cross-Validation auf den Trainingsdaten."""
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

# %% [markdown]
# ## 1. Zusätzliche Hilfsfunktionen
#
# Für die rechenintensiven Experimente wird später eine reproduzierbare,
# stratifizierte Teilstichprobe aus `X_train` gezogen. Dadurch bleibt der Anteil
# der seltenen positiven Klasse erhalten. `summarize_cv` fasst die verwendeten
# CV-Metriken für PCA, SelectKBest und TruncatedSVD einheitlich zusammen.

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


# %% [markdown]
# ## 2. Modellgerechte Vorverarbeitung für LinearSVC
#
# Die drei Merkmalsgruppen werden unterschiedlich verarbeitet:
#
# - Numerische Merkmale: Median-Imputation und Standardisierung, da `LinearSVC`
#   empfindlich auf unterschiedliche Größenordnungen der Merkmale reagiert.
# - Kategoriale Merkmale (`*_cat`): Imputation fehlender Kategorien und
#   One-Hot-Encoding. Kategorien mit weniger als 20 Beobachtungen werden über
#   `min_frequency=20` zusammengefasst.
# - Binäre Merkmale (`*_bin`): Imputation mit dem häufigsten Wert. Eine weitere
#   One-Hot-Codierung ist nicht erforderlich, weil diese Merkmale bereits binär
#   codiert sind.
#
# Die Baseline kombiniert die vollständige Vorverarbeitung mit einem
# `LinearSVC`. Die Klassengewichtung `"balanced"` dient zunächst als sinnvoller
# Startwert für die stark unausgeglichenen Klassen und wird später zusammen mit
# `C` systematisch optimiert.

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
    ("numeric", numeric_base, num_features),
    ("categorical", categorical_base, cat_features),
    ("binary", binary_base, bin_features),
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

# %% [markdown]
# ## 3. Learning Curve
#
# Die Baseline-Pipeline ohne Dimensionsreduktion wird mit wachsenden
# Trainingsmengen untersucht. Maßgeblich ist die CV-PR-AUC; die Trainingskurve
# zeigt ergänzend, wie groß der Abstand zwischen Training und Validierung ist.
# Die Fit-Zeit wird mitgeführt, um den zusätzlichen Leistungsgewinn gegen den
# steigenden Rechenaufwand abzuwägen.

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

# %% [markdown]
# ### Entscheidung zur Arbeitsmenge
#
# In den gespeicherten Notebook-Ergebnissen steigt die CV-PR-AUC von etwa
# 0,0478 bei 10.000 auf 0,0610 bei 150.000 Trainingsbeobachtungen pro Fold.
# Danach verbessert sie sich nur noch vergleichsweise wenig, während die
# mittlere Fit-Zeit deutlich weiter zunimmt. Daher wird für die folgenden
# Experimente ein Kompromiss von ungefähr 150.000 Trainingsbeobachtungen pro
# CV-Fold gewählt.
#
# Bei dreifacher Cross-Validation werden pro Fold zwei Drittel der bereitgestellten
# Beobachtungen zum Training verwendet. Eine stratifizierte Experimentstichprobe
# von 225.000 Beobachtungen ergibt somit rund 150.000 Trainingsbeobachtungen je
# Fold (`225.000 × 2/3 = 150.000`). Das finale Modell wird nach abgeschlossener
# Auswahl wieder auf dem vollständigen Trainingsdatensatz trainiert.

# %%
X_experiment, y_experiment = stratified_subsample(
    X_train,
    y_train,
    n_samples=225_000,
)

print("Experimentelle Trainingsmenge:", X_experiment.shape)
print("Anteil positive Klasse:", round(y_experiment.mean(), 4))

# %% [markdown]
# ## 4. Dimensionsreduktion mit SelectKBest
#
# `SelectKBest(f_classif)` bewertet jedes nach der vollständigen Vorverarbeitung
# entstandene Merkmal einzeln anhand seines Zusammenhangs mit der Zielvariable.
# Numerische, binäre und One-Hot-codierte kategoriale Merkmale werden dabei
# gemeinsam berücksichtigt.
#
# Da `f_classif` die Zielvariable verwendet, liegt die Merkmalsauswahl innerhalb
# der Pipeline. Sie wird in jedem CV-Durchlauf ausschließlich auf dem jeweiligen
# Trainingsfold angepasst, dadurch wird Data Leakage vermieden. Zunächst wird
# die nach der Vorverarbeitung vorhandene Merkmalszahl bestimmt, damit die
# unreduzierte Variante als Referenz in den Kandidatenvergleich eingeht.

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

# %% [markdown]
# ### Ergebnis von SelectKBest
#
# Die Vorverarbeitung erzeugt in den gespeicherten Ergebnissen 227 Merkmale.
# Die beste mittlere CV-PR-AUC erreicht `k = 80` mit rund 0,0626. Die Variante
# mit allen 227 Merkmalen liegt bei rund 0,0615 und benötigt deutlich mehr
# Fit-Zeit. SelectKBest reduziert die Dimensionen somit erheblich, ohne die
# Modellleistung zu verschlechtern; in diesem Vergleich verbessert sie die
# Hauptmetrik sogar leicht.
#
# ## 5. Dimensionsreduktion mit PCA im numerischen Block
#
# PCA ist ein unüberwachtes Verfahren: Es maximiert die erklärte Varianz, nicht
# unmittelbar die Trennleistung der Zielklassen. Deshalb wird zunächst geprüft,
# wie viele Komponenten einen hohen Anteil der Varianz der numerischen Merkmale
# erklären. Die daraus abgeleiteten Kandidaten werden anschließend innerhalb
# der vollständigen `LinearSVC`-Pipeline per Cross-Validation verglichen.
#
# PCA wird nur auf den imputierten und standardisierten numerischen Block
# angewendet. Die binären und One-Hot-codierten kategorialen Merkmale bleiben
# unverändert. Eine klassische PCA auf der vollständigen One-Hot-Matrix würde
# deren Sparse-Vorteil weitgehend aufheben und bei diesem großen Datensatz den
# Speicherbedarf deutlich erhöhen.

# %%
numeric_probe = clone(numeric_base)
X_experiment_numeric = X_experiment[num_features]

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

# %% [markdown]
# ### Auswahl und Bewertung der PCA-Kandidaten
#
# Die Schwellen von 80 %, 90 %, 95 % und 99 % erklärter Varianz dienen nur zur
# Bildung plausibler Kandidaten. In den gespeicherten Ergebnissen entsprechen
# sie 18, 21, 23 und 25 Komponenten. Für die eigentliche Modellauswahl zählt
# anschließend die mittlere CV-PR-AUC und nicht automatisch die höchste
# erklärte Varianz.
#
# Mit 21 Komponenten erreicht PCA die beste CV-PR-AUC dieser Kandidaten von
# rund 0,0617. Da PCA nur den numerischen Block reduziert, ist die Reduktion der
# gesamten transformierten Merkmalsmatrix geringer als bei SelectKBest.

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
        ("numeric", numeric_pca, num_features),
        ("categorical", clone(categorical_base), cat_features),
        ("binary", clone(binary_base), bin_features),
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

# %% [markdown]
# ## 6. TruncatedSVD
#
# `TruncatedSVD` kann im Unterschied zur hier verwendeten klassischen PCA direkt
# auf der vollständigen sparse transformierten Matrix arbeiten. Numerische,
# binäre und One-Hot-codierte kategoriale Merkmale gehen daher gemeinsam in die
# Reduktion ein.
#
# Entscheidend ist, ob TruncatedSVD mit deutlich weniger Komponenten eine
# vergleichbare CV-PR-AUC wie die unreduzierte Variante erreicht. Wenn eine gute
# Leistung erst nahe der ursprünglichen Merkmalsanzahl erreicht wird oder der
# Rechenaufwand stark steigt, bietet das Verfahren für diese Aufgabe keinen
# überzeugenden Reduktionsvorteil.

# %%
svd_results = []
candidate_svd_components = [10, 20, 40, 60, 80, 120, n_transformed_features]

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

# %% [markdown]
# ### Vergleich und Bewertung der Dimensionsreduktionsverfahren
#
# Nach der Vorverarbeitung liegen **227 Merkmale** vor. Die CV-Ergebnisse zeigen:
#
# - **SelectKBest** erzielt das beste Ergebnis bei `k = 80` und reduziert die Merkmalszahl deutlich.
# - **PCA** erreicht ihr bestes Ergebnis mit 21 Komponenten im numerischen Block. Da kategoriale und binäre Merkmale unverändert bleiben, ist die Gesamtreduktion geringer.
# - **TruncatedSVD** erreicht eine vergleichbare Leistung erst bei einer hohen Komponentenanzahl und benötigt dabei deutlich mehr Rechenzeit.
#
# TruncatedSVD bietet somit keinen überzeugenden Kompromiss aus Reduktion, Modellleistung und Rechenaufwand. **SelectKBest erzielt insgesamt das beste Verhältnis**, während PCA als weitere sinnvolle Variante erhalten bleibt.
#
# TruncatedSVD wird daher nicht weiter optimiert. Verglichen werden anschließend:
#
# 1. `LinearSVC` ohne Dimensionsreduktion,
# 2. `LinearSVC` mit SelectKBest,
# 3. `LinearSVC` mit PCA.
#
# ## 7. Variantenauswahl und Hyperparameteroptimierung des LinearSVC
#
# Nach dem Vergleich werden die besten Zeilen von SelectKBest und PCA
# anhand der mittleren CV-PR-AUC bestimmt und ihre Dimensionsparameter
# festgehalten. In der folgenden Optimierung werden nur noch `C` und
# `class_weight` variiert. Es werden direkt die drei benötigten Tuning-Pipelines
# definiert. Auswahl und Optimierung beruhen ausschließlich
# auf Trainingsdaten; der Hold-out-Testdatensatz bleibt weiterhin unberührt.

# %%
best_k_row = select_k_results.loc[select_k_results["PR-AUC"].idxmax()]
best_pca_row = pca_results.loc[pca_results["PR-AUC"].idxmax()]

BEST_K = int(best_k_row["k"])
BEST_PCA_COMPONENTS = int(best_pca_row["PCA-Komponenten"])

print("Bestes SelectKBest-k:", BEST_K)
print("Beste PCA-Komponentenzahl:", BEST_PCA_COMPONENTS)

# %% [markdown]
# ### Tuning-Pipelines und Suchraum
#
# `BEST_K` und `BEST_PCA_COMPONENTS` bleiben während der Optimierung fest.
# Variiert werden:
#
# - `C`: steuert den Kompromiss zwischen Regularisierung und der Bestrafung von
#   Margin-Verletzungen; geprüft werden 0,1; 1,0 und 10,0.
# - `class_weight`: bestimmt die relative Gewichtung von Fehlern der seltenen
#   positiven Klasse; geprüft werden keine Gewichtung, zwei manuelle Varianten
#   und `"balanced"`.
#
# Derselbe Suchraum wird auf die Baseline, SelectKBest und PCA angewendet. Die
# Auswahl erfolgt nach mittlerer CV-PR-AUC auf `X_experiment`; ROC-AUC, Balanced
# Accuracy und F1 werden zur Einordnung mitgeführt. Bei 12 Kombinationen je
# Pipeline, drei Pipelines und dreifacher CV umfasst die Suche insgesamt 108 Fits.

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

# %%
numeric_pca_best = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("pca", PCA(
        n_components=BEST_PCA_COMPONENTS,
        random_state=RANDOM_STATE,
    )),
])

preprocessor_pca_best = ColumnTransformer([
    ("numeric", numeric_pca_best, num_features),
    ("categorical", clone(categorical_base), cat_features),
    ("binary", clone(binary_base), bin_features),
])

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
        scoring=SCORING,
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

# %% [markdown]
# ### Auswahl der optimierten Pipeline
#
# Die drei optimierten Varianten werden primär anhand der mittleren CV-PR-AUC
# verglichen. Die Standardabweichung hilft dabei, kleine Unterschiede nicht zu
# überinterpretieren; zusätzlich werden Reduktionsgrad und Rechenzeit betrachtet.
#
# In den gespeicherten Ergebnissen erzielt SelectKBest mit `k = 80`, `C = 0.1`
# und `class_weight = {0: 1, 1: 10}` die höchste mittlere CV-PR-AUC von rund
# 0,0631. Die PCA-Variante erreicht rund 0,0624, die unreduzierte Variante rund
# 0,0623. Daher wird die optimierte SelectKBest-Pipeline für die finale
# Bewertung ausgewählt.

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

# %% [markdown]
# ## 8. Finale Bewertung auf dem Hold-out-Testdatensatz
#
# Erst nach Abschluss von Learning Curve, Dimensionsreduktion und
# Hyperparameteroptimierung wird die ausgewählte Pipeline auf dem vollständigen
# Trainingsdatensatz neu trainiert. Anschließend wird sie genau einmal auf dem
# zuvor nicht für die Modellauswahl verwendeten Hold-out-Testdatensatz bewertet.
# Nach dieser Auswertung werden keine Hyperparameter anhand des Testdatensatzes
# angepasst.

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

# %% [markdown]
# PR-AUC und ROC-AUC werden aus den kontinuierlichen
# `decision_function`-Scores berechnet, weil `LinearSVC` keine
# `predict_proba`-Wahrscheinlichkeiten bereitstellt. Balanced Accuracy, F1,
# Precision, Recall und die Confusion Matrix basieren dagegen auf den finalen
# Klassenentscheidungen des Modells.

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

# %% [markdown]
# ### Interpretation und Weitergabe an die Gruppe
#
# Die gespeicherten Testergebnisse zeigen eine PR-AUC von rund 0,0649. Dieser
# Wert liegt über dem Anteil der positiven Klasse von etwa 0,0364 und zeigt,
# dass das Modell positive Fälle besser als eine zufällige Rangfolge priorisiert.
# Die ROC-AUC von rund 0,6316 weist ebenfalls auf vorhandene, aber insgesamt nur
# begrenzte Trennschärfe hin.
#
# Beim standardmäßigen Entscheidungsschwellenwert erreicht das Modell eine
# Precision von rund 0,1237, aber nur einen Recall von rund 0,0452: Von 4.339
# positiven Testfällen werden 196 erkannt und 4.143 übersehen. Die Balanced
# Accuracy von rund 0,5165 und der F1-Wert von rund 0,0662 bestätigen, dass die
# harten Klassenentscheidungen trotz der gegenüber der Grundrate verbesserten
# Rangfolge nur wenige positive Fälle erfassen. Für den vereinbarten
# Modellvergleich bleibt dennoch PR-AUC die Hauptmetrik. Eine nachträgliche
# Schwellenwertoptimierung am Testdatensatz wäre methodisch nicht zulässig.
#
# Für den gemeinsamen Modellvergleich wird `final_test_results` weitergegeben.
# Die Tabellen `learning_curve_results`, `pca_results`, `select_k_results`,
# `svd_results` und `tuning_results` dokumentieren zusätzlich die Entscheidungen
# zur Trainingsmenge, Dimensionsreduktion und Hyperparameterwahl.


def main():
    pass


if __name__ == "__main__":
    with measure_runtime("05_dimensionsreduktion_linear_svc"):
        main()
