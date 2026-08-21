"""
06_gesamtauswertung_und_ergebnisse.py
Gemeinsame Gesamtauswertung fuer das gesamte Projekt (Porto Seguro).

Fuehrt die Ergebnisse aller 4 Modellzweige zusammen, standardisiert
alle Spaltennamen und exportiert eine einheitliche finale Ergebnistabelle sowie
die Abbildungen und Laufzeituebersichten.
"""

import os
import time
import warnings
from importlib import import_module

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from config import (
    RANDOM_STATE,
    TABLES_DIR,
    TARGET_COL,
    TEST_SIZE,
    ensure_output_dirs,
)
from data_loading import load_data
from plotting import plot_model_comparison
from preprocessing import build_preprocessor, get_feature_groups
from results_summary import (
    build_final_test_results,
    merge_team_results,
    round_for_report,
    standardize_columns,
)
from timing import aggregate_runtimes, measure_runtime

warnings.filterwarnings("ignore")

PREPROCESSING_DESC = (
    "Median-Imputation + StandardScaler (numerisch); "
    "Modus-Imputation + OneHotEncoder (kategorial); "
    "Modus-Imputation (binaer)"
)


def make_full_train_test_split(df_cleaned: pd.DataFrame):
    """Erstellt den gemeinsamen stratifizierten Split auf dem vollen Datensatz."""
    X = df_cleaned.drop(columns=[TARGET_COL])
    y = df_cleaned[TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def _import_optional_results(module_name: str, attribute_names: tuple[str, ...]) -> pd.DataFrame:
    """Importiert ein Modellmodul dynamisch und extrahiert dessen Ergebnis-DataFrame."""
    try:
        module = import_module(module_name)
    except Exception as exc:
        print(f"  [Hinweis] {module_name} konnte nicht direkt importiert werden: {exc}")
        return pd.DataFrame()

    for attribute_name in attribute_names:
        if hasattr(module, attribute_name):
            value = getattr(module, attribute_name)
            if isinstance(value, pd.DataFrame) and not value.empty:
                return standardize_columns(value)

    print(f"  [Hinweis] In {module_name} wurde kein bekanntes Ergebnis-Attribut gefunden.")
    return pd.DataFrame()


def run_logreg_fallback(preprocessor, X_train, y_train, X_test, y_test):
    """Trainiert die optimierte Logistische Regression als Fallback."""
    print("\n[Fallback] Trainiere Logistische Regression (optimiert)...")
    logreg_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            class_weight="balanced", max_iter=1000, solver="lbfgs",
            random_state=RANDOM_STATE,
        )),
    ])
    param_grid_lr = {"model__C": [0.01, 0.1, 1.0, 10.0]}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    grid_search_lr = GridSearchCV(
        estimator=logreg_pipeline,
        param_grid=param_grid_lr,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        verbose=0,
    )

    fit_start = time.time()
    grid_search_lr.fit(X_train, y_train)
    fit_time = time.time() - fit_start

    best_logreg = grid_search_lr.best_estimator_
    best_c = best_logreg.named_steps["model"].C
    best_cw = str(best_logreg.named_steps["model"].class_weight)

    return build_final_test_results(
        pipeline=best_logreg,
        model_name="Logistische Regression (optimiert)",
        X_test=X_test,
        y_test=y_test,
        fit_time=fit_time,
        cv_roc_auc=grid_search_lr.best_score_,
        best_params=grid_search_lr.best_params_,
        preprocessing_desc=PREPROCESSING_DESC,
        extra_columns={
            "C": best_c,
            "class_weight": best_cw,
        },
    )


def run_random_forest_fallback(preprocessor, X_train, y_train, X_test, y_test):
    """Trainiert Random Forest als Fallback."""
    print("\n[Fallback] Trainiere Random Forest (optimiert)...")
    rf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=50, max_depth=10, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        )),
    ])
    fit_start = time.time()
    rf_pipeline.fit(X_train, y_train)
    fit_time = time.time() - fit_start

    return build_final_test_results(
        pipeline=rf_pipeline,
        model_name="Random Forest (optimiert)",
        X_test=X_test,
        y_test=y_test,
        fit_time=fit_time,
        preprocessing_desc=PREPROCESSING_DESC,
        extra_columns={
            "n_estimators": 50,
            "max_depth": 10,
            "class_weight": "balanced",
            "max_features": "sqrt",
            "Threshold": 0.50,
        },
    )


def run_complete_evaluation():
    """Fuehrt die Gesamtauswertung durch."""
    ensure_output_dirs()

    print("=" * 70)
    print("SCHRITT 1: DATENBASIS & ERGEBNIS-AGGREGATION ALLER MODELLE")
    print("=" * 70)

    df_raw = load_data()
    df_cleaned = df_raw.copy()

    X_train, X_test, y_train, y_test = make_full_train_test_split(df_cleaned)
    num_features, cat_features, bin_features = get_feature_groups(X_train)
    preprocessor = build_preprocessor(
        num_features, cat_features, bin_features, random_state=RANDOM_STATE
    )

    # 1. Logistische Regression (02)
    print("\n-> Lade Ergebnisse: 02_logistic_regression...")
    logreg_row = _import_optional_results(
        "02_logistic_regression",
        ("final_results_table", "final_test_results", "results"),
    )
    if logreg_row.empty:
        logreg_row = run_logreg_fallback(preprocessor, X_train, y_train, X_test, y_test)

    # 2. Random Forest (03)
    print("\n-> Lade Ergebnisse: 03_random_forest...")
    rf_current_results = _import_optional_results(
        "03_random_forest",
        ("final_results", "results", "final_test_results", "final_comparison"),
    )
    if rf_current_results.empty:
        rf_current_results = run_random_forest_fallback(preprocessor, X_train, y_train, X_test, y_test)

    # 3. XGBoost / HistGradientBoosting (04)
    print("\n-> Lade Ergebnisse: 04_xgboost_histgradientboosting...")
    xgb_hgb_results = _import_optional_results(
        "04_xgboost_histgradientboosting",
        ("ergebnis_tabelle", "results", "final_results"),
    )

    # 4. Dimensionsreduktion / LinearSVC (05)
    print("\n-> Lade Ergebnisse: 05_dimensionsreduktion_linear_svc...")
    dimred_results = _import_optional_results(
        "05_dimensionsreduktion_linear_svc",
        ("final_test_results", "results", "ergebnis_tabelle"),
    )

    # 5. Zusammenfuehren der Gesamttabelle
    final_team_table = merge_team_results(
        logreg_row,
        rf_current_results,
        xgb_hgb_results,
        dimred_results,
    )

    print("\n" + "=" * 70)
    print("FINALE ERGEBNISTABELLE DER PROJEKTGRUPPE")
    print("=" * 70)
    print(final_team_table)

    # Exportieren der einzelnen finalen CSV-Tabelle
    csv_final_path = os.path.join(TABLES_DIR, "finale_ergebnistabelle.csv")
    rounded_table = round_for_report(final_team_table)
    rounded_table.to_csv(csv_final_path, index=False)
    print(f"\n[Gespeichert] Finale Ergebnistabelle: {csv_final_path}")

    # 6. Visueller Modellvergleich aller Test-Merkmale
    print("\n" + "=" * 70)
    print("SCHRITT 2: VISUELLER MODELLVERGLEICH (PLOTTING)")
    print("=" * 70)
    plot_model_comparison(final_team_table)
    print("[Gespeichert] Abbildung: output/figures/06_model_comparison/01_modellvergleich_split_scales.png")

    # 7. Aggregation und Zusammenfassung aller Laufzeiten
    print("\n" + "=" * 70)
    print("SCHRITT 3: LAUFZEIT-AGGREGATION ALLER SKRIPTE")
    print("=" * 70)
    runtime_df = aggregate_runtimes(verbose=True)

    print("\n" + "=" * 70)
    print("GESAMTAUSWERTUNG ERFOLGREICH ABGESCHLOSSEN!")
    print("=" * 70)

    return final_team_table, runtime_df


if __name__ == "__main__":
    with measure_runtime("06_gesamtauswertung"):
        run_complete_evaluation()