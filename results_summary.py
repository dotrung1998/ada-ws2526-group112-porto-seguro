"""
results_summary.py
Gemeinsame Funktion zur Erstellung und Zusammenfuehrung der finalen
Ergebnistabellen aller Projektmitglieder.

Strukturiert die Spalten so, dass alle gemeinsamen Test-Metriken direkt
vorne lueckenlos nebeneinander stehen und keine optischen Luecken durch
modellspezifische Parameter entstehen.
"""

import pandas as pd

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardisiert abweichende Spaltennamen der verschiedenen Modellskripte."""
    if df is None or df.empty:
        return pd.DataFrame()

    renamed = df.copy()
    rename_map = {
        "PR-AUC": "Test PR-AUC",
        "ROC-AUC": "Test ROC-AUC",
        "Balanced Accuracy": "Test Balanced Accuracy",
        "F1": "Test F1",
        "F1-Score": "Test F1",
        "Precision": "Test Precision",
        "Recall": "Test Recall",
        "Ausgleich Klassen": "class_weight",
    }
    for old_col, new_col in rename_map.items():
        if old_col in renamed.columns and new_col not in renamed.columns:
            renamed = renamed.rename(columns={old_col: new_col})
        elif old_col in renamed.columns and new_col in renamed.columns:
            renamed[new_col] = renamed[new_col].combine_first(renamed[old_col])
            renamed = renamed.drop(columns=[old_col])

    # Bereinigung: "class_weight=balanced" -> "balanced"
    if "class_weight" in renamed.columns:
        renamed["class_weight"] = renamed["class_weight"].astype(str).str.replace("class_weight=", "", regex=False)

    return renamed


def merge_team_results(*results_dfs: pd.DataFrame) -> pd.DataFrame:
    """
    Fuehrt die einzelnen Ergebniszeilen aller Gruppenmitglieder zusammen.
    Platzierung:
    1. Modell & Klassenausgleich
    2. Alle gemeinsamen Test-Metriken (lueckenlos fuer alle Modelle)
    3. Confusion Matrix & Laufzeiten
    4. Modellspezifische Parameter (C, Baumparameter, Reduktion)
    """
    standardized_dfs = []
    for d in results_dfs:
        if d is not None and not d.empty:
            standardized_dfs.append(standardize_columns(d))

    if not standardized_dfs:
        return pd.DataFrame()

    combined = pd.concat(standardized_dfs, ignore_index=True, sort=False)

    # Logische Reihenfolge: Gemeinsame Metriken zuerst, modellspezifische Parameter danach
    priority_cols = [
        "Modell",
        "class_weight",
        "Test PR-AUC",
        "Test ROC-AUC",
        "Test Balanced Accuracy",
        "Test F1",
        "Test Precision",
        "Test Recall",
        "True Negatives",
        "False Positives",
        "False Negatives",
        "True Positives",
        "Trainingszeit gesamt (s)",
        "Vorhersagezeit Test (s)",
        "C",
        "Threshold",
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "learning_rate",
        "max_leaf_nodes",
        "l2_regularization",
        "SelectKBest k",
        "PCA-Komponenten",
        "CV ROC-AUC",
        "Vorverarbeitung",
        "Beste Hyperparameter",
    ]

    ordered_cols = [c for c in priority_cols if c in combined.columns]
    remaining_cols = [c for c in combined.columns if c not in ordered_cols]
    final_cols = ordered_cols + remaining_cols

    return combined[final_cols]


def round_for_report(results_df: pd.DataFrame) -> pd.DataFrame:
    """Rundet Metriken uebersichtlich fuer den Bericht."""
    if results_df is None or results_df.empty:
        return results_df

    round_map = {
        "Test PR-AUC": 6,
        "Test ROC-AUC": 6,
        "Test Balanced Accuracy": 6,
        "Test F1": 6,
        "Test Precision": 6,
        "Test Recall": 6,
        "CV ROC-AUC": 6,
        "Threshold": 2,
        "Trainingszeit gesamt (s)": 2,
        "Vorhersagezeit Test (s)": 2,
    }
    existing_cols = {k: v for k, v in round_map.items() if k in results_df.columns}
    return results_df.round(existing_cols)
