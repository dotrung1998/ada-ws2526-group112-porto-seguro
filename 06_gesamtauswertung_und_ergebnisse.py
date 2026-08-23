"""
06_gesamtauswertung_und_ergebnisse.py
Gemeinsame Gesamtauswertung fuer das gesamte Projekt (Porto Seguro).

Fuehrt die Ergebnisse aller 4 Modellzweige zusammen, standardisiert
alle Spaltennamen und exportiert eine einheitliche finale Ergebnistabelle sowie
die Abbildungen und Laufzeituebersichten.
"""

import os
from importlib import import_module

import pandas as pd
from config import (
    TABLES_DIR,
    ensure_output_dirs,
)
from plotting import plot_model_comparison
from results_summary import (
    merge_team_results,
    round_for_report,
    standardize_columns,
)
from timing import aggregate_runtimes, measure_runtime

def import_results(module_name: str, attribute_name: str) -> pd.DataFrame:
    """Importiert das Ergebnis-DataFrame eines Modellmoduls."""
    try:
        module = import_module(module_name)
    except Exception as exc:
        print(
            f"  [Hinweis] {module_name} konnte nicht importiert werden: {exc}"
        )
        return pd.DataFrame()

    value = getattr(module, attribute_name, None)

    if isinstance(value, pd.DataFrame) and not value.empty:
        return standardize_columns(value)

    print(
        f"  [Hinweis] In {module_name} wurde unter "
        f"'{attribute_name}' kein gültiges Ergebnis-DataFrame gefunden."
    )
    return pd.DataFrame()

def run_complete_evaluation():
    """Fuehrt die Gesamtauswertung durch."""
    ensure_output_dirs()

    print("=" * 70)
    print("SCHRITT 1: ERGEBNIS-AGGREGATION ALLER MODELLE")
    print("=" * 70)

    # 1. Logistische Regression (02)
    print("\n-> Lade Ergebnisse: 02_logistic_regression...")
    logreg_row = import_results(
        "02_logistic_regression",
        "final_results_table",
    )

    # 2. Random Forest (03)
    print("\n-> Lade Ergebnisse: 03_random_forest...")
    rf_current_results = import_results(
        "03_random_forest",
        "final_results",
    )

    # 3. XGBoost / HistGradientBoosting (04)
    print("\n-> Lade Ergebnisse: 04_xgboost_histgradientboosting...")
    xgb_hgb_results = import_results(
        "04_xgboost_histgradientboosting",
        "ergebnis_tabelle",
    )

    # 4. Dimensionsreduktion / LinearSVC (05)
    print("\n-> Lade Ergebnisse: 05_dimensionsreduktion_linear_svc...")
    dimred_results = import_results(
        "05_dimensionsreduktion_linear_svc",
        "final_test_results",
    )

    # Vor dem Zusammenfuehren sicherstellen, dass alle Modellzweige vorliegen.
    model_results = {
        "Logistische Regression": logreg_row,
        "Random Forest": rf_current_results,
        "XGBoost / HistGradientBoosting": xgb_hgb_results,
        "LinearSVC": dimred_results,
    }
    missing_models = [
        model_name
        for model_name, results in model_results.items()
        if results.empty
    ]

    if missing_models:
        raise RuntimeError(
            "Fuer folgende Modelle fehlen gueltige Ergebnisse: "
            + ", ".join(missing_models)
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