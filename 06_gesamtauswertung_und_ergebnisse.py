"""
06_gesamtauswertung_und_ergebnisse.py
Gemeinsame Gesamtauswertung für das gesamte Projekt (Porto Seguro).

Führt die Ergebnisse aller 4 Modellzweige zusammen, standardisiert
alle Spaltennamen und exportiert eine einheitliche finale Ergebnistabelle sowie
die Abbildungen und Laufzeitübersichten.
"""

import argparse
import os
import subprocess
import sys

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

# Zuordnung: Modellname -> (Skriptdatei, Unterordner in output/tables/, CSV-Dateiname)
MODEL_SCRIPTS = {
    "Logistische Regression": (
        "02_logistic_regression.py",
        "02_logistic_regression",
        "02_logistic_regression_ergebnisse.csv",
    ),
    "Random Forest": (
        "03_random_forest.py",
        "03_random_forest",
        "03_random_forest_ergebnisse.csv",
    ),
    "XGBoost / HistGradientBoosting": (
        "04_xgboost_histgradientboosting.py",
        "04_xgboost_histgradientboosting",
        "04_xgboost_histgradientboosting_ergebnisse.csv",
    ),
    "LinearSVC": (
        "05_dimensionsreduktion_linear_svc.py",
        "05_dimensionsreduktion_linear_svc",
        "05_dimensionsreduktion_linear_svc_ergebnisse.csv",
    ),
}


def ensure_model_results(model_name: str, script_file: str, subdir: str, csv_file: str, force_rerun: bool = False) -> str:
    """
    Stellt sicher, dass die Ergebnis-CSV eines Modells im entsprechenden
    Unterordner von output/tables/ vorhanden ist. Führt bei Bedarf (fehlende
    Datei oder force_rerun=True) das zugehörige Modellskript als eigenen
    Python-Prozess aus. Gibt den Pfad zur CSV zurück.
    """
    csv_path = os.path.join(TABLES_DIR, subdir, csv_file)

    if force_rerun or not os.path.isfile(csv_path):
        if force_rerun:
            print(f"  [Neulauf erzwungen] Starte {script_file} für '{model_name}'...")
        else:
            print(f"  [Fehlt] '{csv_path}' nicht gefunden. Starte {script_file} für '{model_name}'...")

        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_file)
        result = subprocess.run(
            [sys.executable, script_path],
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Das Skript '{script_file}' für Modell '{model_name}' ist mit "
                f"Exit-Code {result.returncode} fehlgeschlagen."
            )

        if not os.path.isfile(csv_path):
            raise RuntimeError(
                f"Nach der Ausführung von '{script_file}' wurde die erwartete "
                f"Ergebnisdatei '{csv_path}' weiterhin nicht gefunden."
            )
    else:
        print(f"  [OK] '{csv_path}' bereits vorhanden - {script_file} wird nicht erneut ausgeführt.")

    return csv_path


def load_results(csv_path: str) -> pd.DataFrame:
    """Lädt eine von einem Modellskript exportierte Ergebnistabelle."""
    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"  [Hinweis] Die Datei '{csv_path}' enthält keine Zeilen.")
        return pd.DataFrame()
    return standardize_columns(df)


def run_complete_evaluation(force_rerun: bool = False):
    """Führt die Gesamtauswertung durch und stößt bei Bedarf die Modellskripte an."""
    ensure_output_dirs()

    print("=" * 70)
    print("SCHRITT 0: SICHERSTELLEN, DASS ALLE MODELL-ERGEBNISSE VORLIEGEN")
    print("=" * 70)

    csv_paths = {}
    for model_name, (script_file, subdir, csv_file) in MODEL_SCRIPTS.items():
        print(f"\n-> Prüfe Ergebnisse: {model_name}...")
        csv_paths[model_name] = ensure_model_results(
            model_name, script_file, subdir, csv_file, force_rerun=force_rerun
        )

    print("\n" + "=" * 70)
    print("SCHRITT 1: ERGEBNIS-AGGREGATION ALLER MODELLE (aus output/tables/<modell>/)")
    print("=" * 70)

    model_results = {}
    for model_name, csv_path in csv_paths.items():
        print(f"\n-> Lade Ergebnisse: {csv_path}...")
        model_results[model_name] = load_results(csv_path)

    missing_models = [
        model_name
        for model_name, results in model_results.items()
        if results.empty
    ]

    if missing_models:
        raise RuntimeError(
            "Für folgende Modelle fehlen gültige Ergebnisse: "
            + ", ".join(missing_models)
        )

    # 2. Zusammenführen der Gesamttabelle
    final_team_table = merge_team_results(
        model_results["Logistische Regression"],
        model_results["Random Forest"],
        model_results["XGBoost / HistGradientBoosting"],
        model_results["LinearSVC"],
    )

    print("\n" + "=" * 70)
    print("FINALE ERGEBNISTABELLE DER PROJEKTGRUPPE")
    print("=" * 70)
    print(final_team_table)

    # Die finale, übergreifende Ergebnistabelle bleibt direkt in output/tables/
    # (kein Unterordner, da sie kein Einzelmodell, sondern alle vier zusammenfasst).
    csv_final_path = os.path.join(TABLES_DIR, "finale_ergebnistabelle.csv")
    rounded_table = round_for_report(final_team_table)
    rounded_table.to_csv(csv_final_path, index=False)
    print(f"\n[Gespeichert] Finale Ergebnistabelle: {csv_final_path}")

    # 3. Visueller Modellvergleich aller Test-Merkmale
    print("\n" + "=" * 70)
    print("SCHRITT 2: VISUELLER MODELLVERGLEICH (PLOTTING)")
    print("=" * 70)
    plot_model_comparison(final_team_table)
    print("[Gespeichert] Abbildung: output/figures/06_model_comparison/01_modellvergleich_split_scales.png")

    # 4. Aggregation und Zusammenfassung aller Laufzeiten
    print("\n" + "=" * 70)
    print("SCHRITT 3: LAUFZEIT-AGGREGATION ALLER SKRIPTE")
    print("=" * 70)
    runtime_df = aggregate_runtimes(verbose=True)

    print("\n" + "=" * 70)
    print("GESAMTAUSWERTUNG ERFOLGREICH ABGESCHLOSSEN!")
    print("=" * 70)

    return final_team_table, runtime_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Führt die Gesamtauswertung aus und startet bei Bedarf automatisch die Modellskripte 02-05."
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Alle Modellskripte (02-05) erneut ausführen, auch wenn bereits Ergebnis-CSVs vorliegen.",
    )
    args = parser.parse_args()

    with measure_runtime("06_gesamtauswertung"):
        run_complete_evaluation(force_rerun=args.force_rerun)
