"""
timing.py
Gemeinsame Laufzeitmessung fuer alle Skripte und Einzelmodelle des Projekts.

Funktionalitaeten:
1. measure_runtime(script_name): Misst die Gesamtlaufzeit eines Skripts und haengt
   sie an output/tables/runtime_log.csv an.
2. log_custom_runtime(name, elapsed_seconds): Protokolliert manuell gemessene 
   Trainings-/Fit-Zeiten einzelner Modelle in output/tables/runtime_log.csv.
3. aggregate_runtimes(): Fasst die Laufzeiten uebersichtlich zusammen.
4. reset_runtime_log(): Loescht alte Log-Eintraege fuer einen sauberen Neustart.
"""

import csv
import os
import time
from contextlib import contextmanager
from datetime import datetime

import pandas as pd

from config import RUNTIME_LOG_PATH, RUNTIME_SUMMARY_PATH, ensure_output_dirs

# Exakt die 6 offiziellen Projektskripte
OFFICIAL_SCRIPTS = [
    "01_eda",
    "02_logistic_regression",
    "03_random_forest",
    "04_xgboost_histgradientboosting",
    "05_dimensionsreduktion_linear_svc",
    "06_gesamtauswertung",
]


def reset_runtime_log() -> None:
    """Setzt das Laufzeit-Log zurueck (loescht alte Testlaeufe)."""
    if os.path.isfile(RUNTIME_LOG_PATH):
        os.remove(RUNTIME_LOG_PATH)
        print(f"[Info] Altes Laufzeit-Log '{RUNTIME_LOG_PATH}' wurde zurueckgesetzt.")


def _log_runtime(entry_name: str, start_dt: datetime, elapsed_seconds: float) -> None:
    """Haengt einen Laufzeit-Eintrag an output/tables/runtime_log.csv an."""
    ensure_output_dirs()
    file_exists = os.path.isfile(RUNTIME_LOG_PATH)
    with open(RUNTIME_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["script", "start_time", "elapsed_seconds"])
        writer.writerow([
            entry_name,
            start_dt.isoformat(timespec="seconds"),
            f"{elapsed_seconds:.4f}",
        ])


def log_custom_runtime(name: str, elapsed_seconds: float) -> None:
    """
    Protokolliert die reine Rechenzeit eines einzelnen Modells / Fits in die CSV.
    """
    start_dt = datetime.now()
    _log_runtime(name, start_dt, elapsed_seconds)


@contextmanager
def measure_runtime(script_name: str):
    """
    Context Manager zur Messung der Gesamtlaufzeit eines Skripts oder Blocks.
    Protokolliert Start, Ende und Dauer in output/tables/runtime_log.csv.
    """
    ensure_output_dirs()
    start_dt = datetime.now()
    start = time.time()
    print(f"[{script_name}] Start: {start_dt:%Y-%m-%d %H:%M:%S}")
    try:
        yield
    finally:
        elapsed = time.time() - start
        _log_runtime(script_name, start_dt, elapsed)
        print(f"[{script_name}] Fertig in {elapsed:.2f} Sekunden ({elapsed / 60:.2f} Minuten).")


def aggregate_runtimes(only_latest: bool = True, filter_official: bool = False, verbose: bool = True) -> pd.DataFrame:
    """
    Liest das Laufzeit-Log ein und erstellt eine saubere Zusammenfassung.

    Parameters
    ----------
    only_latest : bool, default=True
        Wenn True, wird fuer jeden Eintrag nur der letzte Durchlauf gewertet.
    filter_official : bool, default=False
        Wenn True, werden ausschliesslich die 6 Hauptskripte gefiltert.
    verbose : bool, default=True
        Gibt die formatierte Tabelle auf der Konsole aus.

    Returns
    -------
    pd.DataFrame mit Spalten: script, elapsed_seconds, elapsed_minutes
    """
    ensure_output_dirs()

    if not os.path.isfile(RUNTIME_LOG_PATH):
        print("Kein Runtime-Log gefunden.")
        return pd.DataFrame(columns=["script", "elapsed_seconds", "elapsed_minutes"])

    df = pd.read_csv(RUNTIME_LOG_PATH)
    if df.empty:
        return pd.DataFrame(columns=["script", "elapsed_seconds", "elapsed_minutes"])

    df["elapsed_seconds"] = pd.to_numeric(df["elapsed_seconds"], errors="coerce").fillna(0)

    if filter_official:
        valid_df = df[df["script"].isin(OFFICIAL_SCRIPTS)].copy()
        if valid_df.empty:
            valid_df = df.copy()
    else:
        valid_df = df.copy()

    if only_latest:
        summary = (
            valid_df.groupby("script", as_index=False)
            .last()
            .sort_values("elapsed_seconds", ascending=False)
            .reset_index(drop=True)
        )
    else:
        summary = (
            valid_df.groupby("script", as_index=False)["elapsed_seconds"]
            .sum()
            .sort_values("elapsed_seconds", ascending=False)
            .reset_index(drop=True)
        )

    summary["elapsed_minutes"] = summary["elapsed_seconds"] / 60
    summary_export = summary[["script", "elapsed_seconds", "elapsed_minutes"]].copy()

    total_seconds = summary_export["elapsed_seconds"].sum()

    if verbose:
        mode_text = "Letzter Durchlauf je Eintrag" if only_latest else "Summe aller Durchlaeufe"
        print(f"\n=== Laufzeit-Zusammenfassung ({mode_text}) ===")
        print(summary_export.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        print(f"\nGesamtlaufzeit: {total_seconds:.2f} Sekunden ({total_seconds / 60:.2f} Minuten)")

    summary_export.to_csv(RUNTIME_SUMMARY_PATH, index=False)
    return summary_export


if __name__ == "__main__":
    aggregate_runtimes()