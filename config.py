"""
config.py
Zentrale Konfiguration fuer das Porto-Seguro-Projekt.
"""

import os

# OpenML Dataset-ID fuer "Porto Seguro's Safe Driver Prediction"
OPENML_DATA_ID = 42742

# Reproduzierbarkeit
RANDOM_STATE = 42

# Name der Zielvariable im Rohdatensatz
TARGET_COL = "target"

# Zuordnung der technischen Praefixe zu verstaendlichen Kategorien
CATEGORY_MAPPING = {
    "ps_ind": "individuell",
    "ps_reg": "regional",
    "ps_car": "fahrzeug",
    "ps_calc": "berechnet",
}

# Retry-Verhalten beim Herunterladen des Datensatzes von OpenML
MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 5

# Anteil der Daten, der fuer das Sub-Sampling verwendet wird
SAMPLE_FRACTION = 0.2

# Anteil Test-Split innerhalb des Splits
TEST_SIZE = 0.2

# ---------------------------------------------------------------------------
# Zentrale Output-Ordnerstruktur
# output/
#   figures/   -> alle Plots (.png), je Skript in einem eigenen Unterordner
#   tables/    -> Ergebnistabellen (.csv), je Modellskript in einem eigenen
#                 Unterordner (z.B. tables/02_logistic_regression/...csv);
#                 uebergreifende Dateien (runtime_log.csv, runtime_summary.csv,
#                 finale_ergebnistabelle.csv) bleiben direkt in tables/.
#   data/      -> exportierte Teil-Datensaetze (CSV)
# ---------------------------------------------------------------------------
OUTPUT_DIR = "output"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
DATA_EXPORT_DIR = os.path.join(OUTPUT_DIR, "data")

# Pfad zum Laufzeit-Log, in das jedes Skript seine eigene Laufzeit anhaengt
RUNTIME_LOG_PATH = os.path.join(TABLES_DIR, "runtime_log.csv")
# Pfad zur aggregierten Zusammenfassung (wird am Ende erzeugt)
RUNTIME_SUMMARY_PATH = os.path.join(TABLES_DIR, "runtime_summary.csv")


def ensure_output_dirs() -> None:
    """Legt alle benoetigten Output-Unterordner an, falls sie fehlen."""
    for d in (OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, DATA_EXPORT_DIR):
        os.makedirs(d, exist_ok=True)


def get_table_path(subdir: str, filename: str) -> str:
    """
    Gibt den vollen Pfad fuer eine Ergebnistabelle innerhalb eines
    Unterordners von output/tables/ zurueck und legt den Unterordner bei
    Bedarf an.

    Beispiel:
        get_table_path("02_logistic_regression", "02_logistic_regression_ergebnisse.csv")
        -> "output/tables/02_logistic_regression/02_logistic_regression_ergebnisse.csv"
    """
    target_dir = os.path.join(TABLES_DIR, subdir)
    os.makedirs(target_dir, exist_ok=True)
    return os.path.join(target_dir, filename)


# Beim Import von config.py direkt sicherstellen, dass die aktiven Ordner existieren.
ensure_output_dirs()
