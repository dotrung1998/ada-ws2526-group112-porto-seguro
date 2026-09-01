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

# Anteil der Daten für die abschließende Teststichprobe
TEST_SIZE = 0.2

# Anzahl der stratifizierten CV-Folds als Kompromiss aus Stabilität und Rechenzeit
N_SPLITS = 3

# ---------------------------------------------------------------------------
# Zentrale Output-Ordnerstruktur
# output/
#   figures/   -> alle Plots (.png)
#   tables/    -> Ergebnistabellen (.csv), Runtime-Log, Runtime-Summary
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
    Liefert den vollständigen Pfad für eine Ergebnistabelle.
    
    Args:
        subdir: Unterordner unter output/tables/ (z.B. "01_eda", "02_logistic_regression")
        filename: Name der Datei (z.B. "01_eda_duplikate.csv")
    
    Returns:
        Vollständiger Pfad zur Tabelle (z.B. "output/tables/01_eda/01_eda_duplikate.csv")
    """
    table_subdir = os.path.join(TABLES_DIR, subdir)
    os.makedirs(table_subdir, exist_ok=True)
    return os.path.join(table_subdir, filename)


# Beim Import von config.py direkt sicherstellen, dass die aktiven Ordner existieren.
ensure_output_dirs()
