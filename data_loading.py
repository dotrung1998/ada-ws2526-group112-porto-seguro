"""
data_loading.py
Gemeinsame Ladefunktion fuer den Porto-Seguro-Datensatz.

Alle exportierten CSV-Teildatensaetze werden automatisch unter
output/data/ gespeichert (siehe config.DATA_EXPORT_DIR).
"""

import os
import time

import pandas as pd
from sklearn.datasets import fetch_openml

from config import (
    OPENML_DATA_ID,
    CATEGORY_MAPPING,
    MAX_RETRIES,
    RETRY_DELAY_SECONDS,
    DATA_EXPORT_DIR,
    ensure_output_dirs,
)


def load_data() -> pd.DataFrame:
    """
    Laedt den Porto-Seguro-Datensatz direkt von der OpenML-Plattform
    (ID 42742) inklusive Retry-Mechanismus fuer 'Gateway Time-out'-Fehler.

    Returns
    -------
    pd.DataFrame
        Der vollstaendige Rohdatensatz (inkl. Zielvariable "target").
    """
    print(f"Lade Datensatz von OpenML herunter (ID: {OPENML_DATA_ID})... ")

    df_raw = None
    for i in range(MAX_RETRIES):
        try:
            porto_seguro = fetch_openml(
                data_id=OPENML_DATA_ID, as_frame=True, parser="auto"
            )
            df_raw = porto_seguro.frame
            print(f"Datensatz erfolgreich geladen. Form: {df_raw.shape}")
            break
        except Exception as e:
            print(f"Versuch {i + 1}/{MAX_RETRIES} fehlgeschlagen: {e}")
            if i < MAX_RETRIES - 1:
                print(f"Warte {RETRY_DELAY_SECONDS} Sekunden vor dem "
                      f"naechsten Versuch...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                print("Alle Wiederholungsversuche fehlgeschlagen. Bitte "
                      "ueberpruefen Sie Ihre Internetverbindung oder "
                      "versuchen Sie es spaeter erneut.")
                raise
    return df_raw


def export_category_csvs(df_raw: pd.DataFrame) -> None:
    """
    Zerlegt den Rohdatensatz anhand der technischen Praefixe
    (ps_ind, ps_reg, ps_car, ps_calc) in vier verstaendlich benannte
    Teil-Datensaetze und speichert diese als CSV unter output/data/ ab.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Der vollstaendige, ungefilterte Rohdatensatz.
    """
    ensure_output_dirs()

    for prefix, label in CATEGORY_MAPPING.items():
        cols = [col for col in df_raw.columns if col.startswith(prefix)]

        if cols:
            subset_df = df_raw[cols].copy()
            subset_df.columns = [
                col.replace(prefix, label) for col in subset_df.columns
            ]

            filepath = os.path.join(DATA_EXPORT_DIR, f"{label}_daten.csv")
            subset_df.to_csv(filepath, index=False)
            print(f"Datei '{filepath}' wurde mit verstaendlichen "
                  f"Spaltennamen gespeichert.")
        else:
            print(f"Keine Daten fuer Kategorie '{label}' gefunden.")


if __name__ == "__main__":
    df = load_data()
    export_category_csvs(df)
    print(df.head())
