"""
01_eda.py
Explorative Datenanalyse (EDA) - entspricht Abschnitt 3 in v4.ipynb.

Alle Abbildungen landen automatisch unter output/figures/.
Die Laufzeit dieses Skripts wird automatisch in
output/tables/runtime_log.csv protokolliert (siehe timing.py).
"""

import warnings

import pandas as pd

from config import TARGET_COL
from data_loading import load_data, export_category_csvs
from preprocessing import get_feature_groups
from plotting import (
    plot_target_distribution,
    plot_binary_features,
    plot_correlation_heatmap,
)
from timing import measure_runtime

warnings.filterwarnings("ignore")


def run_eda() -> pd.DataFrame:
    # 2. Datenbezug und -inspektion
    df_raw = load_data()
    export_category_csvs(df_raw)
    print(df_raw.head())

    # 3. Explorative Datenanalyse (EDA) & Bereinigung
    # 3.1 Klassenungleichgewicht analysieren
    plot_target_distribution(df_raw, target_col=TARGET_COL)

    # Feature-Gruppen bestimmen (numerisch / kategorial / binaer)
    X_all = df_raw.drop(columns=[TARGET_COL])
    num_features, cat_features, bin_features = get_feature_groups(X_all)

    # 3.6 Binaere Merkmale untersuchen
    plot_binary_features(df_raw, bin_features)

    # 3.8 Korrelationsanalyse zur Zielvariable
    df_cleaned = df_raw.copy()
    plot_correlation_heatmap(df_cleaned, target_col=TARGET_COL)

    print(
        "Erkenntnis aus der EDA: Die Korrelationen einzelner Features mit "
        "der Zielvariable sind insgesamt schwach ausgepraegt (typisch fuer "
        "Versicherungsrisikodaten), was nahelegt, dass kein einzelnes "
        "Merkmal alleine eine starke Vorhersagekraft besitzt."
    )

    return df_cleaned


if __name__ == "__main__":
    with measure_runtime("01_eda"):
        run_eda()
