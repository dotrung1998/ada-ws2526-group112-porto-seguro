"""
01_eda.py
Explorative Datenanalyse (EDA) - entspricht der aktualisierten EDA-Notebook-Version
(p_ADA_EDA_2.ipynb).

Deckt folgende Schritte / Abbildungen aus dem Notebook ab:
    1. Daten laden & erste Uebersicht (Shape, head, info)
    2. Zielvariable: Klassenverteilung (absolut & in Prozent)              -> Abb. 01
    3. Fehlende Werte je Spalte (absolut & in Prozent)
    4. Anzahl unterschiedlicher Werte je Spalte (gesamt & kategorial)
    5. Pruefung auf doppelte Zeilen
    6. Kategoriale Merkmale: Haeufigkeitsverteilung                        -> Abb. 05
    7. Kategoriale Merkmale im Vergleich zur Zielvariable (gestapelt, %)   -> Abb. 08
    8. Binaere Merkmale: Haeufigkeitsverteilung                            -> Abb. 02
    9. Moegliche Ausreisser bei numerischen Merkmalen (Boxplots)           -> Abb. 06
   10. Numerische Merkmale im Vergleich zur Zielvariable (Boxplots)        -> Abb. 07
   11. Korrelationsmatrix der numerischen Merkmale inkl. Zielvariable      -> Abb. 03
   12. Top-10-Korrelationen mit der Zielvariable                          -> Abb. 04

Farbschema: Alle Balken-/Boxplot-Diagramme verwenden konsistent
sns.color_palette("colorblind")[0] (Blau) als Hauptfarbe, damit das
Erscheinungsbild einheitlich ist. Bei gestapelten Diagrammen (Abb. 08)
regelt die "Target"-Legende zusaetzliche Farben je Klasse.

Alle Abbildungen landen automatisch unter output/figures/01_eda/.
Alle Ergebnistabellen landen automatisch unter output/tables/01_eda/
(analog zum Unterordner-Muster der Modellskripte 02-05).
Die Laufzeit dieses Skripts wird automatisch in
output/tables/runtime_log.csv protokolliert (siehe timing.py).
"""

import os
import warnings
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import TARGET_COL, ensure_output_dirs, get_table_path
from data_loading import load_data, export_category_csvs
from plotting import (
    plot_target_distribution,
    plot_binary_features,
    plot_correlation_heatmap,
    save_current_figure,
)
from timing import measure_runtime

warnings.filterwarnings("ignore")

# Einheitliche Hauptfarbe fuer alle Balkendiagramme in diesem Skript
# (identisch zu plot_target_distribution / plot_binary_features in plotting.py)
_PRIMARY_COLOR = sns.color_palette("colorblind")[0]

# Unterordner unter output/tables/, in dem alle Ergebnistabellen dieses
# Skripts abgelegt werden (analog zu den Modellskripten 02-05).
_TABLES_SUBDIR = "01_eda"


def _save_table(df_or_series, filename: str) -> None:
    """Speichert eine Tabelle (DataFrame/Series) als CSV unter output/tables/01_eda/."""
    ensure_output_dirs()
    filepath = get_table_path(_TABLES_SUBDIR, filename)
    df_or_series.to_csv(filepath)
    print(f"Tabelle '{filepath}' wurde gespeichert.")


def print_basic_overview(df: pd.DataFrame) -> None:
    """1. Erste Uebersicht: Shape, head() und info()."""
    print("Anzahl der Zeilen:", df.shape[0])
    print("Anzahl der Spalten:", df.shape[1])
    print("\nDaten anzeigen (head):")
    print(df.head())
    print("\nDatentypen und Non-Null-Counts (info):")
    df.info()


def analyse_target_distribution(df: pd.DataFrame, target_col: str = "target") -> pd.DataFrame:
    """2. Zielvariable: Klassenverteilung absolut & in Prozent, als Tabelle gespeichert."""
    target_summary = pd.DataFrame(
        {
            "Anzahl": df[target_col].value_counts(),
            "Anteil_in_Prozent": df[target_col].value_counts(normalize=True) * 100,
        }
    )
    print("\nKlassenverteilung der Zielvariable:")
    print(target_summary)
    _save_table(target_summary, "01_eda_zielverteilung.csv")
    return target_summary


def analyse_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """3. Fehlende Werte je Spalte (absolut & in Prozent), nur Spalten mit Luecken."""
    missing_values = df.isna().sum()
    missing_summary = pd.DataFrame(
        {
            "Anzahl_fehlend": missing_values,
            "Anteil_in_Prozent": missing_values / len(df) * 100,
            "Datentyp": df.dtypes.astype(str),
        }
    )
    missing_summary = missing_summary[missing_summary["Anzahl_fehlend"] > 0].sort_values(
        by="Anteil_in_Prozent", ascending=False
    )

    print("\nFehlende Werte je Spalte (nur Spalten mit fehlenden Werten):")
    print(missing_summary)
    print("\nSpalten mit fehlenden Werten:", len(missing_summary))
    _save_table(missing_summary, "01_eda_fehlende_werte.csv")
    return missing_summary


def analyse_unique_values(df: pd.DataFrame) -> pd.DataFrame:
    """4. Anzahl unterschiedlicher Werte je Spalte (gesamt & fuer kategoriale Spalten)."""
    category_columns = df.select_dtypes(include="category").columns
    category_unique_counts = pd.DataFrame(
        {"Anzahl unterschiedlicher Werte": df[category_columns].nunique(dropna=False)}
    ).sort_values(by="Anzahl unterschiedlicher Werte")

    print("\nAnzahl unterschiedlicher Werte je kategorialer Spalte:")
    print(category_unique_counts)
    _save_table(category_unique_counts, "01_eda_unique_werte_kategorial.csv")

    unique_summary = pd.DataFrame(
        {
            "Datentyp": df.dtypes.astype(str),
            "Anzahl unterschiedlicher Werte": df.nunique(dropna=False),
            "Fehlende Werte": df.isna().sum(),
        }
    ).sort_values(by="Anzahl unterschiedlicher Werte")

    print("\nAnzahl unterschiedlicher Werte je Spalte (alle Spalten):")
    print(unique_summary)
    _save_table(unique_summary, "01_eda_unique_werte.csv")
    return unique_summary


def check_duplicates(df: pd.DataFrame) -> int:
    """5. Pruefung auf doppelte Zeilen."""
    duplicate_count = df.duplicated().sum()
    print("\nAnzahl doppelter Zeilen:", duplicate_count)
    duplicate_summary = pd.DataFrame(
        {"Anzahl_doppelter_Zeilen": [duplicate_count]}
    )
    _save_table(duplicate_summary, "01_eda_duplikate.csv")
    return duplicate_count


def plot_categorical_features(df: pd.DataFrame, cat_features: List[str]) -> None:
    """6. Kategoriale Merkmale: Haeufigkeitsverteilung je Kategorie (Top 15 je Merkmal)."""
    ncols = 3
    nrows = int(np.ceil(len(cat_features) / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(18, 4 * nrows), constrained_layout=True
    )
    axes = np.atleast_1d(axes).ravel()

    for col, ax in zip(cat_features, axes):
        values = df[col].astype("string").fillna("Missing")
        counts = values.value_counts()
        if len(counts) > 15:
            counts = counts.head(15)
        counts = counts.sort_index()
        counts.plot.bar(ax=ax, color=_PRIMARY_COLOR)
        ax.set_title(col)
        ax.set_xlabel("Kategorie")
        ax.set_ylabel("Anzahl")
        ax.tick_params(axis="x", rotation=45)

    for ax in axes[len(cat_features):]:
        ax.set_visible(False)

    plt.suptitle("Haeufigkeitsverteilung der kategorialen Merkmale", y=1.01, fontsize=16)
    save_current_figure("05_kategoriale_verteilung.png", subdir="01_eda")
    plt.close()


def plot_categorical_vs_target(
    df: pd.DataFrame, cat_features: List[str], target_col: str = "target"
) -> None:
    """7. Kategoriale Merkmale im Vergleich zur Zielvariable (gestapelter Anteil in %)."""
    ncols = 3
    nrows = int(np.ceil(len(cat_features) / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(18, 4 * nrows), constrained_layout=True
    )
    axes = np.atleast_1d(axes).ravel()

    for col, ax in zip(cat_features, axes):
        feature_values = df[col].astype("string").fillna("Missing")
        category_counts = feature_values.value_counts()
        top_categories = category_counts.head(15).index

        mask = feature_values.isin(top_categories)
        comparison = pd.crosstab(
            feature_values[mask],
            df.loc[mask, target_col],
            normalize="index",
        ).mul(100)
        comparison = comparison.reindex(top_categories)

        comparison.plot.bar(stacked=True, ax=ax, legend=False)

        if len(category_counts) > 15:
            ax.set_title(f"{col} - Top 15")
        else:
            ax.set_title(col)
        ax.set_xlabel("Kategorie")
        ax.set_ylabel("Anteil in Prozent")
        ax.tick_params(axis="x", rotation=45)

    for ax in axes[len(cat_features):]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Target", loc="upper right")

    plt.suptitle(
        "Kategoriale Merkmale im Vergleich zur Zielvariable", y=1.01, fontsize=16
    )
    save_current_figure("08_kategoriale_merkmale_vs_target.png", subdir="01_eda")
    plt.close()


def plot_numeric_outliers(df: pd.DataFrame, num_features: List[str]) -> None:
    """9. Moegliche Ausreisser bei den numerischen Merkmalen (Boxplots je Merkmal)."""
    plot_df = df.sample(n=min(50000, len(df)), random_state=42)

    ncols = 4
    nrows = int(np.ceil(len(num_features) / ncols))

    plot_df[num_features].plot(
        kind="box",
        subplots=True,
        layout=(nrows, ncols),
        figsize=(18, 3.5 * nrows),
        sharex=False,
        sharey=False,
        color=dict(boxes=_PRIMARY_COLOR, whiskers=_PRIMARY_COLOR,
                   medians=_PRIMARY_COLOR, caps=_PRIMARY_COLOR),
    )
    plt.suptitle("Boxplots der numerischen Merkmale (Ausreisseranalyse)", y=1.01, fontsize=16)
    plt.tight_layout()
    save_current_figure("06_numerische_merkmale_ausreisser.png", subdir="01_eda")
    plt.close()


def plot_numeric_features_vs_target(
    df: pd.DataFrame, num_features: List[str], target_col: str = "target"
) -> None:
    """10. Numerische Merkmale im Vergleich zur Zielvariable (Boxplots, ohne Ausreisser)."""
    plot_df = df.copy()
    plot_df[target_col] = plot_df[target_col].astype("string")

    ncols = 3
    nrows = int(np.ceil(len(num_features) / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(18, 3.5 * nrows), constrained_layout=True
    )
    axes = np.atleast_1d(axes).ravel()

    for col, ax in zip(num_features, axes):
        sns.boxplot(
            data=plot_df, x=target_col, y=col, ax=ax, showfliers=False,
            color=_PRIMARY_COLOR,
        )
        ax.set_title(col)
        ax.set_xlabel("Target")
        ax.set_ylabel("")

    for ax in axes[len(num_features):]:
        ax.set_visible(False)

    plt.suptitle(
        "Numerische Merkmale im Vergleich zur Zielvariable", y=1.01, fontsize=16
    )
    save_current_figure("07_numerische_merkmale_vs_target.png", subdir="01_eda")
    plt.close()


def run_eda() -> pd.DataFrame:
    # 1. Datenbezug und erste Uebersicht
    df_raw = load_data()
    export_category_csvs(df_raw)
    print_basic_overview(df_raw)

    # 2. Klassenungleichgewicht der Zielvariable analysieren (Tabelle + Abbildung 01)
    analyse_target_distribution(df_raw, target_col=TARGET_COL)
    plot_target_distribution(df_raw, target_col=TARGET_COL)

    # 3. Fehlende Werte je Spalte
    analyse_missing_values(df_raw)

    # 4. Anzahl unterschiedlicher Werte je Spalte
    analyse_unique_values(df_raw)

    # 5. Pruefung auf doppelte Zeilen
    check_duplicates(df_raw)

    # Feature-Gruppen bestimmen (numerisch / kategorial / binaer)
    # Inline statt externem Modul "preprocessing" (existiert im Projekt nicht
    # mehr) - identische Logik wie in den Skripten 02-05: Trennung anhand der
    # Spaltensuffixe "_cat" (kategorial) und "_bin" (binaer); alles andere
    # gilt als numerisch.
    X_all = df_raw.drop(columns=[TARGET_COL])
    cat_features = [c for c in X_all.columns if c.endswith("_cat")]
    bin_features = [c for c in X_all.columns if c.endswith("_bin")]
    num_features = [c for c in X_all.columns if c not in cat_features + bin_features]

    # 6. Kategoriale Merkmale untersuchen (Abbildung 05)
    plot_categorical_features(df_raw, cat_features)

    # 7. Kategoriale Merkmale im Vergleich zur Zielvariable (Abbildung 08)
    plot_categorical_vs_target(df_raw, cat_features, target_col=TARGET_COL)

    # 8. Binaere Merkmale untersuchen (Abbildung 02)
    plot_binary_features(df_raw, bin_features)

    # 9. Moegliche Ausreisser bei numerischen Merkmalen (Abbildung 06)
    plot_numeric_outliers(df_raw, num_features)

    # 10. Numerische Merkmale im Vergleich zur Zielvariable (Abbildung 07)
    plot_numeric_features_vs_target(df_raw, num_features, target_col=TARGET_COL)

    # 11./12. Korrelationsanalyse (numerische Merkmale inkl. Zielvariable) -> Abb. 03 & 04
    df_cleaned = df_raw.copy()
    top_corr = plot_correlation_heatmap(df_cleaned, target_col=TARGET_COL)
    _save_table(top_corr, "01_eda_top10_korrelationen.csv")

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
