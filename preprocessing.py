"""
preprocessing.py
Gemeinsame Preprocessing-Bausteine (Abschnitt 5 in v4.ipynb,
"Aufbau der Preprocessing-Pipeline").

Enthaelt:
- Funktion zur Identifikation von numerischen / kategorialen / binaeren
  Feature-Spalten anhand der ueblichen Suffix-Konvention
  (_cat, _bin, sonst numerisch).
- Funktion zum Aufbau des gemeinsamen ColumnTransformer, der von allen
  Modell-Pipelines (Logistische Regression, Random Forest, ...)
  wiederverwendet werden kann.

WICHTIG (Abgleich mit den finalen Notebooks der Gruppe):
Alle finalen Notebooks (Random Forest, XGBoost/HistGradientBoosting,
LinearSVC/Dimensionsreduktion) kodieren kategoriale Merkmale (_cat)
konsequent per OneHotEncoder. Frueher wurde hier ein TargetEncoder
verwendet, der pro kategorialem Merkmal nur eine einzige Spalte erzeugt.
Dadurch hatte die Logistische Regression (02_logistic_regression.py)
deutlich weniger Merkmale/Spalten als die uebrigen Modelle der Gruppe.
Um konsistent zu den finalen Notebooks zu sein, wird hier jetzt ebenfalls
OneHotEncoder verwendet.
"""

from typing import List, Tuple

import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_feature_groups(X: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    Teilt die Feature-Spalten anhand ihres Namens-Suffix in drei Gruppen:
    numerisch, kategorial (_cat) und binaer (_bin).

    Parameters
    ----------
    X : pd.DataFrame
        Feature-Matrix (ohne Zielvariable).

    Returns
    -------
    (num_features, cat_features, bin_features) : Tuple[List[str], List[str], List[str]]
    """
    cat_features = [col for col in X.columns if col.endswith("_cat")]
    bin_features = [col for col in X.columns if col.endswith("_bin")]
    num_features = [
        col for col in X.columns if col not in cat_features + bin_features
    ]
    return num_features, cat_features, bin_features


def build_preprocessor(
    num_features: List[str],
    cat_features: List[str],
    bin_features: List[str],
    random_state: int = 42,
):
    """
    Baut den gemeinsamen ColumnTransformer fuer die Vorverarbeitung auf.

    1. Numerische Features: Median-Imputation + StandardScaler
       (wichtig fuer gradientenbasierte Modelle wie die Logistische Regression).
    2. Kategoriale Features: Konstante Imputation ("Missing") + OneHotEncoder
       (handle_unknown="ignore"), analog zur Behandlung kategorialer Merkmale
       in den finalen Random-Forest-/XGBoost-/LinearSVC-Notebooks der Gruppe.
       So hat die Logistische Regression exakt die gleichen (bzw. vergleichbar
       viele) Merkmale/Spalten wie die anderen Modelle nach dem Preprocessing.
    3. Binaere Features: einfache Modus-Imputation.

    Parameters
    ----------
    num_features, cat_features, bin_features : List[str]
        Spaltenlisten aus get_feature_groups().
    random_state : int
        Wird nicht mehr fuer den Encoder benoetigt, bleibt aber aus
        Kompatibilitaetsgruenden zur bisherigen Funktionssignatur erhalten.

    Returns
    -------
    sklearn.compose.ColumnTransformer
    """
    numeric_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
    )

    categorical_pipeline = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="Missing"),
        OneHotEncoder(handle_unknown="ignore"),
    )

    binary_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
    )

    preprocessor = make_column_transformer(
        (numeric_pipeline, num_features),
        (categorical_pipeline, cat_features),
        (binary_pipeline, bin_features),
    )
    return preprocessor
