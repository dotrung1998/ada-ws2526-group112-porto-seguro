"""
splitting.py
Gemeinsame Train-Test-Split-Logik (Abschnitt 4 in v4.ipynb sowie
Abschnitt 2.2 "Gemeinsamer Train-Test-Split" der Arbeitsstruktur).

Ausgelagert in ein eigenes Modul, damit sowohl 02_logistic_regression.py
als auch 03_random_forest.py (und ggf. weitere Modell-Skripte) exakt
denselben Split verwenden. Das Testset darf danach nicht mehr zur
Modellwahl oder Hyperparameteroptimierung verwendet werden.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from config import RANDOM_STATE, SAMPLE_FRACTION, TARGET_COL, TEST_SIZE


def make_train_test_split(df_cleaned: pd.DataFrame):
    """
    Erstellt den gemeinsamen, stratifizierten Train-Test-Split.

    Aufgrund des extremen Klassenungleichgewichts ist eine stratifizierte
    Aufteilung (stratify=y) zwingend erforderlich.

    Aus Performance-Gruenden wird zunaechst ein repraesentatives Subset
    der Daten verwendet (SAMPLE_FRACTION in config.py). Fuer die finale
    Abgabe kann SAMPLE_FRACTION auf 1.0 gesetzt werden, um den vollen
    Datensatz zu nutzen.
    """
    X = df_cleaned.drop(columns=[TARGET_COL])
    y = df_cleaned[TARGET_COL].astype(int)

    X_sample, _, y_sample, _ = train_test_split(
        X, y,
        train_size=SAMPLE_FRACTION,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_sample,
    )

    print(f"Trainings-Form: {X_train.shape}")
    print(f"Test-Form: {X_test.shape}")
    return X_train, X_test, y_train, y_test
