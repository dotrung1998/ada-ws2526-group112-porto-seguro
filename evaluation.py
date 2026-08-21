"""
evaluation.py
Gemeinsame Metriken- und Evaluationsfunktionen

Hauptmetrik: ROC-AUC
Zusatzmetrik: Average Precision
Ergaenzend: F1-Score, Classification Report
"""

from typing import Dict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)


def evaluate_model(model, X_test, y_test, label: str = "Modell") -> Dict:
    """
    Wertet ein bereits trainiertes Modell (Pipeline) auf dem Testset aus
    und gibt die gemeinsam vereinbarten Metriken zurueck.

    Parameters
    ----------
    model : sklearn-kompatibles Estimator/Pipeline mit predict/predict_proba
    X_test, y_test : Test-Feature-Matrix bzw. Ziel-Vektor
    label : str
        Bezeichnung des Modells fuer Ausgaben/Ergebnistabelle.

    Returns
    -------
    dict mit Keys: label, y_pred, y_proba, roc_auc, avg_precision, f1
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)

    print(f"--- Classification Report ({label}) ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"Average Precision Score: {avg_precision:.4f}")
    print(f"F1-Score: {f1:.4f}")

    return {
        "label": label,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "roc_auc": roc_auc,
        "avg_precision": avg_precision,
        "f1": f1,
    }


def compare_to_baseline(metric_name: str, optimized_value: float,
                         baseline_value: float) -> None:
    """Kleine Hilfsfunktion fuer den konsistenten Ausdruck von
    'Optimiert vs. Baseline'-Vergleichen."""
    print(f"Optimierter {metric_name}: {optimized_value:.4f} "
          f"(Baseline: {baseline_value:.4f})")
