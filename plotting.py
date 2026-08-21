"""
plotting.py
Gemeinsame Visualisierungsfunktionen fuer EDA, modellspezifische Experimente und Modellvergleich.
"""

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import FIGURES_DIR

os.makedirs(FIGURES_DIR, exist_ok=True)


def _figure_dir(subdir: str = "") -> str:
    directory = os.path.join(FIGURES_DIR, subdir) if subdir else FIGURES_DIR
    os.makedirs(directory, exist_ok=True)
    return directory


def _savefig(filename: str, dpi: int = 150, subdir: str = "") -> None:
    plt.savefig(
        os.path.join(_figure_dir(subdir), filename),
        dpi=dpi,
        bbox_inches="tight",
    )


def save_current_figure(filename: str, subdir: str = "", dpi: int = 150) -> None:
    """Speichert die aktuell aktive Abbildung in output/figures/<subdir>/."""
    _savefig(filename, dpi=dpi, subdir=subdir)


# ==========================================
# 1. EDA PLOTS (01_eda.py)
# ==========================================

def plot_target_distribution(df: pd.DataFrame, target_col: str = "target") -> None:
    """Abbildung: Klassenverteilung des Targets in Prozent."""
    verteilung = df[target_col].value_counts(normalize=True) * 100
    print("Klassenverteilung des Targets in Prozent:")
    print(verteilung)

    plt.figure(figsize=(6, 4))
    verteilung.plot(kind="bar", color=sns.color_palette("colorblind")[0])
    plt.title("Klassenverteilung des Targets")
    plt.xlabel("Target")
    plt.ylabel("Anteil in %")
    plt.tight_layout()
    _savefig("01_klassenverteilung.png", subdir="01_eda")
    plt.close()


def plot_binary_features(df: pd.DataFrame, bin_features: List[str]) -> None:
    """Abbildung: Haeufigkeitsverteilung der binaeren Merkmale."""
    ncols = 3
    nrows = int(np.ceil(len(bin_features) / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(15, 3.5 * nrows), constrained_layout=True
    )

    for col, ax in zip(bin_features, axes.ravel()):
        df[col].astype("string").fillna("Missing").value_counts().sort_index().plot.bar(
            ax=ax, color=sns.color_palette("colorblind")[0]
        )
        ax.set_title(col)
        ax.set_xlabel("")
        ax.set_ylabel("Anzahl")
        ax.tick_params(axis="x", rotation=0)

    for ax in axes.ravel()[len(bin_features):]:
        ax.set_visible(False)

    plt.suptitle("Haeufigkeitsverteilung der binaeren Merkmale", y=1.01, fontsize=16)
    _savefig("02_binaere_verteilung.png", subdir="01_eda")
    plt.close()


def plot_correlation_heatmap(df_cleaned: pd.DataFrame, target_col: str = "target") -> pd.Series:
    """Abbildung: Korrelationsmatrix der numerischen Features inkl. Zielvariable."""
    numeric_df = df_cleaned.select_dtypes("number").copy()
    if target_col not in numeric_df.columns and target_col in df_cleaned.columns:
        numeric_df[target_col] = pd.to_numeric(df_cleaned[target_col], errors="coerce")

    corr_matrix = numeric_df.corr()

    plt.figure(figsize=(14, 11))
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0, annot=False, linewidths=0.3)
    plt.title("Korrelationsmatrix der numerischen Features inkl. Zielvariable")
    plt.tight_layout()
    _savefig("03_korrelationsmatrix.png", subdir="01_eda")
    plt.close()

    target_corr = corr_matrix[target_col].drop(target_col).sort_values(
        key=abs, ascending=False
    )
    print("Top 10 Features mit staerkster absoluter Korrelation zur Zielvariable:")
    print(target_corr.head(10))

    plt.figure(figsize=(8, 5))
    target_corr.head(10).plot(kind="barh", color=sns.color_palette("colorblind")[0])
    plt.title("Top 10 Korrelationen mit der Zielvariable")
    plt.xlabel("Korrelationskoeffizient")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    _savefig("04_top_korrelationen_target.png", subdir="01_eda")
    plt.close()

    return target_corr.head(10)


# ==========================================
# 2. LOGISTISCHE REGRESSION (02_logistic_regression.py)
# ==========================================

def plot_pca_variance(explained_var_cumsum: np.ndarray) -> int:
    """Abbildung: Erklaerte Varianz je Anzahl Hauptkomponenten (PCA)."""
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(explained_var_cumsum) + 1), explained_var_cumsum, marker="o")
    plt.axhline(0.95, color="red", linestyle="--", label="95% erklaerte Varianz")
    plt.xlabel("Anzahl Hauptkomponenten")
    plt.ylabel("Kumulierte erklaerte Varianz")
    plt.title("PCA: Erklaerte Varianz je Anzahl Komponenten")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    _savefig("01_pca_varianz.png", subdir="02_logistic_regression")
    plt.close()

    n_components_95 = int(np.argmax(explained_var_cumsum >= 0.95) + 1)
    print(f"Anzahl Komponenten fuer 95% erklaerte Varianz: {n_components_95} "
          f"von {len(explained_var_cumsum)} Features")
    return n_components_95


def plot_logreg_coefficients(best_logreg, num_features, cat_features, bin_features) -> pd.DataFrame:
    """Abbildung: Top 15 einflussreichste Merkmale der optimierten Logistischen Regression."""
    try:
        feature_names = best_logreg.named_steps["columntransformer"].get_feature_names_out()
    except Exception:
        try:
            feature_names = best_logreg.named_steps["preprocessor"].get_feature_names_out()
        except Exception:
            feature_names = np.array(num_features + cat_features + bin_features)

    coefficients = best_logreg.named_steps["logisticregression"].coef_[0] if "logisticregression" in best_logreg.named_steps else best_logreg.named_steps["model"].coef_[0]
    coef_df = pd.DataFrame({"Feature": feature_names, "Koeffizient": coefficients})
    coef_df["AbsKoeffizient"] = coef_df["Koeffizient"].abs()
    coef_df = coef_df.sort_values("AbsKoeffizient", ascending=False).head(15)

    plt.figure(figsize=(8, 6))
    colors = ["#d62728" if c < 0 else "#1f77b4" for c in coef_df["Koeffizient"]]
    plt.barh(coef_df["Feature"], coef_df["Koeffizient"], color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Koeffizientenwert (Log-Odds)")
    plt.title("Top 15 einflussreichste Merkmale der optimierten Logistischen Regression")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    _savefig("02_logreg_koeffizienten.png", subdir="02_logistic_regression")
    plt.close()

    print(coef_df[["Feature", "Koeffizient"]].to_string(index=False))
    return coef_df


# ==========================================
# 3. DIMENSIONSREDUKTION & LINEAR SVC (05_dimensionsreduktion_linear_svc.py)
# ==========================================

def plot_svc_learning_curve(lc_sizes, lc_train_scores, lc_valid_scores, lc_fit_times, subdir: str = "05_dimensionsreduktion_linear_svc") -> None:
    """Erzeugt und speichert die Learning Curve sowie den Rechenaufwand."""
    plt.figure(figsize=(9, 5))
    plt.plot(lc_sizes, lc_train_scores.mean(axis=1), marker="o", label="Training")
    plt.plot(lc_sizes, lc_valid_scores.mean(axis=1), marker="o", label="Cross-Validation")
    plt.fill_between(
        lc_sizes,
        lc_valid_scores.mean(axis=1) - lc_valid_scores.std(axis=1),
        lc_valid_scores.mean(axis=1) + lc_valid_scores.std(axis=1),
        alpha=0.2,
    )
    plt.xlabel("Anzahl Trainingsbeobachtungen")
    plt.ylabel("PR-AUC")
    plt.title("Learning Curve: LinearSVC")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _savefig("01_learning_curve_pr_auc.png", subdir=subdir)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(lc_sizes, lc_fit_times.mean(axis=1), marker="o")
    plt.xlabel("Anzahl Trainingsbeobachtungen")
    plt.ylabel("Mittlere Fit-Zeit (Sekunden)")
    plt.title("Rechenaufwand in Abhängigkeit von der Trainingsgröße")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _savefig("02_learning_curve_fit_time.png", subdir=subdir)
    plt.close()


def plot_selectkbest_results(select_k_results: pd.DataFrame, n_transformed_features: int, subdir: str = "05_dimensionsreduktion_linear_svc") -> None:
    """Erzeugt und speichert den PR-AUC-Verlauf in Abhaengigkeit von k."""
    plot_k = select_k_results.sort_values("k").copy()
    baseline_pr_auc = plot_k.loc[plot_k["k"] == n_transformed_features, "PR-AUC"].iloc[0]

    plt.figure(figsize=(9, 5))
    plt.errorbar(
        plot_k["k"],
        plot_k["PR-AUC"],
        yerr=plot_k["PR-AUC Std"],
        marker="o",
        capsize=4,
        label="SelectKBest",
    )
    plt.axhline(
        baseline_pr_auc,
        linestyle="--",
        label=f"Alle transformierten Merkmale ({n_transformed_features})",
    )
    plt.xlabel("Beibehaltene Merkmale")
    plt.ylabel("CV PR-AUC")
    plt.title("SelectKBest: PR-AUC in Abhängigkeit von k")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    _savefig("03_selectkbest_pr_auc.png", subdir=subdir)
    plt.close()


def plot_numeric_pca_variance(pca_variance_results: pd.DataFrame, subdir: str = "05_dimensionsreduktion_linear_svc") -> None:
    """Erzeugt und speichert die kumulierte Varianz der PCA fuer den numerischen Block."""
    plt.figure(figsize=(9, 5))
    plt.plot(
        pca_variance_results["Komponenten"],
        pca_variance_results["Kumulierte erklärte Varianz"],
        marker="o",
    )
    for threshold in [0.80, 0.90, 0.95]:
        plt.axhline(threshold, linestyle="--", label=f"{threshold:.0%}")
    plt.xlabel("Anzahl PCA-Komponenten")
    plt.ylabel("Kumulierte erklärte Varianz")
    plt.title("PCA: erklärte Varianz des numerischen Merkmalsblocks")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    _savefig("04_pca_explained_variance.png", subdir=subdir)
    plt.close()


def plot_pca_pipeline_scores(pca_results: pd.DataFrame, baseline_pr_auc: float, subdir: str = "05_dimensionsreduktion_linear_svc") -> None:
    """Erzeugt und speichert die PR-AUC-Ergebnisse der PCA-Pipeline."""
    plt.figure(figsize=(9, 5))
    plt.errorbar(
        pca_results["PCA-Komponenten"],
        pca_results["PR-AUC"],
        yerr=pca_results["PR-AUC Std"],
        marker="o",
        capsize=4,
        label="PCA",
    )
    plt.axhline(
        baseline_pr_auc,
        linestyle="--",
        label="Alle Merkmale ohne Dimensionsreduktion",
    )
    plt.xlabel("PCA-Komponenten im numerischen Block")
    plt.ylabel("CV PR-AUC")
    plt.title("PCA: PR-AUC in Abhängigkeit von der Komponentenzahl")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    _savefig("05_pca_pr_auc.png", subdir=subdir)
    plt.close()


def plot_truncatedsvd_pipeline_scores(svd_results: pd.DataFrame, baseline_pr_auc: float, subdir: str = "05_dimensionsreduktion_linear_svc") -> None:
    """Erzeugt und speichert den Einfluss der TruncatedSVD-Komponentenzahl."""
    plt.figure(figsize=(9, 5))
    plt.errorbar(
        svd_results["SVD-Komponenten"],
        svd_results["PR-AUC"],
        yerr=svd_results["PR-AUC Std"],
        marker="o",
        capsize=4,
    )
    plt.axhline(
        baseline_pr_auc,
        linestyle="--",
        label="Alle Merkmale ohne Reduktion",
    )
    plt.xlabel("Anzahl SVD-Komponenten")
    plt.ylabel("CV PR-AUC")
    plt.title("TruncatedSVD: Einfluss der Komponentenzahl")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    _savefig("06_truncatedsvd_pr_auc.png", subdir=subdir)
    plt.close()


# ==========================================
# 4. GESAMTAUSWERTUNG (06_gesamtauswertung_und_ergebnisse.py)
# ==========================================

def plot_model_comparison(results: pd.DataFrame) -> None:
    """Abbildung: Detaillierter Metriken-Vergleich ueber ALLE Test-Merkmale."""
    df = results.copy()
    rename_dict = {
        "ROC-AUC": "Test ROC-AUC",
        "Average Precision": "Test PR-AUC",
        "PR-AUC": "Test PR-AUC",
        "F1-Score": "Test F1",
        "F1": "Test F1",
        "Balanced Accuracy": "Test Balanced Accuracy",
        "Precision": "Test Precision",
        "Recall": "Test Recall",
    }
    df = df.rename(columns=rename_dict)

    macro_metrics = [c for c in ["Test ROC-AUC", "Test Balanced Accuracy"] if c in df.columns]
    detailed_metrics = [c for c in ["Test PR-AUC", "Test F1", "Test Precision", "Test Recall"] if c in df.columns]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.5))
    palette1 = sns.color_palette("Blues_r", n_colors=len(macro_metrics))
    palette2 = sns.color_palette("Set2", n_colors=len(detailed_metrics))

    if macro_metrics:
        plot_macro = df.set_index("Modell")[macro_metrics].astype(float)
        plot_macro.plot(kind="bar", ax=ax1, color=palette1, width=0.7)
        ax1.set_title("Gesamt-Trennfaehigkeit: ROC-AUC & Balanced Accuracy", fontsize=13, fontweight="bold")
        ax1.set_ylabel("Test-Score", fontsize=11)
        ax1.set_ylim(0, 0.85)
        ax1.legend(loc="upper right", frameon=True)

    if detailed_metrics:
        plot_det = df.set_index("Modell")[detailed_metrics].astype(float)
        plot_det.plot(kind="bar", ax=ax2, color=palette2, width=0.8)
        ax2.set_title("Klassen- & Praezisionsmetriken: PR-AUC, F1, Precision, Recall", fontsize=13, fontweight="bold")
        ax2.set_ylabel("Test-Score", fontsize=11)
        ax2.set_ylim(0, max(0.7, plot_det.max().max() * 1.2 if not plot_det.empty else 0.7))
        ax2.legend(loc="upper right", frameon=True)

    def _annotate(ax):
        for p in ax.patches:
            height = p.get_height()
            if pd.notna(height) and height > 0.0001:
                ax.annotate(
                    f"{height:.3f}",
                    (p.get_x() + p.get_width() / 2.0, height),
                    ha="center", va="bottom",
                    xytext=(0, 4), textcoords="offset points",
                    fontsize=8, fontweight="bold", rotation=0,
                )

    _annotate(ax1)
    _annotate(ax2)

    for ax in (ax1, ax2):
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.tick_params(axis="x", rotation=20, labelsize=10)
        ax.set_xlabel("")

    plt.suptitle("Umfassender Modellvergleich aller Test-Merkmale (Hold-out Testset)", fontsize=16, y=1.02)
    plt.tight_layout()
    _savefig("01_modellvergleich_split_scales.png", subdir="06_model_comparison")
    plt.close()