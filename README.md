# 🚗 Porto Seguro’s Safe Driver Prediction – Machine Learning Pipeline

Dieses Repository enthält ein vollständiges End-to-End Machine-Learning-Projekt zur Vorhersage von Schadensfällen bei Kfz-Versicherungen auf Basis des Datensatzes **Porto Seguro's Safe Driver Prediction** (OpenML ID: `42742`).

Das Projekt beinhaltet explorative Datenanalyse (EDA), standardisiertes Preprocessing, Hyperparameteroptimierung für diverse Modellklassen (Logistische Regression, Random Forest, XGBoost, HistGradientBoosting, LinearSVC mit Dimensionsreduktion) sowie eine zentrale Gesamtauswertung mit automatischer Laufzeit- und Metrikerfassung.

---

## 📑 Inhaltsverzeichnis
1. [Projektübersicht & Architektur](#-projektübersicht--architektur)
2. [Projekt- & Ordnerstruktur](#-projekt--ordnerstruktur)
3. [Voraussetzungen](#-voraussetzungen)
4. [Schnellstart & Installation](#-schnellstart--installation)
   - [Option A: GitHub Codespaces (Empfohlen – 1-Klick)](#option-a-github-codespaces-browserbasiert)
   - [Option B: macOS / Linux (Terminal)](#option-b-macos--linux-terminal)
   - [Option C: Windows (PowerShell / CMD)](#option-c-windows-powershell--cmd)
5. [Pipeline-Ausführung (Empfohlener Ablauf)](#-pipeline-ausführung-empfohlener-ablauf)
6. [Ergebnisse & Metriken](#-ergebnisse--metriken)
7. [Fehlerbehebung (Troubleshooting)](#-fehlerbehebung-troubleshooting)

---

## 🔬 Projektübersicht & Architektur

Aufgrund des starken Klassenungleichgewichts (ca. 3,6 % Schadensfälle) setzt dieses Projekt auf:
- **Stratifizierte Splits** (`stratify=y`) für eine faire Validierung.
- **Primäre Evaluationsmetriken:** PR-AUC (Average Precision) und ROC-AUC.
- **Einheitliche Preprocessing-Pipeline:**
  - *Numerische Features:* Median-Imputation + `StandardScaler`.
  - *Kategoriale Features (`_cat`):* Konstante Imputation (`Missing`) + `OneHotEncoder(handle_unknown="ignore")`.
  - *Binäre Features (`_bin`):* Modus-Imputation (`most_frequent`).
- **Zentrale Konfiguration:** Alle Parameter, Seeds (`random_state=42`) und Exportpfade werden zentral über `config.py` gesteuert.

---

## 📂 Projekt- & Ordnerstruktur

```text
.
├── 01_eda.py                                # Explorative Datenanalyse & Feature-Visualisierung
├── 02_logistic_regression.py                # Einzelmodul: Logistische Regression
├── 03_random_forest.py                      # Einzelmodul: Random Forest
├── 04_xgboost_histgradientboosting.py       # Einzelmodul: Boosting (XGBoost vs. HistGradientBoosting)
├── 05_dimensionsreduktion_linear_svc.py     # Einzelmodul: LinearSVC & Dimensionsreduktion (PCA/SVD)
├── 06_gesamtauswertung_und_ergebnisse.py    # ★ HAUPTSKRIPT: Führt alle Modelle aus & aggregiert Ergebnisse
│
├── config.py                                # Zentrale Pfad- & Hyperparameter-Konfiguration
├── data_loading.py                          # Automatischer OpenML-Download & CSV-Export
├── preprocessing.py                         # Reusable ColumnTransformer & Feature-Gruppierung
├── splitting.py                             # Stratifizierte Train/Test-Split-Logik
├── evaluation.py                            # Metrik-Berechnung & Reporting-Funktionen
├── plotting.py                              # Visualisierungsmodule (PNG-Export)
├── timing.py                                # Automatisches Runtime-Logging
├── requirements.txt                         # Python-Abhängigkeiten
└── output/                                  # Automatisch generierte Ergebnisse
    ├── data/                                # Aufgeteilte CSV-Dateien (individuell, fahrzeug, etc.)
    ├── figures/                             # Exportierte Plots (.png)
    └── tables/                              # finale_ergebnistabelle.csv, runtime_summary.csv
```

---

## ⚙️ Voraussetzungen

- **Python:** Version `3.9` bis `3.11` (empfohlen: `3.10` oder `3.11`)
- **Git** (für lokales Klonen)
- Aktive Internetverbindung beim ersten Ausführen (zum automatischen Download von OpenML)

---

## 🚀 Schnellstart & Installation

### Option A: GitHub Codespaces (Browserbasiert)

1. Öffne das GitHub-Repository.
2. Klicke auf **Code** > Reiter **Codespaces** > **Create codespace on main**.
3. Nach dem Start öffnet sich VS Code im Browser. Führe im Terminal aus:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

### Option B: macOS / Linux (Terminal)

1. **Repository klonen und Ordner betreten:**
   ```bash
   git clone https://github.com/dotrung1998/ada-ws2526-group112-porto-seguro.git
   cd ada-ws2526-group112-porto-seguro
   ```

2. **Virtuelle Umgebung erstellen und aktivieren:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Pakete installieren:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

### Option C: Windows (PowerShell / CMD)

1. **Repository klonen und Ordner betreten:**
   ```powershell
   git clone https://github.com/dotrung1998/ada-ws2526-group112-porto-seguro.git
   cd ada-ws2526-group112-porto-seguro
   ```

2. **Virtuelle Umgebung erstellen und aktivieren:**
   - In der **PowerShell**:
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   - In der klassischen **CMD**:
     ```cmd
     python -m venv venv
     venv\Scripts\activate.bat
     ```

3. **Pakete installieren:**
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## ▶️ Pipeline-Ausführung (Empfohlener Ablauf)

> ⚡ **Wichtiger Hinweis zur Laufzeitersparnis:**  
> Du musst **nicht** alle Skripte einzeln nacheinander ausführen! Um doppelte Rechenzeiten zu vermeiden, besteht der reguläre Ablauf aus nur **zwei Schritten**:

### Schritt 1: Explorative Datenanalyse (EDA)
Lädt den Rohdatensatz automatisch von OpenML herunter, exportiert die Datenkategorien nach `output/data/` und erstellt alle Verteilungs- und Korrelationsgrafiken:
```bash
python 01_eda.py
```

---

### Schritt 2: Gesamtauswertung & Alle Modelle ausführen
Das Skript **`06_gesamtauswertung_und_ergebnisse.py`** ist der zentrale Orchestrator. Es lädt bzw. trainiert alle optimierten Modelle (Logistische Regression, Random Forest, XGBoost / HistGradientBoosting, LinearSVC mit Dimensionsreduktion), fasst die Metriken zusammen und erstellt die finale Vergleichstabelle sowie die Gesamtplots:
```bash
python 06_gesamtauswertung_und_ergebnisse.py
```

---

### ℹ️ Optionale Einzelausführung (Nur für Detailanalysen)

Falls du ein bestimmtes Modell separat untersuchen, detaillierte Lernkurven analysieren oder Koeffizienten prüfen möchtest, kannst du die jeweiligen Skripte auch isoliert ausführen:

| Skript | Fokus / Zweck |
| :--- | :--- |
| `python 02_logistic_regression.py` | Detaillierte Koeffizientenanalyse, PCA-Gegenprüfung & Threshold-Analyse für LogReg. |
| `python 03_random_forest.py` | Detaillierte Learning Curves & manuelle Hyperparametersuche für Random Forest. |
| `python 04_xgboost_histgradientboosting.py` | Direkter Modellvergleich & RandomSearch für Gradient-Boosting-Algorithmen. |
| `python 05_dimensionsreduktion_linear_svc.py` | Feature-Selektion (`SelectKBest`), `PCA` & `TruncatedSVD` kombiniert mit LinearSVC. |

---

## 📊 Ergebnisse & Metriken

Nach der Ausführung von Schritt 1 und 2 liegen alle Resultate im `output/`-Ordner bereit:

1. **`output/tables/finale_ergebnistabelle.csv`:**
   Enthält alle gemeinsamen Test-Metriken (PR-AUC, ROC-AUC, Balanced Accuracy, F1, Precision, Recall, Confusion Matrix) im direkten Vergleich über alle Modellklassen hinweg.
2. **`output/tables/runtime_summary.csv`:**
   Übersicht der Laufzeiten und Rechenaufwände.
3. **`output/figures/`:**
   Hochauflösende PNG-Grafiken aller ROC-/PR-Kurven, EDA-Plots und des finalen Modellvergleichs (`06_model_comparison/`).

---

## 🛠️ Fehlerbehebung (Troubleshooting)

- **OpenML Gateway Timeout / Download-Fehler:**
  `data_loading.py` verfügt über einen automatischen Retry-Mechanismus (`MAX_RETRIES = 5`). Sollte der Download dennoch abbrechen, prüfe deine Internetverbindung und führe das Skript erneut aus.
- **XGBoost unter macOS (Apple Silicon M1/M2/M3):**
  Falls bei der Installation von `xgboost` Probleme auftreten, installiere OpenMP via Homebrew:
  ```bash
  brew install libomp
  ```
- **Stichprobengröße anpassen:**
  In `config.py` steuert `SAMPLE_FRACTION = 0.2` den Anteil der Daten für schnelle Experimente (20 %). Für die vollständige Endabgabe kann dieser Wert auf `1.0` gesetzt werden.
