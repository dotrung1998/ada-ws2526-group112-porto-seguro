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
5. [Pipeline-Ausführung](#-pipeline-ausführung)
6. [Ergebnisse & Metriken](#-ergebnisse--metriken)
7. [Fehlerbehebung (Troubleshooting)](#-fehlerbehebung-troubleshooting)

---

## 🔬 Projektübersicht & Architektur

Aufgrund des starken Klassenungleichgewichts (ca. 3,6 % Schadensfälle) setzt dieses Projekt auf:
- **Stratifizierte Splits** (`stratify=y`) für faire Train-Test-Validierung.
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
├── 02_logistic_regression.py                # Logistische Regression (Baseline, Tuning, Koeffizienten, Threshold)
├── 03_random_forest.py                      # Random Forest (Learning Curves, Hyperparametertuning)
├── 04_xgboost_histgradientboosting.py       # Gradient Boosting Vergleiche (XGBoost vs. HistGradientBoosting)
├── 05_dimensionsreduktion_linear_svc.py     # LinearSVC mit SelectKBest, PCA & TruncatedSVD
├── 06_gesamtauswertung_und_ergebnisse.py    # Zentrale Auswertung, Metriken-Aggregation & Gesamtplots
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
    ├── figures/                             # Exportierte Plots nach Modell getrennt (.png)
    └── tables/                              # finale_ergebnistabelle.csv, runtime_summary.csv
```

---

## ⚙️ Voraussetzungen

- **Python:** Version `3.9` bis `3.11` (empfohlen: `3.10` oder `3.11`)
- **Git** (für lokales Klonen)
- Eine aktive Internetverbindung beim ersten Ausführen (um den Datensatz automatisch von OpenML zu laden)

---

## 🚀 Schnellstart & Installation

### Option A: GitHub Codespaces (Browserbasiert)

1. Öffne das GitHub-Repository.
2. Klicke auf die grüne Schaltfläche **Code** > Reiter **Codespaces** > **Create codespace on main**.
3. Nach dem Start öffnet sich VS Code im Browser. Führe im integrierten Terminal (`Strg + ~` bzw. `Cmd + ~`) folgende Befehle aus:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

### Option B: macOS / Linux (Terminal)

1. **Repository klonen und Ordner betreten:**
   ```bash
   git clone https://github.com/<DEIN-BENUTZERNAME>/<DEIN-REPO-NAME>.git
   cd <DEIN-REPO-NAME>
   ```

2. **Virtuelle Umgebung anlegen und aktivieren:**
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
   git clone https://github.com/<DEIN-BENUTZERNAME>/<DEIN-REPO-NAME>.git
   cd <DEIN-REPO-NAME>
   ```

2. **Virtuelle Umgebung erstellen und aktivieren:**
   - In der **PowerShell**:
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
     *(Falls ein Skript-Ausführungsfehler auftritt: Einmalig `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` ausführen).*
   - In der klassischen **CMD (Eingabeaufforderung)**:
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

## ▶️ Pipeline-Ausführung

Die Skripte sind modular aufgebaut und können entweder einzeln oder als komplette Kette ausgeführt werden.

### 1. Gesamte Pipeline auf einmal ausführen

**macOS / Linux / Codespaces:**
```bash
python 01_eda.py && \
python 02_logistic_regression.py && \
python 03_random_forest.py && \
python 04_xgboost_histgradientboosting.py && \
python 05_dimensionsreduktion_linear_svc.py && \
python 06_gesamtauswertung_und_ergebnisse.py
```

**Windows (PowerShell):**
```powershell
python 01_eda.py; python 02_logistic_regression.py; python 03_random_forest.py; python 04_xgboost_histgradientboosting.py; python 05_dimensionsreduktion_linear_svc.py; python 06_gesamtauswertung_und_ergebnisse.py
```

---

### 2. Einzelne Skripte & Module

| Schritt | Befehl | Beschreibung & Generierte Outputs |
| :--- | :--- | :--- |
| **01. EDA** | `python 01_eda.py` | Lädt OpenML-Daten, exportiert Teildaten nach `output/data/` und erstellt EDA-Plots (`01_eda/`). |
| **02. LogReg** | `python 02_logistic_regression.py` | PCA-Vorabprüfung, GridSearch-Tuning, Koeffizientenanalyse & ROC/PR-Kurven (`02_logistic_regression/`). |
| **03. Random Forest** | `python 03_random_forest.py` | Learning-Curve-Analyse, Stichproben-Tuning & Holdout-Testauswertung (`03_random_forest/`). |
| **04. Boosting** | `python 04_xgboost_histgradientboosting.py` | Vergleich zwischen XGBoost (mit `scale_pos_weight`) und scikit-learns `HistGradientBoostingClassifier`. |
| **05. LinearSVC** | `python 05_dimensionsreduktion_linear_svc.py` | Feature-Selektion (`SelectKBest`), `PCA` & `TruncatedSVD` kombiniert mit Support Vector Classification. |
| **06. Evaluation** | `python 06_gesamtauswertung_und_ergebnisse.py` | Aggregiert alle Modellergebnisse in `output/tables/finale_ergebnistabelle.csv` und generiert Gesamtplots. |

---

## 📊 Ergebnisse & Metriken

Nach Abschluss der Pipeline finden sich alle Artefakte im `output/`-Verzeichnis:

1. **`output/tables/finale_ergebnistabelle.csv`:**
   Enthält den direkten Metriken-Vergleich (Test PR-AUC, Test ROC-AUC, Balanced Accuracy, F1-Score, Precision, Recall, Confusion Matrix und Rechenzeiten) über alle Modelle hinweg.
2. **`output/tables/runtime_summary.csv`:**
   Detaillierte Übersicht über die Trainings-, Fit- und Gesamtlaufzeiten der Module.
3. **`output/figures/`:**
   Hochauflösende PNG-Grafiken aller ROC-/PR-Kurven, Lernkurven, Parametervergleiche und Feature-Wichtigkeiten.

---

## 🛠️ Fehlerbehebung (Troubleshooting)

- **OpenML Gateway Timeout / Download-Fehler:**
  Das Modul `data_loading.py` verfügt über einen automatischen 5-fachen Retry-Mechanismus (`MAX_RETRIES = 5`). Sollte der Download dennoch abbrechen, prüfe deine Internetverbindung und starte das Skript erneut.
- **XGBoost unter macOS (Apple Silicon M1/M2/M3):**
  Falls `xgboost` Installationsprobleme meldet, stelle sicher, dass `libomp` via Homebrew installiert ist:
  ```bash
  brew install libomp
  ```
- **Arbeitsspeicher & Rechenzeit drosseln:**
  In `config.py` kann `SAMPLE_FRACTION = 0.2` (Standard: 20 % Stichprobe für schnelle Experimente) auf kleinere Werte angepasst oder für die finale Vollauswertung auf `1.0` gesetzt werden.
