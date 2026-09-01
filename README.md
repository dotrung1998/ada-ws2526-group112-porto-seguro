# Porto Seguro's Safe Driver Prediction – Machine Learning Pipeline

Dieses Repository enthält ein vollständiges End-to-End Machine-Learning-Projekt zur Vorhersage von Schadensfällen bei Kfz-Versicherungen auf Basis des Datensatzes **Porto Seguro's Safe Driver Prediction** (OpenML ID: `42742`).

Das Projekt beinhaltet explorative Datenanalyse (EDA), standardisiertes Preprocessing, Hyperparameteroptimierung für diverse Modellklassen (Logistische Regression, Random Forest, XGBoost, HistGradientBoosting, LinearSVC mit Dimensionsreduktion) sowie eine zentrale Gesamtauswertung mit automatischer Laufzeit- und Metrikerfassung.

---

## Inhaltsverzeichnis
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

## Projektübersicht & Architektur

Aufgrund des starken Klassenungleichgewichts (ca. 3,6 % Schadensfälle) setzt dieses Projekt auf:
- **Stratifizierte Splits** (`stratify=y`) für eine faire Validierung.
- **Primäre Evaluationsmetriken:** PR-AUC (Average Precision) und ROC-AUC, ergänzend Balanced Accuracy, F1, Precision, Recall und Confusion Matrix.
- **Einheitliche Preprocessing-Logik**, die in jedem Modellskript (02–05) identisch als eigene `ColumnTransformer`-Pipeline aufgebaut wird:
  - *Numerische Features:* Median-Imputation + `StandardScaler`.
  - *Kategoriale Features (`_cat`):* Konstante Imputation (`"Missing"`) + `OneHotEncoder(handle_unknown="ignore")`.
  - *Binäre Features (`_bin`):* Modus-Imputation (`most_frequent`).
- **Zentrale Konfiguration:** Alle Pfade, Seeds (`RANDOM_STATE = 42`) und Exportordner werden zentral über `config.py` gesteuert.
- **Automatische Orchestrierung:** `06_gesamtauswertung_und_ergebnisse.py` prüft für jedes Modell, ob bereits eine Ergebnis-CSV existiert, und startet bei Bedarf automatisch das jeweilige Modellskript (02–05) als eigenen Python-Prozess (`subprocess`). Bereits vorhandene Ergebnisse werden nicht erneut berechnet, außer mit dem Flag `--force-rerun`.

---

## Projekt- & Ordnerstruktur

```text
.
├── 01_eda.py                                # Explorative Datenanalyse & Feature-Visualisierung
├── 02_logistic_regression.py                # Einzelmodul: Logistische Regression (inkl. eigener Preprocessing-Pipeline)
├── 03_random_forest.py                      # Einzelmodul: Random Forest (Learning Curve, manuelle Hyperparametersuche)
├── 04_xgboost_histgradientboosting.py       # Einzelmodul: Boosting (XGBoost vs. HistGradientBoosting)
├── 05_dimensionsreduktion_linear_svc.py     # Einzelmodul: LinearSVC & Dimensionsreduktion (SelectKBest/PCA/TruncatedSVD)
├── 06_gesamtauswertung_und_ergebnisse.py    # ★ HAUPTSKRIPT: Startet fehlende Modellskripte automatisch & aggregiert Ergebnisse
│
├── config.py                                # Zentrale Pfad-, Seed- & Output-Konfiguration
├── data_loading.py                          # Automatischer OpenML-Download (inkl. Retry) & CSV-Export nach Kategorie
├── results_summary.py                       # Zusammenführung & Spalten-Standardisierung der finalen Ergebnistabellen
├── plotting.py                              # Visualisierungsmodule (PNG-Export)
├── timing.py                                # Laufzeitmessung & -aggregation (runtime_log.csv / runtime_summary.csv)
├── requirements.txt                         # Python-Abhängigkeiten
└── output/                                  # Automatisch generierte Ergebnisse (wird bei Bedarf angelegt)
    ├── data/                                # Nach Kategorie exportierte CSV-Dateien (individuell, regional, fahrzeug, berechnet)
    ├── figures/                             # Exportierte Plots (.png), je Skript in einem Unterordner (z. B. 01_eda/, 06_model_comparison/)
    └── tables/                              # Ergebnistabellen je Modell in Unterordnern (z. B. tables/02_logistic_regression/...csv)
                                              #   sowie übergreifend: runtime_log.csv, runtime_summary.csv, finale_ergebnistabelle.csv
```

---

## Voraussetzungen

- **Python:** Version `3.9` bis `3.11` (empfohlen: `3.10` oder `3.11`)
- **Git** (für lokales Klonen)
- Aktive Internetverbindung beim ersten Ausführen (zum automatischen Download von OpenML, Data-ID `42742`)

---

## Schnellstart & Installation

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

## Pipeline-Ausführung (Empfohlener Ablauf)

### Schritt 1: Explorative Datenanalyse (EDA)
Lädt den Rohdatensatz automatisch von OpenML herunter, exportiert die Datenkategorien nach `output/data/` und erstellt alle Verteilungs- und Korrelationsgrafiken:
```bash
python 01_eda.py
```

---

### Schritt 2: Gesamtauswertung & alle Modelle ausführen
Das Skript **`06_gesamtauswertung_und_ergebnisse.py`** ist der zentrale Orchestrator. Für jedes der vier Modelle (Logistische Regression, Random Forest, XGBoost / HistGradientBoosting, LinearSVC mit Dimensionsreduktion) prüft es zunächst, ob bereits eine Ergebnis-CSV in `output/tables/<modell>/` existiert. Fehlt diese, wird das zugehörige Modellskript automatisch als eigener Prozess gestartet; existierende Ergebnisse werden nicht erneut berechnet. Anschließend fasst das Skript alle Metriken zusammen und erstellt die finale Vergleichstabelle sowie die Gesamtplots:
```bash
python 06_gesamtauswertung_und_ergebnisse.py
```

**⚠️ WICHTIG: Um alle vier Modellskripte (02–05) zwangsweise neu auszuführen, muss das Flag `--force-rerun` verwendet werden:**
```bash
python 06_gesamtauswertung_und_ergebnisse.py --force-rerun
```
Ohne dieses Flag werden bereits existierende Ergebnisse übersprungen und nur die finale Aggregation durchgeführt (sehr schnell). Mit `--force-rerun` werden alle Modelle komplett neu trainiert und evaluiert (kann mehrere Minuten dauern).

---

### Optionale Einzelausführung (Nur für Detailanalysen)

Falls wir ein bestimmtes Modell separat untersuchen, detaillierte Lernkurven analysieren oder Koeffizienten prüfen möchtest, können wir die jeweiligen Skripte auch isoliert ausführen:

| Skript | Fokus / Zweck |
| :--- | :--- |
| `python 02_logistic_regression.py` | PCA-Gegenprüfung, GridSearchCV über `C`, Koeffizientenanalyse & Schwellenwert-Analyse (Threshold 0,50–0,70) für LogReg. |
| `python 03_random_forest.py` | Learning Curve, manueller Vergleich mehrerer Hyperparameter-Varianten auf 50k-Stichprobe & finale Bewertung auf Testdaten. |
| `python 04_xgboost_histgradientboosting.py` | RandomizedSearchCV & direkter Modellvergleich für XGBoost und HistGradientBoosting. |
| `python 05_dimensionsreduktion_linear_svc.py` | Learning Curve, Feature-Selektion (`SelectKBest`), `PCA` & `TruncatedSVD` im Vergleich, kombiniert mit `LinearSVC`. |

Jedes Skript exportiert seine finale Ergebnistabelle automatisch nach `output/tables/<skriptname>/` und protokolliert seine Laufzeit in `output/tables/runtime_log.csv`.

---

## Ergebnisse & Metriken

Nach der Ausführung von Schritt 1 und 2 liegen alle Resultate im `output/`-Ordner bereit:

1. **`output/tables/<modell>/…ergebnisse.csv`:**
   Die individuelle finale Ergebnistabelle jedes Modellskripts (02–05).
2. **`output/tables/finale_ergebnistabelle.csv`:**
   Von `06_gesamtauswertung_und_ergebnisse.py` erzeugte, zusammengeführte Tabelle (via `results_summary.py`) mit allen gemeinsamen Test-Metriken (PR-AUC, ROC-AUC, Balanced Accuracy, F1, Precision, Recall, Confusion Matrix) im direkten Vergleich über alle Modellklassen hinweg.
3. **`output/tables/runtime_log.csv`:**
   Rohes, fortlaufendes Laufzeit-Protokoll aller Skript- und Fit-Durchläufe (siehe `timing.py`).
4. **`output/tables/runtime_summary.csv`:**
   Aggregierte Übersicht der jeweils letzten Laufzeit pro Skript/Modell, erzeugt am Ende von `06_gesamtauswertung_und_ergebnisse.py`.
5. **`output/figures/`:**
   Hochauflösende PNG-Grafiken, je Skript in einem eigenen Unterordner (u. a. `01_eda/`, `02_logistic_regression/`, `03_random_forest/`, `05_dimensionsreduktion_linear_svc/` und der finale Modellvergleich in `06_model_comparison/`).

---

## Fehlerbehebung (Troubleshooting)

- **OpenML Gateway Timeout / Download-Fehler:**
  `data_loading.py` verfügt über einen automatischen Retry-Mechanismus (`MAX_RETRIES = 5`, `RETRY_DELAY_SECONDS = 5`). Sollte der Download dennoch abbrechen, prüfe deine Internetverbindung und führe das Skript erneut aus.
- **XGBoost unter macOS (Apple Silicon M1/M2/M3):**
  Falls bei der Installation von `xgboost` Probleme auftreten, installiere OpenMP via Homebrew:
  ```bash
  brew install libomp
  ```
  Ist XGBoost dennoch nicht importierbar, überspringt `04_xgboost_histgradientboosting.py` das XGBoost-Modell automatisch und wertet nur HistGradientBoosting aus.
- **Ergebnisse eines einzelnen Modells neu berechnen:**
  Lösche die entsprechende CSV unter `output/tables/<modell>/` oder starte `06_gesamtauswertung_und_ergebnisse.py --force-rerun`, um alle vier Modellskripte neu auszuführen.
