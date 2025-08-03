# 🐋 FindMyWhale: A Spatiotemporal Forecasting Framework

Welcome to **FindMyWhale** — an open, modular forecasting framework designed to predict **Southern Resident Killer Whale (SRKW)** presence in space and time. This project combines whale sightings, salmon dynamics, environmental variables, and human activity data into a unified pipeline for predictive modeling. Think: data science meets marine conservation with high-resolution spatial grids and cosmic-level covariate linkages.

## 🔮 What We're Building

We aim to generate **7-day forecasts** of SRKW presence using a clean, reproducible pipeline built around:

- Whale sighting reports (Acartia, Orca Network, Spyhopper, etc.)
- Salmon availability indicators (dam counts, hatchery returns)
- Environmental data (SST, chlorophyll-a, tides, bathymetry)
- Human noise + AIS ship traffic integration
- Behavioral & ecological nuance (matriarch loss, prey shifts)
- A full **spatiotemporal H3-based data cube** for modeling 🧊

## 🧠 Why It Matters

Southern Resident Killer Whales are endangered and deeply tied to the availability of Chinook salmon. Understanding when and where they'll appear can help mitigate vessel noise, inform conservation zones, and support ecosystem-based management. With open science, we're giving the orcas their own predictive dashboard.

---

## 🌐 Repo Structure

```bash
findmywhale/
├── data/
│   ├── raw/                  # Unprocessed data from APIs / downloads
│   ├── interim/              # Cleaned & standardized data
│   ├── processed/            # H3-joined modeling-ready datasets
│   └── external/             # Referenced data (e.g., bathymetry maps)
├── notebooks/
│   ├── eda/                  # Exploratory data analysis
│   ├── modeling/             # Forecasting approaches
│   └── utils/                # Utility notebooks or scratchpads
├── src/
│   ├── ingestion/            # Data ingestion scripts (APIs, downloads)
│   ├── preprocessing/        # Cleaning + formatting scripts
│   ├── features/             # Feature engineering (e.g., lags, noise proxies)
│   ├── modeling/             # ML/DL model code
│   └── viz/                  # Maps, plots, dashboards
├── config/
│   ├── h3_levels.yaml        # Hex resolution settings
│   └── env_vars.template     # API keys + sensitive paths (not committed)
├── tests/                    # Unit tests
├── README.md                 # You are here 💅
├── requirements.txt          # Dependencies
└── pyproject.toml            # Optional project metadata
```

---

## 🛠️ Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/stevetylda/FindMyWhale.git
cd FindMyWhale
```

### 2. Set up your environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

TODO: Update with conda install instructions

### 3. Add your environment variables

Copy the template and insert your secrets/API keys.

```bash
cp config/env.template config/.env
```

---

## 📡 Data Sources

We work with both public APIs and local datasets. The pipeline is designed to work offline after initial ingestion, with versioned data snapshots for reproducibility.

| Category | Sources |
|---------|---------|
| **Whale Sightings** | Orca Network, Acartia API, Spyhopper, Hydrophone triangulation (TBD), Satellite tags (TBD) |
| **Salmon Signals** | Columbia/Snake Dam Counts, Hatchery Returns, Coastal Telemetry |
| **Environmental** | MODIS/Sentinel (Chl-a), NOAA ERDDAP (SST, salinity), Bathymetry, Tides, Weather |
| **Human Activity** | AIS Ship Tracks, Hydrophone-recorded noise, Fishing seasons |
| **Competition** | Pinniped estimates, fishery openings, shark proxies |
| **Psych/Meta** | Matriarchal events, cultural paths, TikTok trends, magnetosphere chaos 🌕 |

---

## 🧪 Modeling Philosophy

We embrace multiple modeling approaches:
- Baseline regressions and time series models
- Spatial-temporal neural nets (ConvLSTM, Graph Attention)
- Attention-based transformers (temporal or mixed)
- Human bias correction models (holiday spikes, search trends)

Everything happens **on an H3 grid** for seamless spatial joins + generalization.

---

## 🧼 Goals for Clean Data

- H3-aggregated data at weekly intervals
- Lag features for salmon + noise
- Noise proxies created from AIS + hydrophone proximity
- Observer bias correction via indirect measures
- Model-ready `.parquet` exports with metadata

---

## 🚧 Roadmap

| Phase | Goal |
|-------|------|
| 🧱 Phase 1 | Data ingestion + spatiotemporal stack |
| 🔎 Phase 2 | Exploratory analysis, lag studies, bias modeling |
| 📈 Phase 3 | Predictive model prototyping |
| 🌐 Phase 4 | Deployable dashboard + alerts |
| 📣 Phase 5 | Publishing results, open science writeup |

---

## 💬 Contributing

Pull requests welcome! 

1. Fork the repo
2. Create a new branch: `git checkout -b feat/amazing-idea`
3. Commit changes: `git commit -m 'feat: adds magic'`
4. Push to your fork and open a PR

---

## 📜 License

MIT — because open science deserves open code.

---

## 🌊 Acknowledgments

- Orca Network, Acartia, Spyhopper, NOAA, WDFW, and the whales themselves
- All the open data providers making marine conservation possible

---

## 👀 Stay curious

> *"We don’t need to be the smartest in the sea, just the most persistent."* – Probably an orca