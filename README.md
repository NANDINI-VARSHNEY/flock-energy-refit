# REFIT Data Cleaning and Appliance Disaggregation

An end-to-end, reproducible Data Science and Non-Intrusive Load Monitoring (NILM) pipeline for **Flock Energy**. This project cleans raw high-frequency household electricity data, explores domestic washing machine energy use across multiple UK households, and builds machine learning models to disaggregate washing machine power consumption from whole-house smart-meter aggregate measurements.

---

## 1. Project Overview & Architecture

Electric utilities deploy smart meters to measure aggregate household active power. However, individual high-consumption appliances are rarely metered directly. Disaggregating high-energy appliances (such as washing machines) from whole-home electricity signals allows utilities to provide automated customer feedback, quantify demand-response flexibility, and verify energy efficiency interventions.

```
flock_energy_refit/
├── README.md                  # Project overview, setup, validation design, trade-offs
├── DATA.md                    # Data coverage, cleaning rules, gap decisions, schema
├── RESULTS.md                 # EDA findings, model metrics, failure analysis, plots
├── Recommendation.md          # 1-page utility summary & practical energy recommendations
├── requirements.txt           # Python dependency specification
├── environment.yml            # Conda environment definition
├── data/
│   ├── raw/                   # Raw REFIT data (e.g., House1.csv from REFIT_RAW_081116.7z)
│   ├── cleaned/               # Official cleaned REFIT data (CLEAN_House*.csv)
│   └── processed/             # Resampled 1-minute datasets & pipeline outputs
├── plots/                     # High-resolution generated charts & visualizations
│   ├── house1_cleaning_before_after.png
│   ├── house1_24h_trace.png
│   ├── house1_detected_cycles.png
│   ├── cross_household_comparison.png
│   └── actual_vs_predicted_evaluation.png
├── src/
│   ├── config.py              # Central configuration, file paths, and thresholds
│   ├── clean_raw.py           # Raw data cleaning pipeline for House 1
│   ├── eda_washing_machine.py # Multi-household washing machine EDA & cycle detection
│   ├── disaggregation_model.py# ML disaggregation models, feature engineering, evaluation
│   ├── mock_refit_generator.py# Realistic synthetic sample data generator for verification
│   └── run_pipeline.py        # Master CLI runner orchestrating all stages
└── notebooks/
    └── refit_pipeline.ipynb   # Interactive Jupyter notebook from start to finish
```

---

## 2. Dataset Information & Download

This project utilizes the **REFIT: Electrical Load Measurements** dataset published by the University of Strathclyde, Loughborough University, and the University of East Anglia (EPSRC Grant EP/K002368/1).

- **Nominal Sampling Interval:** ~8 seconds (raw), resampled to 1 minute for analysis and modeling.
- **Physical Units:** Active Power in Watts (W).
- **Temporal Coverage:** October 2013 to June 2015 (20 UK households).
- **Raw REFIT Data Archive (.7z):** [REFIT_RAW_081116.7z (Strathclyde Portal)](https://pureportal.strath.ac.uk/files/62090177/REFIT_RAW_081116.7z)
- **Official Cleaned REFIT Data Archive (.7z):** [CLEAN_REFIT_081116.7z (Strathclyde Portal)](https://pureportal.strath.ac.uk/files/62090184/CLEAN_REFIT_081116.7z)
- **Official Cleaned ReadMe:** [CLEAN_READ_ME_081116.txt](https://pure.strath.ac.uk/ws/portalfiles/portal/62090183/CLEAN_READ_ME_081116.txt)
- **Scientific Publication:** Murray, D., Stankovic, L., & Stankovic, V. (2017). *An electrical load measurements dataset of United Kingdom households from a two-year longitudinal study.* **Scientific Data**, 4, 160122. [DOI: 10.1038/sdata.2016.122](https://doi.org/10.1038/sdata.2016.122).

---

## 3. Environment Setup & Execution Instructions

### Prerequisites
- Python 3.9, 3.10, or 3.11
- `pip` or `conda`

### Quick Setup

```bash
# 1. Clone repository
git clone https://github.com/your-username/flock_energy_refit.git
cd flock_energy_refit

# 2. Option A: Using Conda
conda env create -f environment.yml
conda activate flock_energy_refit

# 2. Option B: Using Pip & Virtual Environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configuring Data Files
1. Download `REFIT_RAW_081116.7z` and extract `House1.csv` into `data/raw/`.
2. Download `CLEAN_REFIT_081116.7z` and extract `CLEAN_House1.csv`, `CLEAN_House2.csv`, `CLEAN_House3.csv`, and `CLEAN_House5.csv` into `data/cleaned/`.

### Running the Complete Pipeline
You can run the entire pipeline end-to-end with a single command:

```bash
python src/run_pipeline.py
```

### Running with Sample Verification Data (Demo Mode)
If the multi-gigabyte raw files are still downloading or you want to immediately verify all unit operations, run with `--demo`:

```bash
python src/run_pipeline.py --demo
```
This initializes a realistic synthetic dataset mirroring the exact schema, sensor dropouts, and multi-stage washing machine cycles of the REFIT houses.

### Running Individual Stages
```bash
# Stage 1: Clean raw House 1 data only
python src/clean_raw.py

# Stage 2: Run washing machine exploratory data analysis across target homes
python src/eda_washing_machine.py

# Stage 3: Train and evaluate disaggregation models
python src/disaggregation_model.py
```

### Running the Interactive Jupyter Notebook
```bash
jupyter notebook notebooks/refit_pipeline.ipynb
```

---

## 4. Key Modeling Assumptions & Preprocessing Decisions

1. **Physical Power Bounds:**
   - **Individual Appliance Monitors (IAMs):** Rated for 16A at 230V (~3,680 W nominal, 4,000 W peak). Any reading $> 4,000\text{ W}$ is clipped to 4,000 W in line with Strathclyde's official cleaning protocol.
   - **Aggregate Main Panel:** Nominal domestic UK electrical panels are 60A–100A at 230V. Spikes $> 12,000\text{ W}$ are capped.
   - **Non-Negativity:** Active power consumption cannot be negative; any negative sensor artifact is clipped to 0 W.
2. **Duplicate Timestamp Resolution:** Duplicate timestamps (sensor re-transmissions) are aggregated via arithmetic mean rather than dropped, preserving total energy integrals.
3. **Outage Flagging vs. Gap Imputation:** Short communication dropouts ($\le 2\text{ minutes}$) are linearly interpolated. Prolonged outages ($\ge 120\text{ minutes}$) are explicitly flagged with `is_outage = 1` rather than blind zero-filling or forward-filling, preventing distortion of downstream load factor statistics.
4. **Washing Machine Cycle Definition:**
   - **Threshold:** $P \ge 20\text{ W}$ separates operational motor/heating stages from standby microcontrollers.
   - **Idle Bridging:** Soak intervals $\le 15\text{ minutes}$ are bridged into the active cycle to prevent premature cycle splitting.
   - **Minimum Run Length:** $\ge 3\text{ minutes}$ and $\ge 0.05\text{ kWh}$ to eliminate transient door-switch or plug-in pulses.

---

## 5. Validation Design & Prevention of Data Leakage

1. **Chronological Splitting (No Shuffling):** Time-series data must never be randomly partitioned. We use a strictly chronological train/test split (first 80% for training, last 20% for testing).
2. **Isolation Buffer Window:** A 60-minute buffer window isolates the training set from the test set. Because backward rolling features span up to 60 minutes, this ensures that no test-window data points ever appear in the rolling history of the training set.
3. **Causal Features Only:** Lags ($t-1, t-2, \dots, t-10$), differences, and rolling statistics are calculated strictly backwards in time. No centered or forward-looking windows are used.

---

## 6. Technical Trade-Offs

| Decision | Selected Approach | Trade-Off & Justification |
| :--- | :--- | :--- |
| **Resampling Resolution** | 1-minute (`1min`) mean | Balances computational efficiency with signature preservation. Sub-second features (harmonics) are lost, but thermal and mechanical cycles remain distinct. |
| **Model Selection** | HistGradientBoosting & Random Forest | Tree ensembles handle non-linear step-changes and multi-scale lags natively without requiring feature scaling or extensive hyperparameter searches. Seq2Point CNNs require substantially longer training times and GPU resources. |
| **Outage Handling** | Explicit binary flagging (`is_outage`) | Preserves data integrity. Blind zero-filling would distort appliance availability and capacity factor metrics. |

---

## 7. AI Tools Used

During this project, AI assistants (Antigravity coding assistant powered by Gemini) were used for:
- Accelerating script scaffolding and boilerplate configuration.
- Generating draft markdown structures and LaTeX mathematical formulations.
- Formulating unit tests and mock synthetic data generators.
All scientific logic, threshold definitions, data leakage prevention checks, and metric evaluations were verified against the official Strathclyde REFIT publications and documentation.

---

## 8. Future Improvements

1. **Sequence-to-Point (Seq2Point) Deep Learning:** Implement a 1D Convolutional Neural Network (CNN) Seq2Point architecture (Zhang et al., 2018) to learn latent temporal feature representations directly from the raw aggregate window without manual feature engineering.
2. **Multi-Appliance Joint Disaggregation:** Expand from a single-appliance model to a joint multi-head architecture disaggregating washing machines, dishwashers, and fridge-freezers simultaneously, enforcing that sum of predicted submeters $\le$ aggregate power.
3. **Bayesian Uncertainty Quantification:** Predict probabilistic power bounds (quantile regression) to quantify uncertainty during overlapping aggregate power spikes.
