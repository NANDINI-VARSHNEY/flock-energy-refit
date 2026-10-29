"""
Configuration file for REFIT Data Cleaning and Appliance Disaggregation Pipeline.
Defines file paths, appliance channel mappings, cleaning thresholds, and model parameters.
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CLEAN_DATA_DIR = DATA_DIR / "cleaned"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PLOTS_DIR = BASE_DIR / "plots"
MODELS_DIR = BASE_DIR / "models"

# Ensure output directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, CLEAN_DATA_DIR, PROCESSED_DATA_DIR, PLOTS_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset URLs & Citation Information
DATASET_METADATA = {
    "title": "REFIT: Electrical Load Measurements",
    "institution": "University of Strathclyde, Loughborough University, University of East Anglia",
    "funder": "EPSRC (Grant Reference EP/K002368/1)",
    "sampling_interval_raw": "8 seconds (approx)",
    "units": "Active Power (Watts)",
    "date_range": "October 2013 - June 2015",
    "raw_url": "https://pureportal.strath.ac.uk/files/62090177/REFIT_RAW_081116.7z",
    "clean_url": "https://pureportal.strath.ac.uk/files/62090184/CLEAN_REFIT_081116.7z",
    "readme_url": "https://pure.strath.ac.uk/ws/portalfiles/portal/62090183/CLEAN_READ_ME_081116.txt",
    "zenodo_doi": "10.5281/zenodo.5063428",
    "paper_doi": "10.1038/sdata.2016.122",
}

# Appliance channel mappings across houses (REFIT metadata)
# Columns in Clean REFIT: Time, Unix, Aggregate, Appliance1, Appliance2, ..., Appliance9, Issues
HOUSE_APPLIANCE_MAPPING = {
    "House 1": {
        "washing_machine": "Appliance5",
        "fridge": "Appliance1",
        "chest_freezer": "Appliance2",
        "upright_freezer": "Appliance3",
        "tumble_dryer": "Appliance4",
        "dishwasher": "Appliance6",
        "computer": "Appliance7",
        "television": "Appliance8",
        "electric_heater": "Appliance9",
    },
    "House 2": {
        "fridge_freezer": "Appliance1",
        "washing_machine": "Appliance2",
        "dishwasher": "Appliance3",
        "television": "Appliance4",
        "microwave": "Appliance5",
        "toaster": "Appliance6",
        "hifi": "Appliance7",
        "kettle": "Appliance8",
        "fan": "Appliance9",
    },
    "House 3": {
        "toaster": "Appliance1",
        "fridge_freezer": "Appliance2",
        "freezer": "Appliance3",
        "tumble_dryer": "Appliance4",
        "washing_machine": "Appliance5",
        "dishwasher": "Appliance6",
        "television": "Appliance7",
        "microwave": "Appliance8",
        "kettle": "Appliance9",
    },
    "House 5": {
        "fridge_freezer": "Appliance1",
        "water_heater": "Appliance2",
        "washing_machine": "Appliance3",
        "dishwasher": "Appliance4",
        "television": "Appliance5",
        "microwave": "Appliance6",
        "kettle": "Appliance7",
        "lighting": "Appliance8",
        "toaster": "Appliance9",
    },
    "House 8": {
        "fridge": "Appliance1",
        "freezer": "Appliance2",
        "dryer": "Appliance3",
        "washing_machine": "Appliance4",
        "toaster": "Appliance5",
        "computer": "Appliance6",
        "television": "Appliance7",
        "microwave": "Appliance8",
        "kettle": "Appliance9",
    },
    "House 9": {
        "fridge_freezer": "Appliance1",
        "washing_machine": "Appliance2",
        "dishwasher": "Appliance3",
        "television": "Appliance4",
        "microwave": "Appliance5",
        "kettle": "Appliance6",
        "hifi": "Appliance7",
        "electric_heater": "Appliance8",
        "toaster": "Appliance9",
    },
}

# Cleaning Parameters
CLEANING_CONFIG = {
    # Individual Appliance Monitor sensor physical limit (Watts)
    "iam_max_power_watts": 4000.0,
    # Household aggregate main service panel rating limit (Watts) (~40-60A at 230V)
    "aggregate_max_power_watts": 12000.0,
    # Short gap threshold for interpolation / forward-filling (minutes)
    "short_gap_threshold_minutes": 2,
    # Long outage threshold for flagging rather than blind filling (minutes)
    "long_outage_threshold_minutes": 120,
    # Target resampling frequency
    "resample_rule": "1min",
}

# Washing Machine Cycle Detection Parameters
CYCLE_DETECTION_CONFIG = {
    # Power threshold to distinguish active washing states from idle/standby electronics (Watts)
    "on_power_threshold_watts": 20.0,
    # Minimum continuous duration above threshold to constitute an intentional wash cycle (minutes)
    "min_cycle_duration_minutes": 3,
    # Maximum idle pause duration within a wash cycle before splitting into separate cycles (minutes)
    # (Accounts for soaking and water heating pauses without prematurely truncating the cycle)
    "max_idle_pause_minutes": 15,
    # Minimum energy consumption for a valid cycle (kWh)
    "min_cycle_energy_kwh": 0.05,
}

# Machine Learning Model Parameters
ML_CONFIG = {
    "random_seed": 42,
    "target_appliance": "washing_machine",
    "test_split_ratio": 0.2,
    "lag_steps_minutes": [1, 2, 3, 5, 10],
    "rolling_windows_minutes": [5, 15, 30, 60],
    "model_types": ["BaselineAlwaysOff", "HistGradientBoosting", "RandomForest"],
    "activation_threshold_watts": 20.0,
}
