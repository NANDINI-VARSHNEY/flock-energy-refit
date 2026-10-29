"""
REFIT Synthetic / Sample Data Generator.
Creates physically realistic, schema-identical sample data mirroring REFIT House 1 to House 5.
Used for verification, unit testing, and immediate end-to-end pipeline execution
when multi-gigabyte raw REFIT archives are still downloading or unavailable.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from config import RAW_DATA_DIR, CLEAN_DATA_DIR


def generate_realistic_washing_machine_cycle(length_mins: int = 90) -> np.ndarray:
    """
    Simulates a realistic multi-stage European front-loading washing machine cycle:
    1. Water Fill & Drum Saturation: ~5 mins, ~50-100W
    2. Water Heating Stage (Resistive Heater): ~20-25 mins, ~2000-2400W
    3. Wash Drum Agitation / Tumbling: ~25 mins, ~100-250W fluctuating
    4. Drain & Rinse: ~15 mins, intermittent spikes ~150-300W
    5. High-Speed Spin Extraction: ~15 mins, ramping up to ~500-800W
    6. Cool-down & Anti-crease: ~5 mins, ~40-60W
    """
    base_length = max(length_mins, 95)
    profile = np.zeros(base_length)
    
    # 1. Fill (0 to 5 min)
    profile[0:5] = np.random.uniform(50, 100, size=5)
    
    # 2. Heating (5 to 30 min) - High resistive load
    profile[5:30] = np.random.uniform(2050, 2350, size=25)
    
    # 3. Wash Tumbling (30 to 55 min) - Motor reversing
    profile[30:55] = np.random.uniform(120, 240, size=25)
    # Brief soak pause
    profile[40:43] = 4.0
    
    # 4. Rinse & Drain (55 to 70 min)
    profile[55:70] = np.random.uniform(100, 300, size=15)
    
    # 5. Spin Cycle (70 to 85 min) - High motor torque
    ramp = np.linspace(300, 750, 15)
    profile[70:85] = ramp + np.random.normal(0, 20, size=15)
    
    # 6. End anti-crease (85 to base_length)
    profile[85:base_length] = np.random.uniform(20, 50, size=base_length - 85)
    
    return np.clip(profile[:length_mins], 0, 4000)


def generate_sample_refit_raw_house1(
    output_path: Path, days: int = 7, sampling_sec: int = 8
) -> Path:
    """
    Generates sample raw House 1 data at ~8s sampling interval,
    incorporating real-world artifacts: duplicate timestamps, out-of-range sensor spikes,
    irregular intervals, and missing gaps.
    """
    print(f"Generating realistic sample raw House 1 data ({days} days at ~{sampling_sec}s)...")
    start_time = pd.Timestamp("2014-03-01 00:00:00")
    total_seconds = days * 24 * 3600
    
    # Generate timestamps with slight irregular jitter
    raw_times = []
    curr = start_time
    while (curr - start_time).total_seconds() < total_seconds:
        raw_times.append(curr)
        # 8s nominal with occasional jitter
        jitter = np.random.choice([8, 8, 8, 7, 9, 16], p=[0.7, 0.15, 0.05, 0.04, 0.04, 0.02])
        curr += pd.Timedelta(seconds=int(jitter))

    n_samples = len(raw_times)
    
    # Baseline base load (~150-300W background)
    base_load = np.random.normal(200, 30, size=n_samples)
    
    # Fridge cyclic load (Appliance 1) ~70-120W compressor cycling
    fridge = np.zeros(n_samples)
    compressor_on = False
    cycle_counter = 0
    for i in range(n_samples):
        cycle_counter += 1
        if not compressor_on and cycle_counter > 300: # ~40 mins off
            compressor_on = True
            cycle_counter = 0
        elif compressor_on and cycle_counter > 150: # ~20 mins on
            compressor_on = False
            cycle_counter = 0
        if compressor_on:
            fridge[i] = np.random.normal(95, 8)

    # Washing Machine (Appliance 5) - runs once every 2 days
    washing_machine = np.zeros(n_samples)
    samples_per_day = int(86400 / 8)
    for day in range(0, days, 2):
        # Start at 10:00 AM on active days
        start_idx = day * samples_per_day + int(10 * 3600 / 8)
        cycle_1m = generate_realistic_washing_machine_cycle(length_mins=90)
        cycle_8s = np.repeat(cycle_1m, int(60 / 8))
        end_idx = min(start_idx + len(cycle_8s), n_samples)
        washing_machine[start_idx:end_idx] = cycle_8s[: end_idx - start_idx]

    # Additional appliances: Kettle, TV, Microwave
    kettle_pulses = np.zeros(n_samples)
    for d in range(days):
        # 3 kettle boils per day (~2200W, 3 mins)
        for h in [8, 13, 19]:
            k_idx = d * samples_per_day + int(h * 3600 / 8)
            k_len = int(3 * 60 / 8)
            kettle_pulses[k_idx : min(k_idx + k_len, n_samples)] = np.random.normal(2200, 50)

    # Aggregate power = base + fridge + washing_machine + kettle + other
    aggregate = base_load + fridge + washing_machine + kettle_pulses + np.random.normal(50, 15, size=n_samples)
    aggregate = np.clip(aggregate, 0, None)

    df_raw = pd.DataFrame({
        "Time": [t.strftime("%Y-%m-%d %H:%M:%S") for t in raw_times],
        "Unix": [int(t.timestamp()) for t in raw_times],
        "Aggregate": aggregate,
        "Appliance1": fridge,
        "Appliance2": np.random.normal(60, 5, size=n_samples), # Chest freezer
        "Appliance3": np.random.normal(50, 5, size=n_samples), # Upright freezer
        "Appliance4": np.zeros(n_samples),                     # Tumble dryer
        "Appliance5": washing_machine,                         # Washing machine
        "Appliance6": np.zeros(n_samples),                     # Dishwasher
        "Appliance7": np.random.choice([0, 150], size=n_samples, p=[0.7, 0.3]), # Computer
        "Appliance8": np.random.choice([0, 90], size=n_samples, p=[0.8, 0.2]),  # TV
        "Appliance9": np.zeros(n_samples),                     # Electric heater
    })

    # Introduce real-world anomalies for cleaning verification:
    # 1. Introduce duplicate timestamps (0.5% duplicates)
    dup_indices = np.random.choice(n_samples, size=int(n_samples * 0.005), replace=False)
    dup_rows = df_raw.iloc[dup_indices].copy()
    # 2. Introduce impossible sensor spike (>4000W on IAM)
    spike_idx = np.random.choice(n_samples, size=3, replace=False)
    df_raw.loc[spike_idx, "Appliance5"] = 5200.0
    # 3. Introduce impossible aggregate spike (>12000W)
    agg_spike_idx = np.random.choice(n_samples, size=2, replace=False)
    df_raw.loc[agg_spike_idx, "Aggregate"] = 18500.0
    # 4. Introduce a 3-hour prolonged outage
    outage_start = int(n_samples * 0.4)
    outage_len = int(3 * 3600 / 8)
    df_raw = df_raw.drop(df_raw.index[outage_start : outage_start + outage_len])

    # Append duplicates and shuffle slightly
    df_raw = pd.concat([df_raw, dup_rows], ignore_index=True)
    df_raw = df_raw.sample(frac=1.0, random_state=42).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_raw.to_csv(output_path, index=False)
    print(f"Sample raw House 1 data generated at: {output_path} ({len(df_raw):,} records)")
    return output_path


def generate_sample_clean_households(clean_dir: Path, days: int = 14) -> List[Path]:
    """
    Generates clean 1-minute sampled datasets for House 1, House 2, House 3, and House 5.
    Matches the schema of CLEAN_House*.csv files.
    """
    clean_dir.mkdir(parents=True, exist_ok=True)
    start_time = pd.Timestamp("2014-04-01 00:00:00")
    total_minutes = days * 24 * 60
    timestamps = [start_time + pd.Timedelta(minutes=i) for i in range(total_minutes)]

    house_configs = [
        ("House1", "Appliance5", 2), # House 1: Washing Machine is Appliance 5, runs every 2 days
        ("House2", "Appliance2", 3), # House 2: Washing Machine is Appliance 2, runs every 3 days
        ("House3", "Appliance5", 2), # House 3: Washing Machine is Appliance 5, runs every 2 days
        ("House5", "Appliance3", 4), # House 5: Washing Machine is Appliance 3, runs every 4 days
    ]

    generated_paths = []
    for house_name, wm_col, freq_days in house_configs:
        file_path = clean_dir / f"CLEAN_{house_name}.csv"
        print(f"Generating sample cleaned data for {house_name} ({days} days at 1-min resolution)...")
        
        base_power = np.random.normal(180, 25, size=total_minutes)
        fridge_power = np.zeros(total_minutes)
        for i in range(total_minutes):
            if (i % 60) < 20: # 20 mins on, 40 mins off
                fridge_power[i] = np.random.normal(85, 5)

        wm_power = np.zeros(total_minutes)
        for d in range(0, days, freq_days):
            # active cycle at 11:00 AM
            start_m = d * 1440 + 11 * 60
            cycle = generate_realistic_washing_machine_cycle(length_mins=80)
            end_m = min(start_m + len(cycle), total_minutes)
            wm_power[start_m:end_m] = cycle[: end_m - start_m]

        # Other intermittent appliances (kettle, microwave)
        other_power = np.zeros(total_minutes)
        for d in range(days):
            for h in [7, 12, 18, 21]:
                m_idx = d * 1440 + h * 60 + np.random.randint(0, 40)
                other_power[m_idx : min(m_idx + 4, total_minutes)] = np.random.normal(2100, 40)

        aggregate = base_power + fridge_power + wm_power + other_power + np.random.normal(40, 10, size=total_minutes)
        aggregate = np.clip(aggregate, 0, None)

        data_dict = {
            "Time": [t.strftime("%Y-%m-%d %H:%M:%S") for t in timestamps],
            "Unix": [int(t.timestamp()) for t in timestamps],
            "Aggregate": aggregate,
        }
        for a_idx in range(1, 10):
            col_name = f"Appliance{a_idx}"
            if col_name == wm_col:
                data_dict[col_name] = wm_power
            elif a_idx == 1:
                data_dict[col_name] = fridge_power
            else:
                data_dict[col_name] = np.random.normal(30, 5, size=total_minutes)

        data_dict["Issues"] = 0

        df_clean = pd.DataFrame(data_dict)
        df_clean.to_csv(file_path, index=False)
        generated_paths.append(file_path)

    print(f"Generated {len(generated_paths)} cleaned household sample files.")
    return generated_paths


if __name__ == "__main__":
    generate_sample_refit_raw_house1(RAW_DATA_DIR / "House1.csv", days=5)
    generate_sample_clean_households(CLEAN_DATA_DIR, days=10)
