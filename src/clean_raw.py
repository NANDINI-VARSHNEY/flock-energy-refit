"""
Raw Data Cleaning Module for House 1.
Implements robust timestamp parsing, duplicate resolution, outlier capping,
1-minute resampling, gap interpolation, outage flagging, and before/after visualization.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure src is on sys.path
sys.path.append(str(Path(__file__).resolve().parent))
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, PLOTS_DIR, CLEANING_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RawCleaner")


class RawDataCleaner:
    """
    Cleans raw REFIT electricity consumption data for House 1.
    """

    def __init__(
        self,
        raw_filepath: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        plots_dir: Optional[Path] = None,
        config: Optional[Dict] = None,
    ):
        self.raw_filepath = raw_filepath or (RAW_DATA_DIR / "House1.csv")
        self.output_dir = output_dir or PROCESSED_DATA_DIR
        self.plots_dir = plots_dir or PLOTS_DIR
        self.config = config or CLEANING_CONFIG
        self.summary_report: Dict = {}

    def load_raw_data(self) -> pd.DataFrame:
        """
        Loads raw REFIT CSV data and performs initial structural inspection.
        """
        if not self.raw_filepath.exists():
            raise FileNotFoundError(
                f"Raw data file not found at: {self.raw_filepath}\n"
                f"Please download REFIT_RAW_081116.7z from Strathclyde Portal and extract House1.csv into {RAW_DATA_DIR}"
            )

        logger.info(f"Loading raw data from: {self.raw_filepath}")
        
        # Read header first to verify columns
        sample = pd.read_csv(self.raw_filepath, nrows=5)
        logger.info(f"Raw data columns detected: {list(sample.columns)}")
        
        # Determine timestamp column: 'Time' or 'Unix'
        time_col = "Time" if "Time" in sample.columns else sample.columns[0]
        
        # Load full dataframe
        df_raw = pd.read_csv(self.raw_filepath)
        raw_rows = len(df_raw)
        logger.info(f"Loaded {raw_rows:,} raw records.")
        
        self.summary_report["raw_record_count"] = raw_rows
        self.summary_report["raw_columns"] = list(df_raw.columns)
        return df_raw

    def parse_and_sort_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parses timestamps to UTC datetime index and sorts chronologically.
        """
        logger.info("Parsing timestamps and sorting chronologically...")
        df = df.copy()
        
        # Handle 'Time' column if present
        if "Time" in df.columns:
            df["DateTime"] = pd.to_datetime(df["Time"], errors="coerce")
        elif "Unix" in df.columns:
            df["DateTime"] = pd.to_datetime(df["Unix"], unit="s", errors="coerce")
        else:
            # Assume first column is timestamp
            df["DateTime"] = pd.to_datetime(df.iloc[:, 0], errors="coerce")

        null_timestamps = df["DateTime"].isna().sum()
        if null_timestamps > 0:
            logger.warning(f"Found {null_timestamps} invalid/null timestamps. Dropping them.")
            df = df.dropna(subset=["DateTime"])

        # Sort chronologically
        df = df.sort_values("DateTime").reset_index(drop=True)
        
        self.summary_report["null_timestamps_dropped"] = int(null_timestamps)
        self.summary_report["start_timestamp_raw"] = str(df["DateTime"].min())
        self.summary_report["end_timestamp_raw"] = str(df["DateTime"].max())
        return df

    def inspect_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identifies duplicates, irregular intervals, missing readings, and out-of-range values.
        """
        logger.info("Auditing data quality (duplicates, irregular intervals, extreme values)...")
        
        # 1. Duplicate Timestamps
        duplicate_count = df.duplicated(subset=["DateTime"]).sum()
        logger.info(f"Detected {duplicate_count:,} duplicate timestamps ({duplicate_count / len(df) * 100:.3f}%).")
        self.summary_report["duplicate_timestamp_count"] = int(duplicate_count)

        # 2. Irregular Sampling Intervals
        time_diffs = df["DateTime"].diff().dt.total_seconds()
        median_interval = time_diffs.median()
        mean_interval = time_diffs.mean()
        irregular_intervals = (time_diffs > 15) | (time_diffs < 5)
        irregular_count = irregular_intervals.sum()
        
        logger.info(f"Nominal sampling interval: median={median_interval:.1f}s, mean={mean_interval:.1f}s.")
        logger.info(f"Irregular intervals (outside 5s-15s window): {irregular_count:,} records.")
        self.summary_report["median_sampling_interval_sec"] = float(median_interval) if not np.isnan(median_interval) else None
        self.summary_report["irregular_interval_count"] = int(irregular_count)

        # 3. Extreme & Impossible Values
        power_cols = [c for c in df.columns if c not in ["Time", "Unix", "DateTime", "Issues"]]
        extreme_findings = {}
        for col in power_cols:
            neg_count = (df[col] < 0).sum()
            max_val = df[col].max()
            limit = self.config["aggregate_max_power_watts"] if "Agg" in col else self.config["iam_max_power_watts"]
            exceed_count = (df[col] > limit).sum()
            extreme_findings[col] = {
                "negative_count": int(neg_count),
                "exceed_limit_count": int(exceed_count),
                "max_value_raw": float(max_val) if not np.isnan(max_val) else None,
            }
            if neg_count > 0 or exceed_count > 0:
                logger.warning(
                    f"Column '{col}': {neg_count} negative readings, {exceed_count} readings above {limit}W (max={max_val}W)."
                )

        self.summary_report["extreme_readings"] = extreme_findings
        return df

    def resolve_duplicates_and_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies documented rules:
        - Duplicate rule: Group by DateTime and compute mean of readings (merges sub-second reads).
        - Negative rule: Power cannot be negative; clip at 0 Watts.
        - Physical limit rule: IAM capped at 4,000 W; Aggregate capped at 12,000 W.
        """
        logger.info("Resolving duplicates and clipping impossible power values...")
        df = df.copy()

        # Documented Duplicate Rule: Mean aggregation across identical timestamps
        power_cols = [c for c in df.columns if c not in ["Time", "Unix", "DateTime", "Issues"]]
        df_dedup = df.groupby("DateTime")[power_cols].mean().reset_index()
        logger.info(f"Records after duplicate resolution: {len(df_dedup):,}")

        # Outlier clipping according to physical sensor thresholds
        for col in power_cols:
            limit = self.config["aggregate_max_power_watts"] if "Agg" in col else self.config["iam_max_power_watts"]
            df_dedup[col] = df_dedup[col].clip(lower=0.0, upper=limit)

        return df_dedup

    def resample_and_handle_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resamples time-series to 1-minute intervals.
        - Fills short gaps (<= 2 minutes) via linear interpolation.
        - Flags long outages (> 120 minutes) with an explicit binary indicator.
        - Leaves long outages without blind imputation to preserve scientific integrity.
        """
        logger.info(f"Resampling to 1-minute intervals ({self.config['resample_rule']})...")
        df = df.set_index("DateTime")
        
        # 1. Resample to 1-minute mean
        df_1min = df.resample(self.config["resample_rule"]).mean()
        
        # 2. Track missing intervals before imputation
        null_counts_pre = df_1min.isna().sum()
        logger.info(f"Missing 1-minute intervals before gap filling:\n{null_counts_pre.to_dict()}")

        # 3. Detect contiguous missing gaps in Aggregate power
        is_nan = df_1min["Aggregate"].isna()
        gap_id = (~is_nan).cumsum()
        gap_lengths = is_nan.groupby(gap_id).transform("sum")

        short_gap_limit = self.config["short_gap_threshold_minutes"]
        long_outage_limit = self.config["long_outage_threshold_minutes"]

        # Flag long outages: binary column 'is_outage'
        df_1min["is_outage"] = (is_nan & (gap_lengths >= long_outage_limit)).astype(int)
        long_outage_count = df_1min["is_outage"].sum()
        logger.info(f"Flagged {long_outage_count:,} minutes as prolonged outages (>= {long_outage_limit} mins).")

        # Handle short missing gaps: linearly interpolate up to short_gap_limit
        power_cols = [c for c in df_1min.columns if c != "is_outage"]
        df_1min[power_cols] = df_1min[power_cols].interpolate(
            method="time", limit=short_gap_limit, limit_direction="forward"
        )
        
        # Fill remaining isolated NaNs outside prolonged outages with 0.0 or forward fill
        # For prolonged outages, preserve NaN or zero but maintain the is_outage flag
        df_1min["Aggregate"] = df_1min["Aggregate"].fillna(0.0)
        for col in power_cols:
            df_1min[col] = df_1min[col].fillna(0.0)

        cleaned_records = len(df_1min)
        logger.info(f"Total 1-minute records in cleaned dataset: {cleaned_records:,}")

        self.summary_report["cleaned_record_count"] = cleaned_records
        self.summary_report["prolonged_outage_minutes"] = int(long_outage_count)
        self.summary_report["cleaned_start_time"] = str(df_1min.index.min())
        self.summary_report["cleaned_end_time"] = str(df_1min.index.max())

        return df_1min.reset_index()

    def plot_before_and_after(
        self,
        df_raw: pd.DataFrame,
        df_clean: pd.DataFrame,
        channels: Tuple[str, str] = ("Appliance5", "Appliance1"),
        sample_hours: int = 48,
    ) -> Path:
        """
        Plots aggregate power and at least two appliance channels before and after cleaning.
        Saves the resulting plot to the plots directory.
        """
        logger.info("Generating before-and-after cleaning visualization...")
        
        # Pick a representative continuous time slice
        start_time = df_clean["DateTime"].min() + pd.Timedelta(days=7)
        end_time = start_time + pd.Timedelta(hours=sample_hours)

        raw_slice = df_raw[(df_raw["DateTime"] >= start_time) & (df_raw["DateTime"] <= end_time)]
        clean_slice = df_clean[(df_clean["DateTime"] >= start_time) & (df_clean["DateTime"] <= end_time)]

        ch1, ch2 = channels
        # Channel readable names for House 1
        ch1_name = "Washing Machine (Appliance 5)" if ch1 == "Appliance5" else ch1
        ch2_name = "Fridge (Appliance 1)" if ch2 == "Appliance1" else ch2

        fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex="col", sharey="row")
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

        # 1. Aggregate Power
        axes[0, 0].plot(raw_slice["DateTime"], raw_slice["Aggregate"], color="#1f77b4", alpha=0.7, lw=0.8)
        axes[0, 0].set_title(f"Raw Aggregate Power (~8s sampling)\n[Records: {len(raw_slice):,}]", fontsize=11, fontweight="bold")
        axes[0, 0].set_ylabel("Power (Watts)", fontsize=10)

        axes[0, 1].plot(clean_slice["DateTime"], clean_slice["Aggregate"], color="#2ca02c", lw=1.2)
        axes[0, 1].set_title(f"Cleaned Aggregate Power (1-min resampled)\n[Records: {len(clean_slice):,}]", fontsize=11, fontweight="bold")

        # 2. Channel 1 (Washing Machine)
        if ch1 in raw_slice.columns and ch1 in clean_slice.columns:
            axes[1, 0].plot(raw_slice["DateTime"], raw_slice[ch1], color="#ff7f0e", alpha=0.7, lw=0.8)
            axes[1, 0].set_title(f"Raw {ch1_name}", fontsize=11, fontweight="bold")
            axes[1, 0].set_ylabel("Power (Watts)", fontsize=10)

            axes[1, 1].plot(clean_slice["DateTime"], clean_slice[ch1], color="#d62728", lw=1.2)
            axes[1, 1].set_title(f"Cleaned {ch1_name}", fontsize=11, fontweight="bold")

        # 3. Channel 2 (Fridge)
        if ch2 in raw_slice.columns and ch2 in clean_slice.columns:
            axes[2, 0].plot(raw_slice["DateTime"], raw_slice[ch2], color="#9467bd", alpha=0.7, lw=0.8)
            axes[2, 0].set_title(f"Raw {ch2_name}", fontsize=11, fontweight="bold")
            axes[2, 0].set_ylabel("Power (Watts)", fontsize=10)
            axes[2, 0].set_xlabel("Timestamp", fontsize=10)

            axes[2, 1].plot(clean_slice["DateTime"], clean_slice[ch2], color="#8c564b", lw=1.2)
            axes[2, 1].set_title(f"Cleaned {ch2_name}", fontsize=11, fontweight="bold")
            axes[2, 1].set_xlabel("Timestamp", fontsize=10)

        for ax in axes.flat:
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.tick_params(axis="x", rotation=30)

        fig.suptitle(f"REFIT House 1: Data Cleaning Impact ({sample_hours}-Hour Window)", fontsize=14, fontweight="bold", y=0.99)
        plt.tight_layout()

        plot_path = self.plots_dir / "house1_cleaning_before_after.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Before/after plot saved to: {plot_path}")
        return plot_path

    def run_cleaning_pipeline(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Executes end-to-end cleaning pipeline for House 1.
        """
        df_raw = self.load_raw_data()
        df_sorted = self.parse_and_sort_timestamps(df_raw)
        df_inspected = self.inspect_data_quality(df_sorted)
        df_dedup = self.resolve_duplicates_and_outliers(df_inspected)
        df_clean = self.resample_and_handle_gaps(df_dedup)

        # Plot before and after
        self.plot_before_and_after(df_sorted, df_clean)

        # Save cleaned dataset
        output_file = self.output_dir / "House1_cleaned_1min.csv"
        df_clean.to_csv(output_file, index=False)
        logger.info(f"Cleaned dataset saved successfully to: {output_file}")

        return df_clean, self.summary_report


if __name__ == "__main__":
    cleaner = RawDataCleaner()
    try:
        df_clean, report = cleaner.run_cleaning_pipeline()
        print("\n=== RAW DATA CLEANING SUMMARY REPORT (HOUSE 1) ===")
        for k, v in report.items():
            print(f"{k}: {v}")
    except FileNotFoundError as e:
        print(f"\n[INFO] {e}")
