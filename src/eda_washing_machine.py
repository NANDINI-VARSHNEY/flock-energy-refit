"""
Exploratory Data Analysis (EDA) Module for Washing Machine Power Consumption.
Analyzes washing machine usage across eligible REFIT households (House 1, House 2, House 3, House 5).
Implements cycle detection, diurnal/weekly patterns, cross-household metrics, and plots.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parent))
from config import CLEAN_DATA_DIR, PLOTS_DIR, HOUSE_APPLIANCE_MAPPING, CYCLE_DETECTION_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("WashingMachineEDA")


class WashingMachineEDA:
    """
    Performs exploratory data analysis on washing machine usage across REFIT households.
    """

    def __init__(
        self,
        clean_data_dir: Optional[Path] = None,
        plots_dir: Optional[Path] = None,
        house_mapping: Optional[Dict] = None,
        cycle_config: Optional[Dict] = None,
    ):
        self.clean_data_dir = clean_data_dir or CLEAN_DATA_DIR
        self.plots_dir = plots_dir or PLOTS_DIR
        self.house_mapping = house_mapping or HOUSE_APPLIANCE_MAPPING
        self.cycle_config = cycle_config or CYCLE_DETECTION_CONFIG
        self.household_stats: Dict[str, Dict] = {}
        self.household_dfs: Dict[str, pd.DataFrame] = {}
        self.detected_cycles: Dict[str, pd.DataFrame] = {}

    def load_and_resample_household(
        self, house_id: str, wm_column: str, sample_max_days: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Loads official cleaned REFIT CSV, extracts Aggregate and Washing Machine channels,
        and resamples to 1-minute resolution.
        """
        file_candidates = [
            self.clean_data_dir / f"CLEAN_{house_id}.csv",
            self.clean_data_dir / f"{house_id}.csv",
            self.clean_data_dir / f"CLEAN_{house_id.replace(' ', '')}.csv",
            self.clean_data_dir / f"{house_id.replace(' ', '')}.csv",
        ]
        
        file_path = next((f for f in file_candidates if f.exists()), None)
        if not file_path:
            raise FileNotFoundError(
                f"Data file for {house_id} not found in {self.clean_data_dir}.\n"
                f"Tried: {[str(f) for f in file_candidates]}"
            )

        logger.info(f"Loading {house_id} from: {file_path}")
        df = pd.read_csv(file_path)
        
        # Verify columns
        if "Time" in df.columns:
            df["DateTime"] = pd.to_datetime(df["Time"], errors="coerce")
        elif "Unix" in df.columns:
            df["DateTime"] = pd.to_datetime(df["Unix"], unit="s", errors="coerce")
        else:
            df["DateTime"] = pd.to_datetime(df.iloc[:, 0], errors="coerce")

        if wm_column not in df.columns:
            raise KeyError(f"Washing machine column '{wm_column}' not found in {house_id} columns: {list(df.columns)}")

        # Keep relevant columns and sort
        df = df[["DateTime", "Aggregate", wm_column]].dropna(subset=["DateTime"])
        df = df.sort_values("DateTime").reset_index(drop=True)

        if sample_max_days:
            start_date = df["DateTime"].min()
            df = df[df["DateTime"] <= start_date + pd.Timedelta(days=sample_max_days)]

        # Resample to 1-minute mean
        df_1min = (
            df.set_index("DateTime")
            .resample("1min")
            .mean()
            .interpolate(method="time", limit=2)
            .fillna(0.0)
            .reset_index()
        )
        df_1min = df_1min.rename(columns={wm_column: "WashingMachine"})
        df_1min["House"] = house_id
        
        logger.info(f"{house_id}: 1-min resampled dataset contains {len(df_1min):,} records.")
        return df_1min

    def detect_cycles(self, df_1min: pd.DataFrame, house_id: str) -> pd.DataFrame:
        """
        Threshold-based washing machine cycle detection:
        - Power >= on_power_threshold (20W) indicates active state.
        - Idles <= max_idle_pause (15 mins) are bridged (accounts for soak/water heating).
        - Min duration >= min_cycle_duration (3 mins) to eliminate spurious spikes.
        - Min energy >= min_cycle_energy (0.05 kWh).
        """
        threshold = self.cycle_config["on_power_threshold_watts"]
        max_pause = self.cycle_config["max_idle_pause_minutes"]
        min_duration = self.cycle_config["min_cycle_duration_minutes"]
        min_energy = self.cycle_config["min_cycle_energy_kwh"]

        wm_series = df_1min["WashingMachine"].values
        dt_series = df_1min["DateTime"].values
        n = len(wm_series)

        is_active = wm_series >= threshold

        # Bridge idle gaps <= max_pause minutes
        active_bridged = is_active.copy()
        gap_start = None
        for i in range(1, n):
            if not active_bridged[i] and active_bridged[i - 1]:
                gap_start = i
            elif active_bridged[i] and gap_start is not None:
                gap_len = i - gap_start
                if gap_len <= max_pause:
                    active_bridged[gap_start:i] = True
                gap_start = None

        # Segment continuous blocks of True
        cycles = []
        in_cycle = False
        start_idx = 0

        for i in range(n):
            if active_bridged[i] and not in_cycle:
                in_cycle = True
                start_idx = i
            elif not active_bridged[i] and in_cycle:
                in_cycle = False
                end_idx = i - 1
                
                # Check cycle validity
                duration_mins = end_idx - start_idx + 1
                cycle_power = wm_series[start_idx : end_idx + 1]
                peak_power = np.max(cycle_power)
                # Energy in kWh: Power (Watts) * (1/60 hr) / 1000 W/kW
                energy_kwh = np.sum(cycle_power) / (60.0 * 1000.0)

                if duration_mins >= min_duration and energy_kwh >= min_energy:
                    cycles.append({
                        "House": house_id,
                        "Start_Time": pd.to_datetime(dt_series[start_idx]),
                        "End_Time": pd.to_datetime(dt_series[end_idx]),
                        "Duration_Minutes": duration_mins,
                        "Peak_Power_Watts": peak_power,
                        "Mean_Power_Watts": np.mean(cycle_power),
                        "Energy_kWh": energy_kwh,
                        "Start_Idx": start_idx,
                        "End_Idx": end_idx,
                    })

        df_cycles = pd.DataFrame(cycles)
        logger.info(f"{house_id}: Detected {len(df_cycles):,} washing-machine operating cycles.")
        return df_cycles

    def analyze_household(self, house_id: str, wm_column: str, sample_max_days: Optional[int] = None) -> Dict:
        """
        Calculates availability, zero/off-time, active-time %, and detected cycle metrics.
        """
        df_1min = self.load_and_resample_household(house_id, wm_column, sample_max_days=sample_max_days)
        df_cycles = self.detect_cycles(df_1min, house_id)

        self.household_dfs[house_id] = df_1min
        self.detected_cycles[house_id] = df_cycles

        threshold = self.cycle_config["on_power_threshold_watts"]
        total_minutes = len(df_1min)
        active_minutes = (df_1min["WashingMachine"] >= threshold).sum()
        zero_minutes = (df_1min["WashingMachine"] < threshold).sum()

        stats = {
            "House": house_id,
            "Date_Range_Start": str(df_1min["DateTime"].min()),
            "Date_Range_End": str(df_1min["DateTime"].max()),
            "Total_Minutes": int(total_minutes),
            "Zero_Off_Time_Percent": float(zero_minutes / total_minutes * 100),
            "Active_Time_Percent": float(active_minutes / total_minutes * 100),
            "Detected_Cycles_Count": len(df_cycles),
            "Mean_Cycle_Duration_Minutes": float(df_cycles["Duration_Minutes"].mean()) if len(df_cycles) > 0 else 0.0,
            "Median_Cycle_Duration_Minutes": float(df_cycles["Duration_Minutes"].median()) if len(df_cycles) > 0 else 0.0,
            "Mean_Peak_Power_Watts": float(df_cycles["Peak_Power_Watts"].mean()) if len(df_cycles) > 0 else 0.0,
            "Max_Peak_Power_Watts": float(df_cycles["Peak_Power_Watts"].max()) if len(df_cycles) > 0 else 0.0,
            "Mean_Energy_Per_Cycle_kWh": float(df_cycles["Energy_kWh"].mean()) if len(df_cycles) > 0 else 0.0,
            "Total_Energy_kWh": float(df_cycles["Energy_kWh"].sum()) if len(df_cycles) > 0 else 0.0,
        }

        self.household_stats[house_id] = stats
        return stats

    def plot_24h_representative_trace(self, house_id: str = "House 1") -> Path:
        """
        Visualization 1: Representative 24-hour trace of whole-house aggregate and washing-machine power.
        """
        df = self.household_dfs[house_id]
        df_cycles = self.detected_cycles[house_id]

        if len(df_cycles) == 0:
            start_date = df["DateTime"].min() + pd.Timedelta(days=1)
        else:
            # Pick a day with an active cycle for an informative plot
            cycle_time = df_cycles.iloc[0]["Start_Time"]
            start_date = cycle_time.floor("D")

        end_date = start_date + pd.Timedelta(hours=24)
        slice_df = df[(df["DateTime"] >= start_date) & (df["DateTime"] <= end_date)]

        fig, ax = plt.subplots(figsize=(14, 5))
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

        ax.plot(slice_df["DateTime"], slice_df["Aggregate"], label="Whole-House Aggregate Power", color="#1f77b4", alpha=0.6, lw=1.2)
        ax.plot(slice_df["DateTime"], slice_df["WashingMachine"], label="Washing Machine Power", color="#d62728", lw=1.8)
        
        ax.fill_between(slice_df["DateTime"], slice_df["WashingMachine"], color="#d62728", alpha=0.25)
        ax.set_title(f"Representative 24-Hour Electricity Profile: {house_id} ({start_date.strftime('%Y-%m-%d')})", fontsize=13, fontweight="bold")
        ax.set_ylabel("Power (Watts)", fontsize=11)
        ax.set_xlabel("Time of Day", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", frameon=True)
        ax.tick_params(axis="x", rotation=25)

        plot_path = self.plots_dir / f"{house_id.lower().replace(' ', '')}_24h_trace.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"24-Hour trace plot saved to: {plot_path}")
        return plot_path

    def plot_detected_cycles(self, house_id: str = "House 1", num_cycles: int = 3) -> Path:
        """
        Visualization 2: Detailed zoom-in plot showing detected washing machine cycles with activation threshold.
        """
        df = self.household_dfs[house_id]
        df_cycles = self.detected_cycles[house_id]

        if len(df_cycles) == 0:
            logger.warning(f"No cycles to plot for {house_id}")
            return self.plots_dir

        # Focus on a window containing the requested number of cycles
        cycle_subset = df_cycles.head(num_cycles)
        t_start = cycle_subset.iloc[0]["Start_Time"] - pd.Timedelta(minutes=30)
        t_end = cycle_subset.iloc[-1]["End_Time"] + pd.Timedelta(minutes=30)

        slice_df = df[(df["DateTime"] >= t_start) & (df["DateTime"] <= t_end)]

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(slice_df["DateTime"], slice_df["WashingMachine"], label="Washing Machine Power (W)", color="#d62728", lw=1.6)
        ax.axhline(self.cycle_config["on_power_threshold_watts"], color="black", linestyle="--", lw=1.2, label=f"Activation Threshold ({self.cycle_config['on_power_threshold_watts']}W)")

        # Shade detected cycle regions
        for idx, row in cycle_subset.iterrows():
            ax.axvspan(row["Start_Time"], row["End_Time"], color="#2ca02c", alpha=0.2, label="Detected Cycle Boundary" if idx == 0 else "")
            # Annotate peak and energy
            mid_time = row["Start_Time"] + (row["End_Time"] - row["Start_Time"]) / 2
            ax.annotate(
                f"Duration: {row['Duration_Minutes']:.0f}m\nPeak: {row['Peak_Power_Watts']:.0f}W\nEnergy: {row['Energy_kWh']:.2f}kWh",
                xy=(mid_time, row["Peak_Power_Watts"] * 0.7),
                xytext=(0, 20),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="#ffffcc", ec="gray", lw=0.8),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color="black"),
            )

        ax.set_title(f"Detected Washing Machine Operating Cycles: {house_id}", fontsize=13, fontweight="bold")
        ax.set_ylabel("Power (Watts)", fontsize=11)
        ax.set_xlabel("Timestamp", fontsize=11)
        ax.legend(loc="upper right", frameon=True)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.tick_params(axis="x", rotation=25)

        plot_path = self.plots_dir / f"{house_id.lower().replace(' ', '')}_detected_cycles.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Detected cycles plot saved to: {plot_path}")
        return plot_path

    def plot_cross_household_comparison(self) -> Path:
        """
        Visualization 3: Cross-household comparison of cycle duration, peak power, and energy consumption.
        """
        all_cycles = []
        for h_id, c_df in self.detected_cycles.items():
            if len(c_df) > 0:
                all_cycles.append(c_df)

        if not all_cycles:
            logger.warning("No cycle data available across households for cross-comparison plot.")
            return self.plots_dir

        combined_df = pd.concat(all_cycles, ignore_index=True)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        palette = sns.color_palette("muted")

        # 1. Cycle Duration
        sns.boxplot(data=combined_df, x="House", y="Duration_Minutes", ax=axes[0], palette=palette)
        axes[0].set_title("Cycle Duration Comparison", fontsize=11, fontweight="bold")
        axes[0].set_ylabel("Duration (Minutes)", fontsize=10)
        axes[0].set_xlabel("")

        # 2. Peak Power
        sns.boxplot(data=combined_df, x="House", y="Peak_Power_Watts", ax=axes[1], palette=palette)
        axes[1].set_title("Peak Power Draw Comparison", fontsize=11, fontweight="bold")
        axes[1].set_ylabel("Peak Power (Watts)", fontsize=10)
        axes[1].set_xlabel("Household", fontsize=10)

        # 3. Energy per Cycle
        sns.boxplot(data=combined_df, x="House", y="Energy_kWh", ax=axes[2], palette=palette)
        axes[2].set_title("Energy Consumption per Cycle", fontsize=11, fontweight="bold")
        axes[2].set_ylabel("Energy (kWh)", fontsize=10)
        axes[2].set_xlabel("")

        for ax in axes:
            ax.grid(True, linestyle="--", alpha=0.5)

        fig.suptitle("Cross-Household Washing Machine Cycle Characteristics (REFIT)", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()

        plot_path = self.plots_dir / "cross_household_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Cross-household comparison plot saved to: {plot_path}")
        return plot_path

    def run_eda(self, sample_max_days: Optional[int] = 30) -> Dict[str, Dict]:
        """
        Executes complete washing machine exploratory data analysis across target households.
        """
        target_houses = [
            ("House 1", "Appliance5"),
            ("House 2", "Appliance2"),
            ("House 3", "Appliance5"),
            ("House 5", "Appliance3"),
        ]

        logger.info(f"Running Washing Machine EDA on {len(target_houses)} households...")
        for h_id, wm_col in target_houses:
            try:
                self.analyze_household(h_id, wm_col, sample_max_days=sample_max_days)
            except Exception as e:
                logger.error(f"Failed analyzing {h_id}: {e}")

        # Generate required plots
        if "House 1" in self.household_dfs:
            self.plot_24h_representative_trace("House 1")
            self.plot_detected_cycles("House 1")

        self.plot_cross_household_comparison()
        return self.household_stats


if __name__ == "__main__":
    eda = WashingMachineEDA()
    try:
        stats = eda.run_eda(sample_max_days=14)
        print("\n=== WASHING MACHINE USAGE SUMMARY ACROSS HOUSEHOLDS ===")
        for h, s in stats.items():
            print(f"\n--- {h} ---")
            for k, v in s.items():
                print(f"  {k}: {v}")
    except Exception as e:
        print(f"\n[INFO] {e}")
