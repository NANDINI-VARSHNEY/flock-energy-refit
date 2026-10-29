"""
Master End-to-End Pipeline Runner for Flock Energy REFIT Project.
Orchestrates:
1. Raw Data Cleaning (House 1)
2. Washing-Machine Exploratory Data Analysis (Houses 1, 2, 3, 5)
3. Appliance Disaggregation Machine Learning Model Training & Evaluation
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to system path
sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    RAW_DATA_DIR,
    CLEAN_DATA_DIR,
    PROCESSED_DATA_DIR,
    PLOTS_DIR,
    MODELS_DIR,
    DATASET_METADATA,
)
from clean_raw import RawDataCleaner
from eda_washing_machine import WashingMachineEDA
from disaggregation_model import DisaggregationPipeline
from mock_refit_generator import generate_sample_refit_raw_house1, generate_sample_clean_households

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PipelineRunner")


def main():
    parser = argparse.ArgumentParser(description="Flock Energy REFIT Data Science Pipeline")
    parser.add_argument("--demo", action="store_true", help="Run with synthetic verification dataset if raw files are not yet downloaded.")
    parser.add_argument("--skip-cleaning", action="store_true", help="Skip raw cleaning stage.")
    parser.add_argument("--skip-eda", action="store_true", help="Skip washing machine EDA stage.")
    parser.add_argument("--skip-ml", action="store_true", help="Skip machine learning modeling stage.")
    args = parser.parse_args()

    print("=" * 80)
    print("   FLOCK ENERGY DATA SCIENCE TAKE-HOME PIPELINE")
    print("   REFIT Data Cleaning and Appliance Disaggregation")
    print("=" * 80)

    # Check data availability
    raw_h1 = RAW_DATA_DIR / "House1.csv"
    clean_h1 = CLEAN_DATA_DIR / "CLEAN_House1.csv"

    if not raw_h1.exists() and not args.demo:
        print(f"\n[NOTICE] Raw dataset file not found at: {raw_h1}")
        print("To run with full Strathclyde raw data, download REFIT_RAW_081116.7z:")
        print(f"  URL: {DATASET_METADATA['raw_url']}")
        print(f"  Extract 'House1.csv' into: {RAW_DATA_DIR}\n")
        print("Or run this script with '--demo' to generate realistic sample data and verify the entire pipeline:")
        print("  python src/run_pipeline.py --demo\n")
        
        user_choice = input("Would you like to initialize sample verification data now? (y/n): ").strip().lower()
        if user_choice == "y":
            args.demo = True
        else:
            print("Aborting. Please configure data files and re-run.")
            return

    if args.demo:
        logger.info("Initializing synthetic REFIT verification dataset...")
        if not raw_h1.exists():
            generate_sample_refit_raw_house1(raw_h1, days=7)
        generate_sample_clean_households(CLEAN_DATA_DIR, days=14)

    pipeline_report = {
        "metadata": DATASET_METADATA,
        "stages": {},
    }

    # ---------------------------------------------------------
    # STAGE 1: RAW DATA CLEANING (HOUSE 1)
    # ---------------------------------------------------------
    if not args.skip_cleaning:
        print("\n" + "#" * 80)
        print("STAGE 1: RAW DATA INSPECTION & CLEANING (HOUSE 1)")
        print("#" * 80)
        cleaner = RawDataCleaner()
        df_clean_h1, cleaning_report = cleaner.run_cleaning_pipeline()
        pipeline_report["stages"]["cleaning_house1"] = cleaning_report
        print("\nStage 1 Results Summary:")
        for k, v in cleaning_report.items():
            print(f"  {k}: {v}")

    # ---------------------------------------------------------
    # STAGE 2: WASHING MACHINE EXPLORATORY DATA ANALYSIS
    # ---------------------------------------------------------
    if not args.skip_eda:
        print("\n" + "#" * 80)
        print("STAGE 2: WASHING MACHINE EXPLORATORY DATA ANALYSIS (HOUSES 1, 2, 3, 5)")
        print("#" * 80)
        eda = WashingMachineEDA()
        eda_stats = eda.run_eda(sample_max_days=30)
        pipeline_report["stages"]["washing_machine_eda"] = eda_stats
        print("\nStage 2 Results Summary Across Households:")
        for h, s in eda_stats.items():
            print(f"\n--- {h} ---")
            for k, v in s.items():
                print(f"  {k}: {v}")

    # ---------------------------------------------------------
    # STAGE 3: APPLIANCE DISAGGREGATION MODEL
    # ---------------------------------------------------------
    if not args.skip_ml:
        print("\n" + "#" * 80)
        print("STAGE 3: APPLIANCE DISAGGREGATION MACHINE LEARNING MODEL")
        print("#" * 80)
        
        # Load cleaned House 1 dataset from Stage 1 or Clean directory
        h1_processed = PROCESSED_DATA_DIR / "House1_cleaned_1min.csv"
        if h1_processed.exists():
            df_model_input = pd.read_csv(h1_processed)
            df_model_input["DateTime"] = pd.to_datetime(df_model_input["DateTime"])
            # Map Appliance5 to WashingMachine if needed
            if "Appliance5" in df_model_input.columns and "WashingMachine" not in df_model_input.columns:
                df_model_input["WashingMachine"] = df_model_input["Appliance5"]
        elif (CLEAN_DATA_DIR / "CLEAN_House1.csv").exists():
            eda_loader = WashingMachineEDA()
            df_model_input = eda_loader.load_and_resample_household("House 1", "Appliance5")
        else:
            raise FileNotFoundError("Cleaned House 1 data not available for modeling.")

        ml_pipeline = DisaggregationPipeline()
        df_featured = ml_pipeline.engineer_features(df_model_input)
        train_df, test_df = ml_pipeline.split_data(df_featured, split_method="chronological")
        model_results = ml_pipeline.train_and_evaluate(train_df, test_df)
        
        # Plot evaluation
        ml_pipeline.plot_actual_vs_predicted(test_df)
        
        # Failure analysis
        failure_analysis = ml_pipeline.analyze_failure_modes(test_df)
        
        pipeline_report["stages"]["disaggregation_model"] = {
            "evaluation_metrics": model_results,
            "failure_analysis": failure_analysis,
        }

        print("\nStage 3 Model Performance Comparison on Held-Out Test Set:")
        metrics_df = pd.DataFrame(model_results).T
        print(metrics_df[["Model", "MAE_Watts", "RMSE_Watts", "Energy_Error_Percent", "Activation_F1_Score"]].to_string())

    # Save complete pipeline report
    report_path = PROCESSED_DATA_DIR / "pipeline_run_results.json"
    with open(report_path, "w") as f:
        json.dump(pipeline_report, f, indent=2, default=str)
    
    print("\n" + "=" * 80)
    print(f"PIPELINE RUN COMPLETE! All results & plots saved.")
    print(f"Results report: {report_path}")
    print(f"Plots directory: {PLOTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
