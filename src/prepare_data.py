"""
Data preparation script for California Housing dataset.
Supports creating baseline version (v1) and modified version (v2).
"""
import argparse
import os
import pandas as pd
from sklearn.datasets import fetch_california_housing


def generate_v1(output_path: str = "data/dataset.csv") -> pd.DataFrame:
    """Download baseline California Housing dataset and save to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.copy()
    df.to_csv(output_path, index=False)
    print(f"[v1] Baseline dataset generated at '{output_path}': {len(df)} rows, {len(df.columns)} columns.")
    return df


def generate_v2(output_path: str = "data/dataset.csv", num_synthetic_rows: int = 100) -> pd.DataFrame:
    """
    Generate v2 of the dataset by appending realistic synthetic observations
    representing new housing surveys and applying minor precision cleanup.
    """
    if not os.path.exists(output_path):
        df = generate_v1(output_path)
    else:
        df = pd.read_csv(output_path)

    # Sample representative rows and add minor realistic jitter to simulate newly collected survey data
    sampled = df.sample(n=num_synthetic_rows, random_state=42).copy()
    # Add slight random perturbation
    numeric_cols = [c for c in df.columns if c != "MedHouseVal"]
    sampled[numeric_cols] = sampled[numeric_cols] * 1.02
    # Ensure MedHouseVal stays within valid range
    sampled["MedHouseVal"] = (sampled["MedHouseVal"] * 1.01).clip(upper=5.0)

    # Append new survey records to baseline
    df_v2 = pd.concat([df, sampled], ignore_index=True)
    df_v2.to_csv(output_path, index=False)
    print(f"[v2] Updated dataset generated at '{output_path}': {len(df_v2)} rows (+{num_synthetic_rows} new records), {len(df_v2.columns)} columns.")
    return df_v2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare California Housing dataset")
    parser.add_argument("--version", choices=["v1", "v2"], default="v1", help="Dataset version to generate")
    parser.add_argument("--output", default="data/dataset.csv", help="Output CSV path")
    args = parser.parse_args()

    if args.version == "v1":
        generate_v1(args.output)
    else:
        generate_v2(args.output)
