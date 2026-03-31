import os
import glob
import pandas as pd

results_dir = "ReID_Test_Results"
metrics_files = glob.glob(os.path.join(results_dir, "*_metrics.csv"))

if not metrics_files:
    print("No metrics CSVs found.")
    exit(0)

# Ensure we don't recursively add our own combined file if run multiple times
metrics_files = [f for f in metrics_files if "Combined_Metrics.csv" not in f]

dfs = []
for f in metrics_files:
    try:
        df = pd.read_csv(f)
        dfs.append(df)
    except Exception as e:
        print(f"Error reading {f}: {e}")

if dfs:
    combined_df = pd.concat(dfs, ignore_index=True)
    out_path = os.path.join(results_dir, "Combined_Metrics.csv")
    combined_df.to_csv(out_path, index=False)
    print(f"✅ Successfully combined {len(dfs)} individual metrics tracks into {out_path}")
else:
    print("Failed to combine any metrics.")
