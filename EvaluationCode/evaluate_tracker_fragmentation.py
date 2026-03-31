import os
import pandas as pd
from glob import glob
import numpy as np

results_dir = "ReID_V2_Test_Results"

# Find all default botsort csvs
default_csvs = glob(os.path.join(results_dir, "*_default_botsort_tracks.csv"))

results = []

for default_path in default_csvs:
    base_name = os.path.basename(default_path).replace("_default_botsort_tracks.csv", "")
    raw_path = os.path.join(results_dir, f"{base_name}_raw_tracks.csv")
    reid_path = os.path.join(results_dir, f"{base_name}_reid_tracks.csv")
    
    # Check if raw_tracks exists (it should based on Test_Re_ID_V2_Advanced.py)
    if not os.path.exists(raw_path):
        continue
        
    try:
        df_default = pd.read_csv(default_path)
        default_ids = df_default['id'].nunique() if not df_default.empty else 0
    except Exception:
        default_ids = 0
        
    try:
        df_raw = pd.pd.read_csv(raw_path)
        raw_ids = df_raw['id'].nunique() if not df_raw.empty else 0
    except Exception:
        try:
            df_raw = pd.read_csv(raw_path)
            raw_ids = df_raw['id'].nunique() if not df_raw.empty else 0
        except Exception:
            raw_ids = 0
            
    try:
        df_reid = pd.read_csv(reid_path)
        reid_ids = df_reid['id'].nunique() if not df_reid.empty else 0
    except Exception:
        reid_ids = 0
        
    # Calculate improvements
    custom_improvement = default_ids - raw_ids
    
    results.append({
        'Video': base_name,
        'Default_BoTSORT_IDs': default_ids,
        'Custom_BoTSORT_IDs': raw_ids,
        'V2_ReID_IDs': reid_ids,
        'Custom_Tracker_Reduction': custom_improvement,
        'Percent_Reduction': (custom_improvement / default_ids * 100) if default_ids > 0 else 0
    })

# Convert to DataFrame
df_results = pd.DataFrame(results)

if not df_results.empty:
    print(f"Evaluated {len(df_results)} videos.")
    print("\n--- Summary Statistics ---")
    print(f"Average IDs (Default BoT-SORT):  {df_results['Default_BoTSORT_IDs'].mean():.2f}")
    print(f"Average IDs (Custom BoT-SORT):   {df_results['Custom_BoTSORT_IDs'].mean():.2f}")
    print(f"Average IDs (V2 ReID):           {df_results['V2_ReID_IDs'].mean():.2f}")
    
    total_default = df_results['Default_BoTSORT_IDs'].sum()
    total_custom = df_results['Custom_BoTSORT_IDs'].sum()
    total_reid = df_results['V2_ReID_IDs'].sum()
    
    print("\n--- Totals Across All Videos ---")
    print(f"Total IDs (Default BoT-SORT): {total_default}")
    print(f"Total IDs (Custom BoT-SORT):  {total_custom}")
    print(f"Total IDs (V2 ReID):          {total_reid}")
    
    print("\n--- Fragmentation Reduction ---")
    abs_reduction = total_default - total_custom
    pct_reduction = (abs_reduction / total_default * 100) if total_default > 0 else 0
    print(f"Custom BoT-SORT eliminated {abs_reduction} fragmented IDs ({pct_reduction:.2f}% reduction from Default).")
    
    abs_reid_reduction = total_custom - total_reid
    pct_reid_reduction = (abs_reid_reduction / total_custom * 100) if total_custom > 0 else 0
    print(f"V2 Re-ID eliminated an additional {abs_reid_reduction} fragmented IDs ({pct_reid_reduction:.2f}% reduction from Custom BoT-SORT).")
    
    total_reduction = total_default - total_reid
    total_pct = (total_reduction / total_default * 100) if total_default > 0 else 0
    print(f"\nTotal Pipeline Improvement: {total_reduction} fragmented IDs eliminated ({total_pct:.2f}% overall reduction).")
    
    df_results.to_csv(os.path.join(results_dir, "Custom_Tracker_Improvement_Stats.csv"), index=False)
else:
    print("No matching tracking CSVs found in ReID_V2_Test_Results.")

