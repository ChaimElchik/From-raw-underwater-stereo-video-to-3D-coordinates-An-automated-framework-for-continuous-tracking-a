import os
import pandas as pd

results_dir = "ReID_V2_Test_Results"

# Load the two CSVs
frag_path = os.path.join(results_dir, "Custom_Tracker_Improvement_Stats.csv")
fp_path = os.path.join(results_dir, "False_Positive_ID_Stats.csv")

try:
    df_frag = pd.read_csv(frag_path)
    df_fp = pd.read_csv(fp_path)
    
    # Merge on 'Video' column
    df_merged = pd.merge(df_frag, df_fp, on='Video', how='inner')
    
    # Clean up and select the relevant columns
    # From df_frag: Default_BoTSORT_IDs, Custom_BoTSORT_IDs, Custom_Tracker_Reduction
    # From df_fp: Default_FP_IDs, Custom_FP_IDs
    
    df_final = df_merged[[
        'Video',
        'Default_BoTSORT_IDs',
        'Custom_BoTSORT_IDs',
        'Custom_Tracker_Reduction', # Fragmented pieces stitched together
        'Default_FP_IDs',
        'Custom_FP_IDs'
    ]].copy()
    
    # Calculate FP reduction explicitly
    df_final['Ghost_Tracks_Prevented'] = df_final['Default_FP_IDs'] - df_final['Custom_FP_IDs']
    
    # Rename columns for extreme clarity
    df_final.rename(columns={
        'Default_BoTSORT_IDs': 'Total_IDs_Default',
        'Custom_BoTSORT_IDs': 'Total_IDs_Custom',
        'Custom_Tracker_Reduction': 'Fragmented_Fish_Stitched',
        'Default_FP_IDs': 'Ghost_Tracks_Default',
        'Custom_FP_IDs': 'Ghost_Tracks_Custom'
    }, inplace=True)
    
    # Sort by the videos where Custom Tracker had the biggest impact
    # Impact = Fragmented Fish Stitched + Ghost Tracks Prevented
    df_final['Total_Improvement'] = df_final['Fragmented_Fish_Stitched'] + df_final['Ghost_Tracks_Prevented']
    df_final.sort_values(by='Total_Improvement', ascending=False, inplace=True)
    
    # Drop the sorting column so it's clean
    df_final.drop(columns=['Total_Improvement'], inplace=True)

    out_path = os.path.join(results_dir, "Combined_Tracking_Improvement_Summary.csv")
    df_final.to_csv(out_path, index=False)
    
    print(f"Successfully combined {len(df_final)} video reports!")
    print(f"Saved to: {out_path}")
    
except Exception as e:
    print(f"Error merging CSVs: {e}")
