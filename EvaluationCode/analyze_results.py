import pandas as pd
import numpy as np

def analyze_metrics():
    df = pd.read_csv("ReID_Test_Results/Combined_Metrics.csv")
    
    # We want to group by Method and find the mean for most metrics, 
    # but the sum for IDSW and Frag and total IDs.
    
    # Summarize per method
    summary = df.groupby('Method').agg({
        'MOTA': 'mean',
        'IDF1': 'mean',
        'IDP': 'mean',
        'IDR': 'mean',
        'IDSW': 'sum',
        'PT': 'sum',
        'ML': 'sum',
        'MT': 'sum',
        'Number of IDs': 'sum',
        'Amount of Merges Missed': 'sum',
        'Merges RE-ID Correct': 'sum',
        'Merges RE-ID Correct Merge but FP': 'sum',
        'Incorrect Merges RE-ID': 'sum'
    }).reset_index()
    
    # Order the methods
    order = ['Default BoT-SORT', 'Custom Baseline', 'Custom After-ReID']
    summary['Method'] = pd.Categorical(summary['Method'], categories=order, ordered=True)
    summary = summary.sort_values('Method')
    
    print("\n" + "="*80)
    print("TRACKING PERFORMANCE ANALYSIS: Custom Configuration vs. Re-ID".center(80))
    print("="*80)
    
    print("\n1. OVERALL AVERAGES (Across all videos)")
    print("-" * 80)
    print(f"{'Method':<20} | {'MOTA (%)':<10} | {'IDF1 (%)':<10} | {'IDP (%)':<10} | {'IDR (%)':<10}")
    print("-" * 80)
    for _, row in summary.iterrows():
        print(f"{row['Method']:<20} | {row['MOTA']*100:>8.2f} % | {row['IDF1']*100:>8.2f} % | {row['IDP']*100:>8.2f} % | {row['IDR']*100:>8.2f} %")
        
    print("\n2. AGGREGATE TOTALS (Sum across all videos)")
    print("-" * 80)
    print(f"{'Method':<20} | {'Total IDs':<10} | {'ID Switches':<12} | {'Mostly Tracked':<15} | {'Mostly Lost':<12}")
    print("-" * 80)
    for _, row in summary.iterrows():
        print(f"{row['Method']:<20} | {int(row['Number of IDs']):<10} | {int(row['IDSW']):<12} | {int(row['MT']):<15} | {int(row['ML']):<12}")

    print("\n3. RE-ID MERGE IMPACT (Custom After-ReID Only)")
    print("-" * 80)
    reid_row = summary[summary['Method'] == 'Custom After-ReID'].iloc[0]
    total_merges = reid_row['Merges RE-ID Correct'] + reid_row['Merges RE-ID Correct Merge but FP'] + reid_row['Incorrect Merges RE-ID']
    
    print(f"Total Merges Attempted:               {int(total_merges)}")
    print(f"Correct Merges (Fixed track):         {int(reid_row['Merges RE-ID Correct'])}")
    print(f"False Positive Merges (Non-GT track): {int(reid_row['Merges RE-ID Correct Merge but FP'])}")
    print(f"Incorrect Merges (Identity Switch):   {int(reid_row['Incorrect Merges RE-ID'])}")
    print(f"Remaining Missed Merges:              {int(reid_row['Amount of Merges Missed'])}")
    
    print("\n" + "="*80)

    # Save summary to CSV
    summary_out_path = "ReID_Test_Results/Summary_Metrics.csv"
    summary.to_csv(summary_out_path, index=False)
    print(f"\n✅ Successfully saved summary metrics to {summary_out_path}")
if __name__ == "__main__":
    analyze_metrics()
