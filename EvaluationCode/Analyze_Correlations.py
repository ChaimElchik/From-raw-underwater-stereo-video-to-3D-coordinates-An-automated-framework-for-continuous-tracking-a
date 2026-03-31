import os
import pandas as pd
import json

CODE_DIR = "/Users/chaim/Documents/Thesis_ReWrite/Final_Pipeline_Code"
csv_path = os.path.join(CODE_DIR, "evaluation_results.csv")
gt_dir = os.path.join(CODE_DIR, "Ground_Truth_Matches")

def analyze_correlations():
    print("=== Analyzing Correlations ===")
    
    if not os.path.exists(csv_path):
        print("evaluation_results.csv not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    video_lengths = []
    total_ids = []
    
    for _, row in df.iterrows():
        video_id = str(int(row["video"]))
        
        # 1. Total IDs from the GT Pairs count
        num_ids = int(row["total_gt_pairs"])
        
        # 2. Video length from original MOT frames
        mot_path = f"/Users/chaim/Documents/Thesis_ReWrite/mots/{video_id}_1.txt"
        max_frame = 0
        if os.path.exists(mot_path):
            mot_df = pd.read_csv(mot_path, header=None, names=["frame", "id", "x", "y", "w", "h", "conf", "cls", "vis"])
            if not mot_df.empty:
                max_frame = int(mot_df["frame"].max())
        
        video_lengths.append(max_frame)
        total_ids.append(num_ids)
        
    df["Video_Length_Frames"] = video_lengths
    
    # Calculate correlations
    corr_len_p = df["Video_Length_Frames"].corr(df["Precision"])
    corr_len_r = df["Video_Length_Frames"].corr(df["Recall"])
    corr_len_f1 = df["Video_Length_Frames"].corr(df["F1_Score"])
    
    corr_ids_p = df["total_gt_pairs"].corr(df["Precision"])
    corr_ids_r = df["total_gt_pairs"].corr(df["Recall"])
    corr_ids_f1 = df["total_gt_pairs"].corr(df["F1_Score"])
    
    print("\n--- Correlation with Video Length (Max Frame) ---")
    print(f"Video Length vs Precision: {corr_len_p:.4f}")
    print(f"Video Length vs Recall:    {corr_len_r:.4f}")
    print(f"Video Length vs F1-Score:  {corr_len_f1:.4f}")
    
    print("\n--- Correlation with Number of IDs (Total GT Pairs) ---")
    print(f"Number of IDs vs Precision: {corr_ids_p:.4f}")
    print(f"Number of IDs vs Recall:    {corr_ids_r:.4f}")
    print(f"Number of IDs vs F1-Score:  {corr_ids_f1:.4f}")
    
    # Optional: General interpretation
    def interpret(val):
        abs_v = abs(val)
        if abs_v < 0.3: return "Weak/None"
        if abs_v < 0.7: return "Moderate"
        return "Strong"
        
    print("\nInterpretation:")
    print(f"Video Length vs F1 is {interpret(corr_len_f1)} ({corr_len_f1:.4f})")
    print(f"Number of IDs vs F1 is {interpret(corr_ids_f1)} ({corr_ids_f1:.4f})")

if __name__ == "__main__":
    analyze_correlations()
