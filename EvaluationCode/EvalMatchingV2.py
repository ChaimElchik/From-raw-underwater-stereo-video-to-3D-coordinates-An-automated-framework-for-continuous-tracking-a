import pandas as pd
import os
import sys

# Import your matching scripts
import strV5
import strV3
import strV4

def compare_stereo_matching(video_names, ground_truth_dict, mat_file_path='stereoParams_Dep4.mat', mots_folder='mots'):
    """
    Runs 3 versions of stereo matching and compares against a ground truth dictionary.
    Evaluates using Precision, Recall, and F1-Score.
    """
    
    results = []
    versions = {
        'Version 5': strV5,
        'Version 3': strV3,
        'Version 4': strV4
    }

    print(f"Starting evaluation on {len(video_names)} videos...")
    
    for video in video_names:
        print(f"\nProcessing Video: {video}")
        
        # 1. Construct File Paths
        f1_path = os.path.join(mots_folder, f"{video}_1_clean.txt")
        f2_path = os.path.join(mots_folder, f"{video}_2_clean.txt")
        
        if not os.path.exists(f1_path) or not os.path.exists(f2_path):
            print(f"  [Error] Files not found for {video}. Skipping.")
            continue
            
        # 2. Retrieve Ground Truth
        # Handle string/int key differences
        if video in ground_truth_dict:
            gt_entry = ground_truth_dict[video]
        elif int(video) in ground_truth_dict:
            gt_entry = ground_truth_dict[int(video)]
        else:
            print(f"  [Warning] No ground truth found for {video}. Skipping.")
            continue
            
        gt_ids1 = gt_entry[0]
        gt_ids2 = gt_entry[1]
        
        # Create Ground Truth Map: {id1: id2}
        gt_map = dict(zip(gt_ids1, gt_ids2))
        
        # 3. Run Each Version
        for v_name, module in versions.items():
            print(f"  Running {v_name}...", end=" ", flush=True)
            try:
                # Run the algorithm
                pred_mapping, _ = module.run_geometric_matching(f1_path, f2_path, mat_file_path)
                
                # Convert prediction to dictionary: {id1: predicted_id2}
                pred_map = dict(zip(pred_mapping['id1'], pred_mapping['id2']))
                
                # --- METRICS CALCULATION ---
                tp = 0  # True Positive: Correct match found
                fp = 0  # False Positive: Match found, but it was wrong OR not in GT
                fn = 0  # False Negative: GT had a match, but we missed it
                
                # Check 1: Iterate over Ground Truth (Find TPs and FNs)
                for gt_id1, gt_id2 in gt_map.items():
                    if gt_id1 in pred_map:
                        if pred_map[gt_id1] == gt_id2:
                            tp += 1
                        else:
                            # We predicted something, but it was wrong (Counted as FP below usually, 
                            # but strictly this specific GT pair is a Miss/FN)
                            fn += 1 
                    else:
                        # We predicted nothing for this ID
                        fn += 1
                
                # Check 2: Iterate over Predictions (Find FPs)
                for pred_id1, pred_id2 in pred_map.items():
                    # If this ID1 isn't in GT at all, it's a hallucination (FP)
                    if pred_id1 not in gt_map:
                        fp += 1
                    # If ID1 is in GT, but we predicted the WRONG ID2, it's an FP
                    elif gt_map[pred_id1] != pred_id2:
                        fp += 1
                
                # Calculate Metrics (Guard against division by zero)
                precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
                recall    = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
                
                if (precision + recall) > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    f1 = 0.0
                
                results.append({
                    'Video': video,
                    'Version': v_name,
                    'F1-Score': round(f1/100, 3), # Normalized 0-1 usually preferred for F1
                    'Precision (%)': round(precision, 1),
                    'Recall (%)': round(recall, 1),
                    'TP': tp,
                    'FP': fp,
                    'FN': fn
                })
                print(f"Done. F1: {f1/100:.3f}")
                
            except Exception as e:
                print(f"Failed! Error: {e}")
                results.append({
                    'Video': video,
                    'Version': v_name,
                    'F1-Score': 0.0,
                    'Precision (%)': 0.0,
                    'Recall (%)': 0.0,
                    'TP': 0, 'FP': 0, 'FN': len(gt_map),
                    'Error': str(e)
                })

    # Create Final DataFrame
    df_results = pd.DataFrame(results)
    
    # Reorder columns
    cols = ['Video', 'Version', 'F1-Score', 'Precision (%)', 'Recall (%)', 'TP', 'FP', 'FN']
    if 'Error' in df_results.columns:
        cols.append('Error')
        
    return df_results[cols]

# ==========================================
#              USER CONFIGURATION
# ==========================================
if __name__ == "__main__":
    
    my_videos = ["8", "10", "13", "14", "15", "16", "23", "129", "406"]
    my_mat_file = "stereoParams_Dep1.mat" 
    
    # (Paste your huge my_ground_truth dict here)
    my_ground_truth = {
        "8": [[8,7,17,2,5,18,6,4,3], [14,13,16,1,11,15,12,9,10]],
        "10": [[7,6,3,4,5,16,17,13,2], [12,11,8,9,10,18,15,14,1]],
        "13": [[11,12,13,10,9,8,7,2,1,4,3,5], [16,15,14,25,24,17,22,21,18,19,20,23]],
        "14": [[4,3,2,25,5,11,9,10,6,1,7,8], [15,13,12,28,16,18,20,22,19,17,21,14]],
        "15": [[13,12,11,1,14,4,2,3,6,7,17,16,8], [31,29,30,32,28,34,19,20,22,23,33,26,24]],
        "16": [[4,7,32,10,2,1,24,3,5,6,8,11,9,30], [18,19,33,22,17,16,25,15,13,14,12,21,23,27]],
        "23": [[5,4,3,2,22,17,16,15,7,1,6], [12,11,10,9,21,18,20,19,14,8,13]],
        "129": [[9,7,8,2,1,4,3,6,5,11,10,12], [5,8,9,7,6,1,4,2,3,11,10,12]],
        "406": [[9,6,7,1,2,3,5,4,10,8], [7,8,2,1,4,3,5,9,10,6]]
    }

    final_report = compare_stereo_matching(
        my_videos, 
        my_ground_truth, 
        mat_file_path=my_mat_file,
        mots_folder='mots'
    )

    print("\n" + "="*50)
    print("      DETAILED PERFORMANCE REPORT      ")
    print("="*50)
    print(final_report.to_string(index=False))
    
    # Calculate Average F1 per Version for quick summary
    print("\n" + "="*20)
    print("   AVERAGE SCORES   ")
    print("="*20)
    print(final_report.groupby('Version')[['F1-Score', 'Precision (%)', 'Recall (%)']].mean())
    
    final_report.to_csv("comparison_results.csv", index=False)