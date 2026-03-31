import os
import pandas as pd
import numpy as np
from glob import glob
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

def calculate_iou(boxA, boxB):
    """Calculates Intersection over Union for two bounding boxes [xmin, ymin, width, height]"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea)

def load_ground_truth(filepath):
    """Loads GT file: frame,id,x,y,x_offset,y_offset"""
    df = pd.read_csv(filepath)
    df.rename(columns={'x': 'xmin', 'y': 'ymin', 'x_offset': 'w', 'y_offset': 'h'}, inplace=True)
    return df

def load_predictions(filepath, frame_offset=1):
    """Loads RAW/Pred file: frame,id,x,y,w,h,xmin,ymin,xmax,ymax"""
    df = pd.read_csv(filepath)
    # Align 0-indexed predictions to 1-indexed GT
    df['frame'] = df['frame'] + frame_offset 
    return df

def count_false_positive_ids(gt_df, pred_df, iou_threshold=0.5):
    """Counts predicted track IDs that NEVER overlap with a Ground Truth object."""
    if gt_df is None or gt_df.empty or pred_df is None or pred_df.empty:
        if pred_df is not None:
             return pred_df['id'].nunique() # All IDs are false positives if no ground truth
        return 0

    votes = defaultdict(lambda: defaultdict(int))
    frames = pred_df['frame'].unique()
    
    for f in frames:
        gt_frame = gt_df[gt_df['frame'] == f]
        pred_frame = pred_df[pred_df['frame'] == f]
        
        if gt_frame.empty or pred_frame.empty:
            continue
            
        gt_boxes = gt_frame[['xmin', 'ymin', 'w', 'h']].values
        gt_ids = gt_frame['id'].values
        
        pred_boxes = pred_frame[['xmin', 'ymin', 'w', 'h']].values
        pred_ids = pred_frame['id'].values
        
        iou_matrix = np.zeros((len(pred_ids), len(gt_ids)))
        for i, p_box in enumerate(pred_boxes):
            for j, gt_box in enumerate(gt_boxes):
                iou_matrix[i, j] = calculate_iou(p_box, gt_box)
                
        cost_matrix = 1.0 - iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= iou_threshold:
                p_id = pred_ids[r]
                g_id = gt_ids[c]
                votes[p_id][g_id] += 1

    unique_pred_ids = set(pred_df['id'].unique())
    
    pred_to_gt_map = {}
    for p_id, gt_votes in votes.items():
        if gt_votes:
            # Get the GT ID with the maximum votes
            best_gt_id = max(gt_votes, key=gt_votes.get)
            pred_to_gt_map[p_id] = best_gt_id
            
    # False Positive IDs are any predicted ID that never successfully map to a GT ID
    mapped_ids = set(pred_to_gt_map.keys())
    false_positive_ids = unique_pred_ids - mapped_ids
    
    return len(false_positive_ids)

if __name__ == "__main__":
    results_dir = "ReID_V2_Test_Results"
    gt_dir = "../mots"
    
    default_csvs = glob(os.path.join(results_dir, "*_default_botsort_tracks.csv"))
    
    results = []
    
    for default_path in default_csvs:
        base_name = os.path.basename(default_path).replace("_default_botsort_tracks.csv", "")
        raw_path = os.path.join(results_dir, f"{base_name}_raw_tracks.csv")
        gt_path = os.path.join(gt_dir, f"{base_name}_clean.txt")
        
        if not os.path.exists(raw_path) or not os.path.exists(gt_path):
            continue
            
        try:
            gt_df = load_ground_truth(gt_path)
            
            # Default Tracker False Positives (Unique IDs)
            df_def = load_predictions(default_path)
            fp_def_ids = count_false_positive_ids(gt_df, df_def)
            
            # Custom Tracker False Positives (Unique IDs)
            df_raw = load_predictions(raw_path)
            fp_raw_ids = count_false_positive_ids(gt_df, df_raw)
            
        except Exception as e:
            print(f"Error processing {base_name}: {e}")
            continue

        results.append({
            'Video': base_name,
            'Default_FP_IDs': fp_def_ids,
            'Custom_FP_IDs': fp_raw_ids,
            'Difference': fp_def_ids - fp_raw_ids, # Positive means Custom reduced false positives
            'Improved': fp_def_ids > fp_raw_ids
        })

    df = pd.DataFrame(results)
    if not df.empty:
        total_def_fp_ids = df['Default_FP_IDs'].sum()
        total_cust_fp_ids = df['Custom_FP_IDs'].sum()
        
        print("--- FALSE POSITIVE (UNIQUE ID) ANALYSIS ---")
        print(f"Evaluated {len(df)} videos against Ground Truth.")
        print(f"Total Ghost Track IDs (Default BoT-SORT): {total_def_fp_ids:,}")
        print(f"Total Ghost Track IDs (Custom BoT-SORT):  {total_cust_fp_ids:,}")
        
        diff = total_cust_fp_ids - total_def_fp_ids
        
        if diff < 0:
            reduction = abs(diff)
            pct = (reduction / total_def_fp_ids * 100) if total_def_fp_ids > 0 else 0
            print(f"\n✅ Result: Custom BoT-SORT REDUCED false positive IDs!")
            print(f"It prevented exactly {reduction:,} ghost track IDs from being created ({pct:.2f}% improvement).")
        elif diff > 0:
            print(f"\n❌ Result: Custom BoT-SORT INCREASED ghost track IDs.")
            print(f"It generated {diff:,} MORE false positive IDs.")
        else:
            print(f"\nResult: No change in False Positive Track creation between configurations.")
            
        df.to_csv(os.path.join(results_dir, "False_Positive_ID_Stats.csv"), index=False)
        print(f"\nSaved detailed metrics to: {os.path.join(results_dir, 'False_Positive_ID_Stats.csv')}")
    else:
        print("No valid paired tracks/ground truth found.")
