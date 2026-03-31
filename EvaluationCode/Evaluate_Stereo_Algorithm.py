import os
import glob
import json
import pandas as pd
from StereoMatching import run_geometric_matching

# --- Configuration Paths ---
BASE_DIR = "/Users/chaim/Documents/Thesis_ReWrite"
CODE_DIR = os.path.join(BASE_DIR, "Final_Pipeline_Code")
GROUND_TRUTH_DIR = os.path.join(CODE_DIR, "Ground_Truth_Matches")
STEREO_PARAMS_DIR = os.path.join(CODE_DIR, "stereo_parameters")
XLSX_PATH = os.path.join(BASE_DIR, "usable_project2_12Oct2023.xlsx - Sheet1.csv")
OUTPUT_CSV = os.path.join(CODE_DIR, "evaluation_results.csv")


def load_calibration_mapping(csv_path):
    """
    Reads the configuration sheet and returns a dictionary 
    mapping string Video_ID to integer Dep_ID.
    """
    print(f"Loading Calibration mapping from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
        mapping = {}
        for _, row in df.iterrows():
            video_id = str(int(row["Video_ID"]))
            dep_id = int(row["Dep_ID"])
            mapping[video_id] = dep_id
        return mapping
    except Exception as e:
        print(f"Failed to load mapping: {e}")
        return {}


def get_mat_path_for_video(video_id, mapping_dict):
    """
    Resolves the specific .mat file for the camera configuration used in the video.
    """
    if video_id not in mapping_dict:
        return None
    
    dep_event = mapping_dict[video_id]
    mat_filename = f"stereoParams_Dep{dep_event}.mat"
    mat_path = os.path.join(STEREO_PARAMS_DIR, mat_filename)
    
    if not os.path.exists(mat_path):
        return None
        
    return mat_path


def evaluate_video(json_path, mat_mapping, correct_refraction, max_epipolar_dist=0.12):
    """
    Evaluates the stereo matching algorithm on a specific video using its Ground Truth JSON.
    """
    # 1. Load Ground Truth Data
    with open(json_path, "r") as f:
        gt_data = json.load(f)
        
    video_id = str(gt_data["video"])
    
    # Intentionally casting JSON strings map back to Integer -> Integer tracking for comparison safely
    gt_mapping = {int(k): int(v) for k, v in gt_data["mapping_dict"].items()}
    
    if len(gt_mapping) == 0:
        return None
        
    # 2. Map Calibration .mat File
    mat_path = get_mat_path_for_video(video_id, mat_mapping)
    if not mat_path:
        print(f"[{video_id}] Error: Could not find valid matching stereo parameter .mat file!")
        return None
        
    # 3. Locate Source MOTs
    # Note: Create_GT_Matches.py copies isolating clean MOT txts into Ground_Truth_Matches
    mot1_path = os.path.join(GROUND_TRUTH_DIR, f"{video_id}_1.txt")
    mot2_path = os.path.join(GROUND_TRUTH_DIR, f"{video_id}_2.txt")
    
    if not os.path.exists(mot1_path) or not os.path.exists(mot2_path):
        print(f"[{video_id}] Error: Missing isolated Ground Truth MOT tracking data files!")
        return None
        
    # 4. Run Stereo Matching Algorithm
    try:
        best_mapping_df, _ = run_geometric_matching(
            file1_path=mot1_path,
            file2_path=mot2_path,
            mat_path=mat_path,
            correct_refraction=correct_refraction,
            max_epipolar_dist=max_epipolar_dist
        )
    except Exception as e:
        print(f"[{video_id}] Error during geometric matching: {e}")
        return None
        
    # Extract prediction map (id1 -> id2)
    # The dataframe output contains 'id1' (from cam 1) and 'id2' (from cam 2)
    pred_mapping = {}
    if not best_mapping_df.empty:
        for _, row in best_mapping_df.iterrows():
            pred_mapping[int(row['id1'])] = int(row['id2'])
            
    # 5. Calculate Metrics (Precision, Recall)
    # TP (True Positive) = Model predicts pair (A,B), and GT contains pair (A,B).
    # FP (False Positive) = Model predicts pair (A,B), but GT does NOT contain pair (A,B).
    # FN (False Negative) = GT contains pair (A,B), but Model does NOT predict (A,B).
    
    tp, fp, fn = 0, 0, 0
    incorrect = 0
    unmatched = 0
    
    # Calculate True Positives and False Positives
    for p_id1, p_id2 in pred_mapping.items():
        if p_id1 in gt_mapping and gt_mapping[p_id1] == p_id2:
            tp += 1
        else:
            fp += 1
            
    # Calculate False Negatives (unmatched vs incorrect)
    for gt_id1, gt_id2 in gt_mapping.items():
        if gt_id1 not in pred_mapping:
            unmatched += 1
            fn += 1
        elif pred_mapping[gt_id1] != gt_id2:
            incorrect += 1
            fn += 1
            
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "video": video_id,
        "calibration_mat": os.path.basename(mat_path),
        "total_gt_pairs": len(gt_mapping),
        "total_pred_pairs": len(pred_mapping),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Unmatched": unmatched,
        "Incorrect": incorrect,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1
    }

def main():
    print("=== Stereo Matching Evaluation Pipeline ===")
    
    # 1. Load Calibration Layout
    mat_mapping = load_calibration_mapping(XLSX_PATH)
    if not mat_mapping:
        print("Fatal: Cannot proceed without usable_project2_12Oct2023.xlsx configurations.")
        return
        
    # 2. Iterate GT Matches
    json_files = glob.glob(os.path.join(GROUND_TRUTH_DIR, "*_gt_matches.json"))
    
    if not json_files:
        print(f"No Ground Truth JSONs found in {GROUND_TRUTH_DIR}.")
        return
        
    print(f"Found {len(json_files)} GT datasets to evaluate natively.")
    
    results = []
    
    for json_path in json_files:
        # As requested, bypassing refraction calibration via correct_refraction=False
        res = evaluate_video(json_path, mat_mapping, correct_refraction=False)
        if res:
            results.append(res)
            
    if not results:
        print("No evaluations succeeded.")
        return
        
    # 3. Compute Macro-Average Metrics
    df = pd.DataFrame(results)
    
    macro_precision = df["Precision"].mean()
    macro_recall = df["Recall"].mean()
    macro_f1 = df["F1_Score"].mean()
    total_tp = df["TP"].sum()
    total_fp = df["FP"].sum()
    total_fn = df["FN"].sum()
    
    print("\n-------------------------------------------")
    print("      GLOBAL EVALUATION RESULTS")
    print("-------------------------------------------")
    print(f"Total Videos Analyzed: {len(df)}")
    print(f"Total True Positives:  {total_tp}")
    print(f"Total False Positives: {total_fp}")
    print(f"Total False Negatives: {total_fn}")
    print(f"-------------------------------------------")
    print(f"Macro Average Precision: {macro_precision:.2%}")
    print(f"Macro Average Recall:    {macro_recall:.2%}")
    print(f"Macro Average F1-Score:  {macro_f1:.2%}")
    print(f"-------------------------------------------")
    
    # 4. Save to CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Detailed per-video breakdown saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
