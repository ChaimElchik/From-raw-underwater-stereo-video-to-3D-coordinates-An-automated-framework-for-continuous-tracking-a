import os
import glob
import pandas as pd
from Evaluate_Stereo_Algorithm import load_calibration_mapping, evaluate_video, XLSX_PATH, GROUND_TRUTH_DIR

def run_optimization():
    print("=== Epipolar Threshold Optimization ===")
    
    mat_mapping = load_calibration_mapping(XLSX_PATH)
    if not mat_mapping:
        print("Fatal: Cannot proceed without configurations.")
        return
        
    json_files = glob.glob(os.path.join(GROUND_TRUTH_DIR, "*_gt_matches.json"))
    
    thresholds = [0.01, 0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 2.0, 5.0, 1e9]
    
    best_f1 = -1
    best_thresh = None
    
    summary_results = []
    
    for thresh in thresholds:
        print(f"\nEvaluating Threshold: {thresh}...")
        results = []
        for json_path in json_files:
            res = evaluate_video(json_path, mat_mapping, correct_refraction=False, max_epipolar_dist=thresh)
            if res:
                results.append(res)
                
        df = pd.DataFrame(results)
        macro_precision = df["Precision"].mean()
        macro_recall = df["Recall"].mean()
        macro_f1 = df["F1_Score"].mean()
        total_tp = df["TP"].sum()
        total_fp = df["FP"].sum()
        total_fn = df["FN"].sum()
        unmatched = df["Unmatched"].sum()
        incorrect = df["Incorrect"].sum()
        
        print(f"  Macro P: {macro_precision:.2%} | Macro R: {macro_recall:.2%} | Macro F1: {macro_f1:.2%}")
        print(f"  TP: {total_tp} | FP: {total_fp} | FN: {total_fn} (Unmatched: {unmatched}, Incorrect: {incorrect})")
        
        summary_results.append({
            "Threshold": thresh,
            "Macro_Precision": macro_precision,
            "Macro_Recall": macro_recall,
            "Macro_F1": macro_f1,
            "TP": total_tp,
            "FP": total_fp,
            "FN": total_fn,
            "Unmatched": unmatched,
            "Incorrect": incorrect
        })
        
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_thresh = thresh
            
    print(f"\n=========================================")
    print(f"Optimal Threshold Found: {best_thresh}")
    print(f"Peak Macro F1-Score: {best_f1:.2%}")
    print(f"=========================================")

    pd.DataFrame(summary_results).to_csv("threshold_optimization_results.csv", index=False)
    print("Optimization summary saved to threshold_optimization_results.csv")

if __name__ == "__main__":
    run_optimization()
