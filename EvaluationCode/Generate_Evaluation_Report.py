import os
import json
import pandas as pd
from StereoMatching import run_geometric_matching

CODE_DIR = "/Users/chaim/Documents/Thesis_ReWrite/Final_Pipeline_Code"
csv_path = os.path.join(CODE_DIR, "evaluation_results.csv")
xlsx_path = "/Users/chaim/Documents/Thesis_ReWrite/usable_project2_12Oct2023.xlsx - Sheet1.csv"
out_txt_path = os.path.join(CODE_DIR, "detailed_evaluation_report.txt")

print("Generating detailed evaluation report...")

# Load excel mapping
df_excel = pd.read_csv(xlsx_path)
excel_mapping = {}
for _, row in df_excel.iterrows():
    video_id = str(int(row["Video_ID"]))
    dep_id = int(row["Dep_ID"])
    excel_mapping[video_id] = f"stereoParams_Dep{dep_id}.mat"

# Load CSV
df_csv = pd.read_csv(csv_path)

output_lines = []
output_lines.append("=== Stereo Matching Evaluation Report ===")
output_lines.append("")

output_lines.append(f"Total Videos Analyzed: {len(df_csv)}")
output_lines.append(f"Macro Average Precision: {df_csv['Precision'].mean():.2%}")
output_lines.append(f"Macro Average Recall:    {df_csv['Recall'].mean():.2%}")
output_lines.append(f"Macro Average F1-Score:  {df_csv['F1_Score'].mean():.2%}")
output_lines.append("")

output_lines.append("--- Metrics Per Video Pair ---")
for _, row in df_csv.iterrows():
    video_id = str(int(row["video"]))
    expected_mat = excel_mapping.get(video_id, "Unknown")
    actual_mat = row["calibration_mat"]
    match_status = "MATCH" if expected_mat == actual_mat else "MISMATCH"
    
    tp = int(row['TP'])
    fp = int(row['FP'])
    fn = int(row['FN'])
    unmatched = int(row.get('Unmatched', 0))
    incorrect = int(row.get('Incorrect', 0))
    total_expected = int(row.get('total_gt_pairs', tp + fn))
    
    output_lines.append(f"Video {video_id}:")
    output_lines.append(f"  Expected Params (from Excel): {expected_mat}")
    output_lines.append(f"  Used Params:                 {actual_mat} [{match_status}]")
    output_lines.append(f"  Precision: {row['Precision']:.2%} | Recall: {row['Recall']:.2%} | F1: {row['F1_Score']:.2%}")
    output_lines.append(f"  TP: {tp}, FP: {fp}, FN: {fn}")
    output_lines.append(f"  -> Total Expected Matches (GT Pairs): {total_expected}")
    output_lines.append(f"  -> Unmatched IDs (No match pred):     {unmatched}")
    output_lines.append(f"  -> Incorrect IDs (Wrong match pred):  {incorrect}")
    output_lines.append("")

# Find lowest scoring video (by F1 score)
lowest_video_row = df_csv.loc[df_csv['F1_Score'].idxmin()]
lowest_video_id = str(int(lowest_video_row["video"]))
lowest_mat = lowest_video_row["calibration_mat"]

output_lines.append(f"--- Lowest Scoring Video: {lowest_video_id} ---")
output_lines.append("Analysis of predictions vs ground truth...")

# Run matcher for the lowest scoring video
mot1_path = os.path.join(CODE_DIR, "Ground_Truth_Matches", f"{lowest_video_id}_1.txt")
mot2_path = os.path.join(CODE_DIR, "Ground_Truth_Matches", f"{lowest_video_id}_2.txt")
json_path = os.path.join(CODE_DIR, "Ground_Truth_Matches", f"{lowest_video_id}_gt_matches.json")
mat_path = os.path.join(CODE_DIR, "stereo_parameters", lowest_mat)

with open(json_path, "r") as f:
    gt_data = json.load(f)
gt_mapping = {int(k): int(v) for k, v in gt_data["mapping_dict"].items()}

best_mapping_df, _ = run_geometric_matching(
    file1_path=mot1_path,
    file2_path=mot2_path,
    mat_path=mat_path,
    correct_refraction=False
)

pred_mapping = {}
if not best_mapping_df.empty:
    for _, row in best_mapping_df.iterrows():
        pred_mapping[int(row['id1'])] = int(row['id2'])

output_lines.append("")
output_lines.append(f"Predicted Matches vs Ground Truth for Video {lowest_video_id}:")
all_id1s = sorted(list(set(gt_mapping.keys()) | set(pred_mapping.keys())))

for id1 in all_id1s:
    pred_val = pred_mapping.get(id1, "None")
    gt_val = gt_mapping.get(id1, "None")
    
    if pred_val == gt_val:
        status = "CORRECT"
    elif pred_val == "None":
        status = "FALSE NEGATIVE (Missed by model)"
    elif gt_val == "None":
        status = "FALSE POSITIVE (Model predicted match not in GT)"
    else:
        status = "MISMATCH (Model chose wrong ID)"
        
    output_lines.append(f"  Cam1 ID {id1:02d} -> Model Pred: {str(pred_val).ljust(4)} | True GT: {str(gt_val).ljust(4)} | [{status}]")

with open(out_txt_path, "w") as f:
    f.write("\n".join(output_lines) + "\n")
print(f"Report correctly generated to {out_txt_path}")
