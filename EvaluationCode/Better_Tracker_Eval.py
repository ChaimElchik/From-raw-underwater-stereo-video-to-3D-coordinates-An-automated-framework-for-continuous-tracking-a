import pandas as pd
import numpy as np
import os
import glob
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import argparse

# --- Constants ---
IOU_THRESHOLD = 0.3

def calculate_iou(boxA, boxB):
    # box: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def evaluate_sequence(gt_path, tracker_path):
    # 1. Load Data
    try:
        gt_df = pd.read_csv(gt_path)
        # GT format: frame, id, x, y, x_offset, y_offset
        # Convert to: frame, id, x1, y1, x2, y2
        gt_df['x1'] = gt_df['x']
        gt_df['y1'] = gt_df['y']
        gt_df['x2'] = gt_df['x'] + gt_df['x_offset']
        gt_df['y2'] = gt_df['y'] + gt_df['y_offset']
    except Exception as e:
        print(f"Error loading GT {gt_path}: {e}")
        return None

    try:
        tr_df = pd.read_csv(tracker_path)
        # Tracker format from ProcessVideoPair: frame, id, x, y, w, h
        # Convert to x1, y1, x2, y2
        # Note: Tracker frame is 0-indexed, GT is 1-indexed. Adjust tracker.
        tr_df['frame'] = tr_df['frame'] + 1
        
        tr_df['x1'] = tr_df['x'] - tr_df['w']/2
        tr_df['y1'] = tr_df['y'] - tr_df['h']/2
        tr_df['x2'] = tr_df['x'] + tr_df['w']/2
        tr_df['y2'] = tr_df['y'] + tr_df['h']/2
    except Exception as e:
        print(f"Error loading Tracker {tracker_path}: {e}")
        return None

    # 2. Frame-by-Frame Association
    all_frames = sorted(list(set(gt_df['frame'].unique()) | set(tr_df['frame'].unique())))
    
    # Storage for metrics
    # Maps for IDSW: GT_ID -> List of assigned Tracker IDs
    gt_to_track_map = defaultdict(list)
    
    # Maps for Bad Merge (Purity): Track_ID -> List of assigned GT IDs (weighted by occurrence)
    track_to_gt_map = defaultdict(list)
    
    # IDF1 Counters
    matches_count = 0
    gt_count = len(gt_df)
    tr_count = len(tr_df)

    for frame in all_frames:
        gts = gt_df[gt_df['frame'] == frame]
        trs = tr_df[tr_df['frame'] == frame]
        
        if len(gts) == 0 or len(trs) == 0:
            continue
            
        gt_boxes = gts[['x1', 'y1', 'x2', 'y2']].values
        gt_ids = gts['id'].values
        tr_boxes = trs[['x1', 'y1', 'x2', 'y2']].values
        tr_ids = trs['id'].values
        
        # IoU Matrix
        iou_matrix = np.zeros((len(gt_boxes), len(tr_boxes)))
        for i, gb in enumerate(gt_boxes):
            for j, tb in enumerate(tr_boxes):
                iou_matrix[i, j] = calculate_iou(gb, tb)
                
        # Hungarian
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= IOU_THRESHOLD:
                gid = gt_ids[r]
                tid = tr_ids[c]
                
                gt_to_track_map[gid].append(tid)
                track_to_gt_map[tid].append(gid)
                matches_count += 1

    # 3. Compute Metrics
    
    # --- ID Switches (IDSW) ---
    total_idsw = 0
    for gid, tracks in gt_to_track_map.items():
        if len(tracks) < 2: continue
        curr = tracks[0]
        for t in tracks[1:]:
            if t != curr:
                total_idsw += 1
                curr = t

    # --- Tracker Purity (Bad Merge Indicator) ---
    # For each tracker ID, what % of its frames belonged to the Dominant GT ID?
    purity_scores = []
    bad_merge_tracks = 0
    
    for tid, gids in track_to_gt_map.items():
        if not gids: continue
        total_len = len(gids)
        counts = pd.Series(gids).value_counts()
        dominant_count = counts.iloc[0]
        diversity = len(counts)
        
        purity = dominant_count / total_len
        purity_scores.append(purity)
        
        # A "Bad Merge" is a track that significantly overlaps with multiple GTs
        # Criteria: Purity < 0.8 AND covers at least 2 GTs
        if purity < 0.8 and diversity > 1:
            bad_merge_tracks += 1

    avg_purity = np.mean(purity_scores) if purity_scores else 0.0

    return {
        "IDSW": total_idsw,
        "Avg_Purity": avg_purity,
        "Bad_Merge_Tracks": bad_merge_tracks,
        "Matches": matches_count,
        "GT_Objects": gt_count,
        "Tracker_Objects": tr_count
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True, help="Root of eval_results (e.g. eval_results/mode_1...)")
    parser.add_argument("--gt_dir", required=True, help="Path to mots/ folder with _clean.txt files")
    args = parser.parse_args()
    
    results_dir = args.results_dir
    gt_dir = args.gt_dir
    
    # Aggregate Metrics
    total_idsw = 0
    total_purity = []
    total_bad_merges = 0
    total_matches = 0
    total_gt = 0
    total_tr = 0
    
    # Walk through results
    # Expected structure: {results_dir}/{vid_id}/{cam1_tracked.csv, cam2_tracked.csv}
    
    video_dirs = glob.glob(os.path.join(results_dir, "*"))
    
    processed_count = 0
    
    print(f"Evaluating results in: {results_dir}")
    print(f"{'Video':<10} {'Cam':<5} {'IDSW':<5} {'Purity':<8} {'BadMerges':<10}")
    print("-" * 50)
    
    for vdir in video_dirs:
        if not os.path.isdir(vdir): continue
        vid_id = os.path.basename(vdir)
        
        for cam in ['1', '2']:
            track_file = os.path.join(vdir, f"cam{cam}_tracked.csv")
            gt_file = os.path.join(gt_dir, f"{vid_id}_{cam}_clean.txt")
            
            if os.path.exists(track_file) and os.path.exists(gt_file):
                res = evaluate_sequence(gt_file, track_file)
                if res:
                    print(f"{vid_id:<10} {cam:<5} {res['IDSW']:<5} {res['Avg_Purity']:.4f}   {res['Bad_Merge_Tracks']:<10}")
                    
                    total_idsw += res['IDSW']
                    total_purity.append(res['Avg_Purity'])
                    total_bad_merges += res['Bad_Merge_Tracks']
                    total_matches += res['Matches']
                    total_gt += res['GT_Objects']
                    total_tr += res['Tracker_Objects']
                    processed_count += 1
            else:
                 # Debug missing files
                 pass

    #IDF1 Calculation (Approximate Global)
    rec = total_matches / total_gt if total_gt > 0 else 0
    prec = total_matches / total_tr if total_tr > 0 else 0
    idf1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    
    avg_purity_global = np.mean(total_purity) if total_purity else 0
    
    print("=" * 50)
    print(f"Generating Summary for {os.path.basename(results_dir)}")
    print(f"Total Videos Processed: {processed_count/2} (approx pairs)")
    print(f"Global IDF1: {idf1:.4f}")
    print(f"Total ID Switches: {total_idsw}")
    print(f"Average Tracker Purity: {avg_purity_global:.4f}")
    print(f"Total Bad Merge Tracks: {total_bad_merges}")
    
    # Save to simplistic report
    report_path = os.path.join(results_dir, "summary_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Global IDF1: {idf1:.4f}\n")
        f.write(f"Total ID Switches: {total_idsw}\n")
        f.write(f"Average Tracker Purity: {avg_purity_global:.4f}\n")
        f.write(f"Total Bad Merge Tracks: {total_bad_merges}\n")
    print(f"Summary saved to {report_path}")

if __name__ == "__main__":
    main()
