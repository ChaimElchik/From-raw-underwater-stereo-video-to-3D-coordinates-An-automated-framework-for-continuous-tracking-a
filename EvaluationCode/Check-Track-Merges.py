import pandas as pd
import numpy as np
import argparse
import os
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
    # Ensure standard naming: x, y are top-left, x_offset/y_offset are width/height
    df.rename(columns={'x': 'xmin', 'y': 'ymin', 'x_offset': 'w', 'y_offset': 'h'}, inplace=True)
    return df

def load_predictions(filepath, frame_offset=1):
    """Loads RAW/Pred file: frame,id,x,y,w,h,xmin,ymin,xmax,ymax"""
    df = pd.read_csv(filepath)
    # Align 0-indexed predictions to 1-indexed GT
    df['frame'] = df['frame'] + frame_offset 
    return df

def map_tracks_to_gt(gt_df, pred_df, iou_threshold=0.5):
    """Maps predicted track IDs to Ground Truth IDs via frame-by-frame majority vote."""
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
        
        # Build IoU cost matrix
        iou_matrix = np.zeros((len(pred_ids), len(gt_ids)))
        for i, p_box in enumerate(pred_boxes):
            for j, gt_box in enumerate(gt_boxes):
                iou_matrix[i, j] = calculate_iou(p_box, gt_box)
                
        # Hungarian matching to ensure 1-to-1 mapping in the frame
        cost_matrix = 1.0 - iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= iou_threshold:
                p_id = pred_ids[r]
                g_id = gt_ids[c]
                votes[p_id][g_id] += 1

    # Resolve majority votes
    pred_to_gt_map = {}
    for p_id, gt_votes in votes.items():
        if gt_votes:
            # Get the GT ID with the maximum votes
            best_gt_id = max(gt_votes, key=gt_votes.get)
            pred_to_gt_map[p_id] = best_gt_id
            
    return pred_to_gt_map

def derive_before_to_after_map(before_df, after_df, iou_threshold=0.5):
    """Maps Before-ReID IDs to After-ReID IDs based on spatial overlap (majority vote)."""
    mapping = {} # before_id -> after_id
    votes = defaultdict(lambda: defaultdict(int))
    
    # Process frame by frame where both exist
    frames = before_df['frame'].unique()
    
    for f in frames:
        b_frame = before_df[before_df['frame'] == f]
        a_frame = after_df[after_df['frame'] == f]
        
        if b_frame.empty or a_frame.empty:
            continue
            
        b_boxes = b_frame[['xmin', 'ymin', 'w', 'h']].values
        b_ids = b_frame['id'].values
        
        a_boxes = a_frame[['xmin', 'ymin', 'w', 'h']].values
        a_ids = a_frame['id'].values
        
        # Build IoU cost matrix
        iou_matrix = np.zeros((len(b_ids), len(a_ids)))
        for i, bb in enumerate(b_boxes):
            for j, ab in enumerate(a_boxes):
                iou_matrix[i, j] = calculate_iou(bb, ab)
                
        # Hungarian matching
        cost_matrix = 1.0 - iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= iou_threshold:
                before_id = b_ids[r]
                after_id = a_ids[c]
                votes[before_id][after_id] += 1
                
    # Resolve majority votes
    for b_id, a_votes in votes.items():
        if a_votes:
            # Get the After ID with the maximum votes
            best_a_id = max(a_votes, key=a_votes.get)
            mapping[b_id] = best_a_id
            
    return mapping
            
    return mapping

# def evaluate_reid_merges(before_to_gt_map, before_to_after_map):
#     """Analyzes the merges performed by the Re-ID module."""
    
#     # Reverse the map to see which before_ids make up a single after_id
#     after_to_befores = defaultdict(list)
#     for b_id, a_id in before_to_after_map.items():
#         after_to_befores[a_id].append(b_id)
        
#     correct_merges = 0
#     incorrect_merges = 0
    
#     print("--- Re-ID Merge Analysis ---")
#     for a_id, b_ids in after_to_befores.items():
#         if len(b_ids) > 1: # A merge happened!
#             # Look up the true GT identities of the fragments that were merged
#             mapped_gts = [before_to_gt_map.get(b) for b in b_ids]
            
#             # Remove unmapped (None) fragments, these were false positives
#             valid_gts = [gt for gt in mapped_gts if gt is not None]
            
#             if not valid_gts:
#                 continue
                
#             unique_gts = set(valid_gts)
            
#             if len(unique_gts) == 1:
#                 print(f"✅ Correct Merge: After-ID {a_id} successfully merged Before-IDs {b_ids} (All belong to GT {list(unique_gts)[0]})")
#                 correct_merges += 1
#             else:
#                 print(f"❌ Incorrect Merge: After-ID {a_id} merged Before-IDs {b_ids} which belong to different GTs: {valid_gts}")
#                 incorrect_merges += 1

#     print("\n--- Summary ---")
#     print(f"Total Merges Executed: {correct_merges + incorrect_merges}")
#     print(f"Correct Merges (Fixed Fragmentation): {correct_merges}")
#     print(f"Incorrect Merges (Identity Switches): {incorrect_merges}")

#     # Check for MISSED Merges
#     # Group before_ids by their GT to see what SHOULD have been merged
#     gt_to_befores = defaultdict(list)
#     for b_id, gt_id in before_to_gt_map.items():
#         gt_to_befores[gt_id].append(b_id)
        
#     missed_merges = 0
#     for gt_id, b_ids in gt_to_befores.items():
#         if len(b_ids) > 1:
#             # Check if they ended up with the same after_id
#             after_ids = set([before_to_after_map.get(b) for b in b_ids])
#             if len(after_ids) > 1:
#                 missed_merges += 1
                
#     print(f"Missed Merges (Unfixed Fragmentations): {missed_merges}")
def evaluate_reid_merges(before_to_gt_map, before_to_after_map):
    """Analyzes the merges performed by the Re-ID module."""
    
    # Reverse the map to see which before_ids make up a single after_id
    after_to_befores = defaultdict(list)
    for b_id, a_id in before_to_after_map.items():
        after_to_befores[a_id].append(b_id)
        
    correct_merges = 0
    incorrect_merges = 0
    false_positive_merges = 0
    
    print("--- Re-ID Merge Analysis ---")
    for a_id, b_ids in after_to_befores.items():
        if len(b_ids) > 1: # A merge happened!
            # Look up the true GT identities of the fragments that were merged
            mapped_gts = [before_to_gt_map.get(b) for b in b_ids]
            
            # Remove unmapped (None) fragments, these were false positives
            valid_gts = [gt for gt in mapped_gts if gt is not None]
            
            if not valid_gts:
                # All fragments in this merge are False Positives
                print(f"👻 False Positive Merge: After-ID {a_id} merged Before-IDs {b_ids} (None mapped to Ground Truth)")
                false_positive_merges += 1
                continue
                
            unique_gts = set(valid_gts)
            
            if len(unique_gts) == 1:
                print(f"✅ Correct Merge: After-ID {a_id} successfully merged Before-IDs {b_ids} (All belong to GT {list(unique_gts)[0]})")
                correct_merges += 1
            else:
                print(f"❌ Incorrect Merge: After-ID {a_id} merged Before-IDs {b_ids} which belong to different GTs: {valid_gts}")
                incorrect_merges += 1

# Check for MISSED Merges
    # Group before_ids by their GT to see what SHOULD have been merged
    gt_to_befores = defaultdict(list)
    for b_id, gt_id in before_to_gt_map.items():
        if gt_id is not None: # Ensure we don't group false positives
            gt_to_befores[gt_id].append(b_id)
        
    missed_merges = 0
    print("\n--- Missed Merges Analysis ---")
    for gt_id, b_ids in gt_to_befores.items():
        if len(b_ids) > 1:
            # Check if they ended up with the same after_id
            after_ids = [before_to_after_map.get(b) for b in b_ids]
            unique_after_ids = set([a for a in after_ids if a is not None])
            
            if len(unique_after_ids) > 1:
                # Clean up numpy types by casting to standard ints
                clean_gt = int(gt_id)
                clean_b_ids = [int(b) for b in b_ids]
                clean_after_ids = [int(a) for a in unique_after_ids]
                
                print(f"⚠️ Missed Merge (GT Person {clean_gt}):")
        
                print(f"   - Originally fragmented as Track IDs: {clean_b_ids}")
                print(f"   - Left unfixed as Re-ID Tracks:       {clean_after_ids}\n")
                
                missed_merges += 1

    print("\n--- Summary ---")
    print(f"Total Merges Executed: {correct_merges + incorrect_merges + false_positive_merges}")
    print(f"Correct Merges (Fixed Fragmentation): {correct_merges}")
    print(f"Incorrect Merges (Identity Switches): {incorrect_merges}")
    print(f"False Positive Merges (Non-GT Tracks): {false_positive_merges}")
    print(f"Missed Merges (Unfixed Fragmentations): {missed_merges}")

    return {
        "Amount of Merges Missed": missed_merges,
        "Merges RE-ID Correct": correct_merges,
        "Merges RE-ID Correct Merge but FP": false_positive_merges,
        "Incorrect Merges RE-ID": incorrect_merges
    }

def calculate_tracking_metrics(gt_df, pred_df, iou_threshold=0.5, label="Tracker"):
    """Calculates standard tracking metrics (MOTA, IDF1, IDSW, Frag) for a prediction dataframe."""
    
    # Sort dataframes strictly by frame
    gt_df = gt_df.sort_values('frame')
    pred_df = pred_df.sort_values('frame')
    
    # Tracking variables over time
    gt_ids_tracked = set()            # GT IDs that have been tracked at all
    pred_to_gt = {}                   # Current mapping in the active frame
    
    # Accumulators for MOTA
    total_gt = 0
    false_positives = 0
    false_negatives = 0
    id_switches = 0
    fragmentations = 0
    
    # State tracking for IDSW and Frag
    last_gt_mapped_pred = {}          # gt_id -> last pred_id it was mapped to
    gt_last_seen = {}                 # gt_id -> last frame it was seen in GT
    
    # Accumulators for IDF1
    # Bipartite matching totals over the whole video
    # We build a global cost matrix: rows=pred_ids, cols=gt_ids
    # Cost = number of frames they OVERLAPPED minus number of frames ONLY ONE existed
    gt_max = gt_df['id'].max() if not gt_df.empty else 0
    pred_max = pred_df['id'].max() if not pred_df.empty else 0
    
    # Mappings from raw ID to 0-indexed contiguous ints for matrix indexing
    unique_gt_ids = gt_df['id'].unique()
    unique_pred_ids = pred_df['id'].unique()
    
    gt_idx = {val: i for i, val in enumerate(unique_gt_ids)}
    pred_idx = {val: i for i, val in enumerate(unique_pred_ids)}
    
    overlap_matrix = np.zeros((len(unique_pred_ids), len(unique_gt_ids)))
    pred_frame_counts = pred_df['id'].value_counts().to_dict()
    gt_frame_counts = gt_df['id'].value_counts().to_dict()

    frames = sorted(list(set(gt_df['frame'].unique()).union(set(pred_df['frame'].unique()))))
    
    for f in frames:
        g_frame = gt_df[gt_df['frame'] == f]
        p_frame = pred_df[pred_df['frame'] == f]
        
        total_gt += len(g_frame)
        
        # Build distance matrix for THIS frame
        # rows = preds, cols = gts
        if not g_frame.empty and not p_frame.empty:
            g_boxes = g_frame[['xmin', 'ymin', 'w', 'h']].values
            p_boxes = p_frame[['xmin', 'ymin', 'w', 'h']].values
            g_ids = g_frame['id'].values
            p_ids = p_frame['id'].values
            
            iou_matrix = np.zeros((len(p_ids), len(g_ids)))
            for i, p_box in enumerate(p_boxes):
                for j, g_box in enumerate(g_boxes):
                    iou_matrix[i, j] = calculate_iou(p_box, g_box)
                    
            # Match
            cost_matrix = 1.0 - iou_matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Record matches
            matched_p = set()
            matched_g = set()
            
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= iou_threshold:
                    p_id = p_ids[r]
                    g_id = g_ids[c]
                    
                    matched_p.add(p_id)
                    matched_g.add(g_id)
                    
                    # Log overlap for IDF1
                    overlap_matrix[pred_idx[p_id], gt_idx[g_id]] += 1
                    
                    # Check for ID Switch or Fragmentation
                    if g_id in last_gt_mapped_pred:
                        if last_gt_mapped_pred[g_id] != p_id:
                            id_switches += 1
                        elif f > gt_last_seen.get(g_id, 0) + 1:
                            # It was the SAME prediction ID, but there was a gap in frames
                            fragmentations += 1
                    
                    last_gt_mapped_pred[g_id] = p_id
            
            # False Positives = predictions not matched to anything
            false_positives += len(p_ids) - len(matched_p)
            
            # False Negatives = GT objects not matched to anything
            false_negatives += len(g_ids) - len(matched_g)
            
        else:
            # Entire frame is FN or FP
            false_negatives += len(g_frame)
            false_positives += len(p_frame)

        # Update last seen frames
        for gid in g_frame['id'].values:
            gt_last_seen[gid] = f

    # Calculate MOTA
    mota = 1.0 - (false_positives + false_negatives + id_switches) / total_gt if total_gt > 0 else 0.0
    
    # Calculate IDF1 via global bipartite matching
    # Cost to match Pred P to GT G = (Frames P exists) + (Frames G exists) - 2 * (Frames they overlap)
    # This is equivalent to finding the matching that maximizes 2 * Overlap
    id_cost_matrix = np.zeros_like(overlap_matrix)
    for p_id_val, p_i in pred_idx.items():
        for g_id_val, g_i in gt_idx.items():
            p_len = pred_frame_counts.get(p_id_val, 0)
            g_len = gt_frame_counts.get(g_id_val, 0)
            overlap = overlap_matrix[p_i, g_i]
            id_cost_matrix[p_i, g_i] = (p_len + g_len - 2 * overlap)

    row_ind, col_ind = linear_sum_assignment(id_cost_matrix)
    total_overlap = 0
    total_pred_frames_in_matches = 0
    total_gt_frames_in_matches = 0
    
    for r, c in zip(row_ind, col_ind):
        # We only consider it a match if it's strictly beneficial (reduces cost compared to leaving both unassigned)
        # Leaving unassigned cost = p_len + g_len. 
        # Assigned cost = p_len + g_len - 2*overlap
        # So it's always beneficial if overlap > 0.
        if overlap_matrix[r, c] > 0:
            total_overlap += overlap_matrix[r, c]
            p_id = unique_pred_ids[r]
            g_id = unique_gt_ids[c]
            total_pred_frames_in_matches += pred_frame_counts.get(p_id, 0)
            total_gt_frames_in_matches += gt_frame_counts.get(g_id, 0)

    # Global sum of all GT and Pred frames
    sum_gt_frames = sum(gt_frame_counts.values())
    sum_pred_frames = sum(pred_frame_counts.values())

    idp = total_overlap / sum_pred_frames if sum_pred_frames > 0 else 0
    idr = total_overlap / sum_gt_frames if sum_gt_frames > 0 else 0
    idf1 = (2 * idp * idr) / (idp + idr) if (idp + idr) > 0 else 0

    # Calculate MT, PT, ML
    mt = 0
    pt = 0
    ml = 0
    for g_id_val, g_i in gt_idx.items():
        g_len = gt_frame_counts.get(g_id_val, 0)
        if g_len > 0:
            max_overlap = np.max(overlap_matrix[:, g_i]) if overlap_matrix.shape[0] > 0 else 0
            ratio = max_overlap / g_len
            if ratio >= 0.8:
                mt += 1
            elif ratio <= 0.2:
                ml += 1
            else:
                pt += 1

    print(f"\n[{label} Metrics]")
    print(f"  MOTA:  {mota * 100:.1f}%")
    print(f"  IDF1:  {idf1 * 100:.1f}%")
    print(f"  IDP:   {idp * 100:.1f}%")
    print(f"  IDR:   {idr * 100:.1f}%")
    print(f"  MT:    {mt}")
    print(f"  PT:    {pt}")
    print(f"  ML:    {ml}")
    print(f"  IDSW:  {id_switches}")
    print(f"  Frag:  {fragmentations}")
    print(f"  FP:    {false_positives}")
    print(f"  FN:    {false_negatives}")

    return {
        "MOTA": mota,
        "IDF1": idf1,
        "IDP": idp,
        "IDR": idr,
        "MT": mt,
        "PT": pt,
        "ML": ml,
        "IDSW": id_switches,
        "Number of IDs": len(unique_pred_ids)
    }

# ==========================================
# How to run the pipeline
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Re-ID and tracking merges against ground truth.")
    parser.add_argument("--video", required=True, help="Base name of the video (e.g., '129_1')")
    parser.add_argument("--gt_dir", default="../mots", help="Directory containing the ground truth files")
    parser.add_argument("--results_dir", default="ReID_Test_Results", help="Directory containing the tracking CSV results")
    args = parser.parse_args()

    # 1. Load Data
    gt_path = os.path.join(args.gt_dir, f"{args.video}_clean.txt")
    raw_csv = os.path.join(args.results_dir, f"{args.video}_raw_tracks.csv")
    reid_csv = os.path.join(args.results_dir, f"{args.video}_reid_tracks.csv")
    default_botsort_csv = os.path.join(args.results_dir, f"{args.video}_default_botsort_tracks.csv")

    print(f"Loading Ground Truth: {gt_path}")
    gt_df = load_ground_truth(gt_path)
    
    print(f"Loading Custom Baseline: {raw_csv}")
    before_df = load_predictions(raw_csv)
    
    # Load After-ReID CSV
    print(f"Loading Custom After-ReID: {reid_csv}")
    after_df = load_predictions(reid_csv) 

    # Load Default BoT-SORT CSV
    print(f"Loading Default BoT-SORT: {default_botsort_csv}")
    default_botsort_df = load_predictions(default_botsort_csv)

    print("\n--- Unique ID Counts ---")
    print(f"Ground Truth IDs:      {gt_df['id'].nunique()}")
    print(f"Default BoT-SORT IDs:  {default_botsort_df['id'].nunique()}")
    print(f"Custom Baseline IDs:   {before_df['id'].nunique()}")
    print(f"Custom After-ReID IDs: {after_df['id'].nunique()}\n")
    
    print("\n=======================================================")
    print("                TRACKING METRIC EVALUATION             ")
    print("=======================================================")
    default_metrics = calculate_tracking_metrics(gt_df, default_botsort_df, label="Default BoT-SORT")
    baseline_metrics = calculate_tracking_metrics(gt_df, before_df, label="Custom Baseline")
    reid_metrics = calculate_tracking_metrics(gt_df, after_df, label="Custom After-ReID")
    print("=======================================================\n")
    
    print("Mapping Before-ReID tracks to Ground Truth...")
    before_to_gt_map = map_tracks_to_gt(gt_df, before_df)
    
    print("Mapping Before-ReID tracks to After-ReID tracks...")
    before_to_after_map = derive_before_to_after_map(before_df, after_df)
    
    print("\nEvaluating Re-ID specific mechanics...")
    merge_stats = evaluate_reid_merges(before_to_gt_map, before_to_after_map)

    # Export to CSV
    csv_out_path = os.path.join(args.results_dir, f"{args.video}_metrics.csv")
    
    rows = []
    for method, metrics in [("Default BoT-SORT", default_metrics), 
                            ("Custom Baseline", baseline_metrics), 
                            ("Custom After-ReID", reid_metrics)]:
        
        row = {
            "Video ID": args.video,
            "Method": method,
            "MOTA": metrics["MOTA"],
            "IDF1": metrics["IDF1"],
            "IDP": metrics["IDP"],
            "IDR": metrics["IDR"],
            "IDSW": metrics["IDSW"],
            "PT": metrics["PT"],
            "ML": metrics["ML"],
            "MT": metrics["MT"],
            "Number of IDs": metrics["Number of IDs"],
            "Amount of Merges Missed": merge_stats["Amount of Merges Missed"],
            "Merges RE-ID Correct": merge_stats["Merges RE-ID Correct"],
            "Merges RE-ID Correct Merge but FP": merge_stats["Merges RE-ID Correct Merge but FP"],
            "Incorrect Merges RE-ID": merge_stats["Incorrect Merges RE-ID"]
        }
        rows.append(row)
        
    df_out = pd.DataFrame(rows)
    df_out.to_csv(csv_out_path, index=False)
    print(f"\n✅ Successfully saved tracking metrics to {csv_out_path}")