import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment

def analyze_tracking_quality(gt_path, tracker_path):
    print(f"Loading and Analyzing: {tracker_path}...")
    
    # ---------------------------------------------------------
    # 1. Load and Align Data
    # ---------------------------------------------------------
    try:
        gt_df = pd.read_csv(gt_path)
        tracker_df = pd.read_csv(tracker_path)
    except FileNotFoundError:
        print("Error: File not found.")
        return

    # Align Frames (Tracker 0-based -> GT 1-based)
    tracker_df['frame'] = tracker_df['frame'] + 1
    
    # Standardize Coordinates to x1, y1, x2, y2
    # GT Format: x, y, x_offset (w), y_offset (h)
    gt_df['x1'] = gt_df['x']
    gt_df['y1'] = gt_df['y']
    gt_df['x2'] = gt_df['x'] + gt_df['x_offset']
    gt_df['y2'] = gt_df['y'] + gt_df['y_offset']
    
    # Tracker Format: Usually xmin, ymin, xmax, ymax
    if 'xmin' in tracker_df.columns:
        tracker_df = tracker_df.rename(columns={'xmin': 'x1', 'ymin': 'y1', 'xmax': 'x2', 'ymax': 'y2'})

    # ---------------------------------------------------------
    # 2. Build Mappings (Who Matched Who?)
    # ---------------------------------------------------------
    # Map Tracker_ID -> Set of GT_IDs (For Purity)
    tracker_to_gt_map = {tid: set() for tid in tracker_df['id'].unique()}
    
    # Map GT_ID -> Set of Tracker_IDs (For Fragmentation)
    gt_to_tracker_map = {gid: set() for gid in gt_df['id'].unique()}
    
    all_frames = sorted(gt_df['frame'].unique())
    
    # Helper IoU Function
    def calculate_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    # Frame-by-Frame Matching
    for frame in all_frames:
        gts = gt_df[gt_df['frame'] == frame]
        trs = tracker_df[tracker_df['frame'] == frame]
        
        if gts.empty or trs.empty: continue
            
        gt_boxes = gts[['x1', 'y1', 'x2', 'y2']].values
        tr_boxes = trs[['x1', 'y1', 'x2', 'y2']].values
        gt_ids = gts['id'].values
        tr_ids = trs['id'].values
        
        # IoU Matrix
        iou_matrix = np.zeros((len(gt_boxes), len(tr_boxes)))
        for i, gb in enumerate(gt_boxes):
            for j, tb in enumerate(tr_boxes):
                iou_matrix[i, j] = calculate_iou(gb, tb)
        
        # Hungarian Matching
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        
        for r, c in zip(row_ind, col_ind):
            # If overlap is good, record the link both ways
            if iou_matrix[r, c] >= 0.3:
                t_id = tr_ids[c]
                g_id = gt_ids[r]
                tracker_to_gt_map[t_id].add(g_id)
                gt_to_tracker_map[g_id].add(t_id)

    # ---------------------------------------------------------
    # 3. Report 1: Purity (Tracker Perspective)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("REPORT 1: TRACKER ID PURITY")
    print("="*50)
    print(f"{'Tracker ID':<12} | {'Status':<12} | {'Matched GT IDs'}")
    print("-" * 50)
    
    pure_count = 0
    dirty_count = 0
    ghost_count = 0
    
    for t_id in sorted(tracker_to_gt_map.keys()):
        gt_set = tracker_to_gt_map[t_id]
        gt_list = sorted(list(gt_set))
        
        if len(gt_list) == 0:
            status = "GHOST" # Detected something that wasn't in GT
            ghost_count += 1
        elif len(gt_list) == 1:
            status = "PURE"
            pure_count += 1
        else:
            status = "DIRTY"
            dirty_count += 1
            
        print(f"{t_id:<12} | {status:<12} | {gt_list}")

    print("-" * 50)
    print(f"Total Pure IDs:  {pure_count} (Valid data)")
    print(f"Total Dirty IDs: {dirty_count} (Merged wrong fish - BAD)")
    print(f"Total Ghosts:    {ghost_count} (False Positives)")

    # ---------------------------------------------------------
    # 4. Report 2: Fragmentation (Ground Truth Perspective)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("REPORT 2: FRAGMENTATION (Broken Tracks)")
    print("="*50)
    print(f"{'GT Fish ID':<12} | {'Status':<15} | {'Tracker IDs Used'}")
    print("-" * 55)
    
    total_fragments = 0
    perfect_tracks = 0
    missed_tracks = 0
    
    for gid in sorted(gt_to_tracker_map.keys()):
        tid_set = gt_to_tracker_map[gid]
        tids = sorted(list(tid_set))
        count = len(tids)
        
        # A "Fragment" is an extra track ID beyond the first one.
        fragments = max(0, count - 1)
        total_fragments += fragments
        
        if count == 1:
            perfect_tracks += 1
            status = "PERFECT"
        elif count == 0:
            missed_tracks += 1
            status = "MISSED"
        else:
            status = f"BROKEN ({count})"
            
        print(f"{gid:<12} | {status:<15} | {tids}")

    print("-" * 55)
    print(f"Total Fragmentation Errors: {total_fragments}")
    print(f"Perfectly Tracked Fish:     {perfect_tracks}/{len(gt_to_tracker_map)}")
    print(f"Completely Missed Fish:     {missed_tracks}")

# ---------------------------------------------------------
# Run the Script
# ---------------------------------------------------------
# Replace filenames with your actual file paths
analyze_tracking_quality('mots/406_1_clean.txt', 'tracking_raw406.csv')
