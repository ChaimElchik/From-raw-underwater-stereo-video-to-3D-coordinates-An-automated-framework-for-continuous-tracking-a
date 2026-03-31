import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
GT_FILE = 'mots/406_1_clean.txt'
TRACKER_FILES = {
    'Raw (No BoTSORT)': 'tracking_raw46nobotsort.csv',
    'Raw (BoTSORT)':    'tracking_raw406.csv',
    'Processed':        'tracking_processedV4.csv'
}
IOU_THRESHOLD = 0.3

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def calculate_iou(boxA, boxB):
    # box format: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def evaluate_tracker(tracker_name, tracker_path, gt_df):
    # Load Tracker Data
    try:
        t_df = pd.read_csv(tracker_path)
    except FileNotFoundError:
        print(f"File not found: {tracker_path}")
        return None

    # --- Preprocessing ---
    # 1. Align Frame Numbers (Tracker 0-based -> GT 1-based)
    t_df['frame'] = t_df['frame'] + 1
    
    # 2. Standardize Coordinates to x1, y1, x2, y2
    # Tracker files are already xmin, ymin, xmax, ymax. We just rename for consistency.
    # Check if columns are named 'xmin' or just 'x'
    if 'xmin' in t_df.columns:
        t_df = t_df.rename(columns={'xmin': 'x1', 'ymin': 'y1', 'xmax': 'x2', 'ymax': 'y2'})
    else:
        # Fallback if names are different, assuming x,y,w,h or similar
        # For this specific dataset, we know it's xmin/ymin/xmax/ymax
        pass

    # --- Evaluation Loop ---
    gt_matches = {gid: [] for gid in gt_df['id'].unique()}
    all_frames = sorted(gt_df['frame'].unique())
    
    total_missed_frames = 0
    
    for frame in all_frames:
        gts = gt_df[gt_df['frame'] == frame]
        trs = t_df[t_df['frame'] == frame]
        
        gt_boxes = gts[['x1', 'y1', 'x2', 'y2']].values
        gt_ids = gts['id'].values
        
        # If no GT in this frame, skip
        if len(gts) == 0:
            continue
            
        # If no Tracker predictions in this frame, all GTs are missed
        if len(trs) == 0:
            for gid in gt_ids:
                gt_matches[gid].append(None)
            total_missed_frames += len(gts)
            continue
            
        tr_boxes = trs[['x1', 'y1', 'x2', 'y2']].values
        tr_ids = trs['id'].values
        
        # 1. Calculate IoU Matrix
        iou_matrix = np.zeros((len(gt_boxes), len(tr_boxes)))
        for i, gb in enumerate(gt_boxes):
            for j, tb in enumerate(tr_boxes):
                iou_matrix[i, j] = calculate_iou(gb, tb)
                
        # 2. Hungarian Algorithm (Linear Assignment)
        # We want to maximize IoU, so we minimize negative IoU
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        
        matched_gt_indices = set()
        
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= IOU_THRESHOLD:
                # Valid Match
                gt_matches[gt_ids[r]].append(tr_ids[c])
                matched_gt_indices.add(r)
        
        # 3. Handle Unmatched GTs (False Negatives)
        for i in range(len(gt_boxes)):
            if i not in matched_gt_indices:
                gt_matches[gt_ids[i]].append(None)
                total_missed_frames += 1

    # --- Calculate Metrics ---
    never_detected_count = 0
    total_id_switches = 0
    
    for gid, history in gt_matches.items():
        # Remove None values (missed frames) to check for ID consistency
        clean_history = [x for x in history if x is not None]
        
        if not clean_history:
            never_detected_count += 1
        else:
            # Count how many times the ID changes in the sequence
            curr_id = clean_history[0]
            for next_id in clean_history[1:]:
                if next_id != curr_id:
                    total_id_switches += 1
                    curr_id = next_id

    return {
        "Tracker Version": tracker_name,
        "Total Unique IDs": t_df['id'].nunique(),
        "Never Detected Fish": never_detected_count,
        "Total ID Switches": total_id_switches,
        "Total Missed Detections": total_missed_frames
    }

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
# 1. Load and Standardize Ground Truth
gt_df = pd.read_csv(GT_FILE)
# GT format: x, y (top-left), x_offset (width), y_offset (height)
gt_df['x1'] = gt_df['x']
gt_df['y1'] = gt_df['y']
gt_df['x2'] = gt_df['x'] + gt_df['x_offset']
gt_df['y2'] = gt_df['y'] + gt_df['y_offset']

# 2. Run Comparison
results = []
print(f"Comparing against Ground Truth: {GT_FILE}")
print("-" * 60)

for name, file_path in TRACKER_FILES.items():
    print(f"Processing {name}...")
    res = evaluate_tracker(name, file_path, gt_df)
    if res:
        results.append(res)

# 3. Display Results
results_df = pd.DataFrame(results)
print("\n" + "="*60)
print("FINAL COMPARISON RESULTS")
print("="*60)
print(results_df.to_string(index=False))