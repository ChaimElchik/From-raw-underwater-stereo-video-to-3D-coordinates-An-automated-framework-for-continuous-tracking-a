import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment

def evaluate_tracking(gt_path, tracker_path):
    print(f"Evaluating: {tracker_path}")
    
    # 1. Load Data
    gt = pd.read_csv(gt_path)
    tr = pd.read_csv(tracker_path)
    
    # 2. Align Data (GT is 1-based, Tracker is 0-based)
    tr['frame'] = tr['frame'] + 1
    
    # Standardize Column Names
    gt['x1'], gt['y1'] = gt['x'], gt['y']
    gt['x2'], gt['y2'] = gt['x'] + gt['x_offset'], gt['y'] + gt['y_offset']
    
    if 'xmin' in tr.columns:
        tr = tr.rename(columns={'xmin': 'x1', 'ymin': 'y1', 'xmax': 'x2', 'ymax': 'y2'})

    # 3. Initialize Metrics
    FN, FP, IDSW = 0, 0, 0
    gt_frames = 0
    
    # ID Confusion Matrix for IDF1 (Rows=GT, Cols=Tracker)
    gt_ids = sorted(gt['id'].unique())
    tr_ids = sorted(tr['id'].unique())
    id_matrix = np.zeros((len(gt_ids), len(tr_ids)))
    gt_map = {gid: i for i, gid in enumerate(gt_ids)}
    tr_map = {tid: i for i, tid in enumerate(tr_ids)}
    
    # Helper: IoU
    def get_iou(boxA, boxB):
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        inter = max(0, xB-xA) * max(0, yB-yA)
        areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
        areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
        return inter / (areaA + areaB - inter + 1e-6)

    # 4. Frame-by-Frame Processing
    last_matches = {} # GT_ID -> Tracker_ID
    
    for frame in sorted(gt['frame'].unique()):
        gts = gt[gt['frame'] == frame]
        trs = tr[tr['frame'] == frame]
        gt_frames += len(gts)
        
        if trs.empty:
            FN += len(gts)
            continue
            
        # IoU Matrix
        iou_mat = np.zeros((len(gts), len(trs)))
        gt_list = gts[['x1', 'y1', 'x2', 'y2', 'id']].values
        tr_list = trs[['x1', 'y1', 'x2', 'y2', 'id']].values
        
        for i, g in enumerate(gt_list):
            for j, t in enumerate(tr_list):
                iou_mat[i, j] = get_iou(g[:4], t[:4])
                
        # Hungarian Matching
        row_ind, col_ind = linear_sum_assignment(-iou_mat)
        
        matched_gt_indices = set()
        matched_tr_indices = set()
        
        for r, c in zip(row_ind, col_ind):
            if iou_mat[r, c] >= 0.3:
                matched_gt_indices.add(r)
                matched_tr_indices.add(c)
                
                gid = int(gt_list[r][4])
                tid = int(tr_list[c][4])
                
                # Update IDF1 Matrix
                if gid in gt_map and tid in tr_map:
                    id_matrix[gt_map[gid], tr_map[tid]] += 1
                
                # Check ID Switch
                if gid in last_matches:
                    if last_matches[gid] != tid:
                        IDSW += 1
                last_matches[gid] = tid
        
        FN += (len(gts) - len(matched_gt_indices))
        FP += (len(trs) - len(matched_tr_indices))

    # 5. Calculate Final Metrics
    MOTA = 1 - (FN + FP + IDSW) / gt_frames
    
    # Calculate IDF1 (Maximum Weight Matching on Global ID Matrix)
    row_ind, col_ind = linear_sum_assignment(-id_matrix)
    IDTP = id_matrix[row_ind, col_ind].sum()
    IDFN = gt_frames - IDTP
    IDFP = len(tr) - IDTP
    IDF1 = (2 * IDTP) / (2 * IDTP + IDFP + IDFN)
    
    print("-" * 30)
    print(f"MOTA (Accuracy):       {MOTA:.3f}")
    print(f"IDF1 (Identity Score): {IDF1:.3f}")
    print(f"ID Switches:           {IDSW}")
    print(f"False Negatives:       {FN}")
    print(f"False Positives:       {FP}")
    print("-" * 30)

# Example Usage
evaluate_tracking('mots/406_1_clean.txt', 'tracking_processed406.csv')