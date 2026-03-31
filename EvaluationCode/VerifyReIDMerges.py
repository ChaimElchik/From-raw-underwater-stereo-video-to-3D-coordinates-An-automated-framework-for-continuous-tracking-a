import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
import argparse
import os

def calculate_iou(boxA, boxB):
    # box: [x, y, w, h] (center format from pipeline) or [x1, y1, x2, y2]
    # We need to standardize. 
    # Pipeline output is x, y, w, h (center)
    
    # Convert A (Center -> Corner)
    xA_1 = boxA[0] - boxA[2]/2
    yA_1 = boxA[1] - boxA[3]/2
    xA_2 = boxA[0] + boxA[2]/2
    yA_2 = boxA[1] + boxA[3]/2
    
    # Convert B (GT usually x, y, w, h or x1, y1, x2, y2?)
    # GT file: frame, id, x, y, w, h (often MOT format is top-left x,y,w,h)
    # Let's assume GT is read into x, y, w, h (center) for consistency OR check format.
    # Usually MOT 1.1: frame, id, left, top, width, height...
    # So we need to handle that.
    
    # Let's assume passed boxes are already [x1, y1, x2, y2] for simplicity in this func
    # RE-WRITING for x1,y1,x2,y2 input
    
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    if float(boxAArea + boxBArea - interArea) == 0: return 0.0
    return interArea / float(boxAArea + boxBArea - interArea)

def load_gt(gt_path):
    # Load MOT format: frame, id, x_left, y_top, w, h
    # The file has a header: "frame,id,x,y,x_offset,y_offset"
    try:
        df = pd.read_csv(gt_path)
        # Check if columns are correct strings "frame", "id" etc.
        # If the file has a header, pandas reads it automatically with read_csv(path).
        # We just need to ensure we map them to standard names.
        
        # Mapping based on observation: x->left, y->top, x_offset->w, y_offset->h
        # But let's check column names.
        # If columns are ['frame', 'id', 'x', 'y', 'x_offset', 'y_offset']
        rename_map = {
            'x': 'left', 
            'y': 'top', 
            'x_offset': 'w', 
            'y_offset': 'h'
        }
        df.rename(columns=rename_map, inplace=True)
        
        # Ensure numeric
        cols_to_numeric = ['frame', 'id', 'left', 'top', 'w', 'h']
        for c in cols_to_numeric:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
                
        df.dropna(subset=['frame', 'id'], inplace=True)
        return df
        
    except Exception as e:
        print(f"Error loading GT: {e}")
        return None

def map_tracks_to_gt(track_df, gt_df):
    track_to_gt = {}
    track_ids = track_df['id'].unique()
    print(f"Mapping {len(track_ids)} tracks to Ground Truth...")
    
    # Index GT by frame
    gt_by_frame = gt_df.groupby('frame')
    
    for tid in track_ids:
        t_data = track_df[track_df['id'] == tid]
        votes = {}
        
        # Sample frames (every 5th frame for speed)
        # But tracks can be short, so be careful.
        for _, row in t_data.iterrows():
            f = int(row['frame'])
            f_gt = f + 1 # 0-based -> 1-based
            
            if f_gt not in gt_by_frame.groups: continue
            gts = gt_by_frame.get_group(f_gt)
            
            # Tracker box (Center -> Corner) for IoU logic
            # My IoU func expects [x1, y1, x2, y2]
            tx_c, ty_c, tw, th = row['x'], row['y'], row['w'], row['h']
            t_box = [tx_c - tw/2, ty_c - th/2, tx_c + tw/2, ty_c + th/2]
            
            best_iou = 0
            best_gid = -1
            
            for _, g_row in gts.iterrows():
                # GT Box (Top-Left -> Corner)
                gx1, gy1 = g_row['left'], g_row['top']
                gw, gh = g_row['w'], g_row['h']
                g_box = [gx1, gy1, gx1 + gw, gy1 + gh]
                
                iou = calculate_iou(t_box, g_box)
                if iou > 0.3 and iou > best_iou: # IoU Threshold 0.3
                    best_iou = iou
                    best_gid = int(g_row['id'])
            
            if best_gid != -1:
                votes[best_gid] = votes.get(best_gid, 0) + 1
        
        if votes:
            winner_gt = max(votes, key=votes.get)
            # Confidence check?
            track_to_gt[tid] = winner_gt
        else:
            track_to_gt[tid] = None 
            
    return track_to_gt

def verify_merges(log_path, track_df, gt_df):
    mapping = map_tracks_to_gt(track_df, gt_df)
    
    good_merges = []
    bad_merges = []
    unknown_merges = []
    
    print("\n--- Verifying Merges from Log ---")
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        if "Merged ID" not in line: continue
        # Format: "Pass 1: Merged ID {child} -> {parent} (Cost: ...)"
        # Or: "Merged ID {child} -> {parent} (Cost: ...)" (Legacy)
        try:
            # Normalize line
            content = line.split("Merged ID")[1].strip()
            parts = content.split()
            # parts[0] = child, parts[2] = parent (assuming '->' is parts[1])
            
            child_id = int(parts[0])
            parent_id = int(parts[2])
            cost_str = parts[4].replace(')', '') # (Cost: 0.123) -> 0.123
            
            gt_child = mapping.get(child_id)
            gt_parent = mapping.get(parent_id)
            
            info = {
                'line': line.strip(),
                'child': child_id,
                'parent': parent_id,
                'gt_child': gt_child,
                'gt_parent': gt_parent
            }
            
            if gt_child is None or gt_parent is None:
                unknown_merges.append(info)
            elif gt_child == gt_parent:
                good_merges.append(info)
            else:
                bad_merges.append(info)
                
        except Exception as e:
            print(f"Skipping line: {line.strip()} ({e})")
            
    print("\n✅ CONFIRMED GOOD MATCHES:")
    for m in good_merges:
        print(f"  {m['line']}  [Both map to GT {m['gt_child']}]")
        
    print("\n❌ CONFIRMED BAD MATCHES:")
    for m in bad_merges:
        print(f"  {m['line']}  [GT {m['gt_child']} != GT {m['gt_parent']}]")
        
    print("\n❓ IMPROPER/NOISE MATCHES (One/Both is Noise):")
    for m in unknown_merges:
        print(f"  {m['line']}  [GT {m['gt_child']} -> {m['gt_parent']}]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--track_csv", required=True)
    parser.add_argument("--gt_file", required=True)
    parser.add_argument("--merge_log", required=True)
    args = parser.parse_args()
    
    tdf = pd.read_csv(args.track_csv)
    gdf = load_gt(args.gt_file)
    
    if gdf is not None:
        verify_merges(args.merge_log, tdf, gdf)
