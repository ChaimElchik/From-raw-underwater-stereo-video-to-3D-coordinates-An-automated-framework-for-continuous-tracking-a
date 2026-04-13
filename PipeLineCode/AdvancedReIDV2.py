import torch
import torchvision.transforms as T
from torchvision.models import vit_b_16, ViT_B_16_Weights
import cv2
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import os
import argparse
import sys

# --- Configuration ---
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using Device: {DEVICE}")

# --- Helper Classes ---

class FeatureExtractor:
    def __init__(self):
        self.weights = ViT_B_16_Weights.DEFAULT
        self.model = vit_b_16(weights=self.weights).to(DEVICE)
        self.model.eval()
        self.transforms = self.weights.transforms()

    def extract(self, img_crops):
        if not img_crops:
            return None
        
        batch = []
        for img in img_crops:
            # Convert BGR (OpenCV) to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 1. Resize STRICTLY to 224x224 for ViT
            img_resized = cv2.resize(img_rgb, (224, 224))
            
            # 2. ToTensor (Converts to [C, H, W] and [0,1])
            tensor = T.ToTensor()(img_resized)
            
            # 3. Apply model-specific normalization (if needed, usually in self.transforms)
            # The default transforms usually include Resize and Normalize. 
            # We bypass the built-in resize by doing it manually above (to avoid bugs),
            # and normalize using standard ImageNet stats if self.transforms fails.
            try:
                # Some PyTorch versions allow applying just the normalization part
                # If transforms expects PIL, this might fail, hence manual fallback below.
                tensor = self.transforms(tensor) 
            except Exception:
                # Manual standard ImageNet normalization
                normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                tensor = normalize(tensor)

            batch.append(tensor)

        with torch.no_grad():
            batch_tensor = torch.stack(batch).to(DEVICE)
            # Output is [batch_size, 1000] (class logits), we use it as a feature vector
            features = self.model(batch_tensor)
            
            # Average features across the crops for this tracklet
            avg_feature = features.mean(dim=0).cpu().numpy()
            
            # Normalize to unit length for Cosine Similarity
            return avg_feature / np.linalg.norm(avg_feature)


class AdvancedReID:
    def __init__(self, tracking_data, video_path):
        if isinstance(tracking_data, str):
            print(f"Loading tracking data from {tracking_data}...")
            self.df = pd.read_csv(tracking_data)
        else:
            self.df = tracking_data.copy()
            
        self.video_path = video_path
        self.extractor = FeatureExtractor()
        
        # Precompute Stats to save massive time
        # dict: id -> {frames: set(), start: int, end: int, pos_end: (x,y), embedding: np.array}
        self.track_stats = {} 
        self.precompute_track_stats()

    def _get_crops(self, track_id, num_samples=5):
        """Extracts random crops for a track."""
        track_data = self.df[self.df['id'] == track_id].sort_values('frame')
        if len(track_data) == 0: return []

        # --- BAD TAIL REJECTION ---
        # If the track is long enough, drop the first 10% and last 10% of frames
        n_frames = len(track_data)
        if n_frames >= 5:
            trim_amount = max(1, int(n_frames * 0.1))
            track_data = track_data.iloc[trim_amount:-trim_amount]
        
        # Sample random frames
        frames_to_sample = track_data.sample(n=min(len(track_data), num_samples))
        
        crops = []
        cap = cv2.VideoCapture(self.video_path)
        
        for _, row in frames_to_sample.iterrows():
            frame_idx = int(row['frame'])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret: continue
            
            # Extract Crop (Handle standard x,y,w,h or x1,y1,x2,y2)
            # Assuming ProcessVideoPair output: frame, id, x, y, w, h
            # where x,y are CENTER
            x, y, w_b, h_b = row['x'], row['y'], row['w'], row['h']
            x1 = int(max(0, x - w_b/2))
            y1 = int(max(0, y - h_b/2))
            x2 = int(min(frame.shape[1], x + w_b/2))
            y2 = int(min(frame.shape[0], y + h_b/2))
            
            if x2 <= x1 or y2 <= y1: continue
            
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)
                
        cap.release()
        return crops

    def precompute_track_stats(self):
        print("Precomputing track stats and embeddings...")
        unique_ids = self.df['id'].unique()
        
        for uid in tqdm(unique_ids):
            track = self.df[self.df['id'] == uid].sort_values('frame')
            
            # Temporal
            frames = set(track['frame'].values)
            start_f = track['frame'].min()
            end_f = track['frame'].max()
            
            # Spatial
            last_row = track.iloc[-1]
            last_pos = (last_row['x'], last_row['y'])
            
            # Size
            avg_area = (track['w'] * track['h']).mean()
            
            # Visual Embedding
            crops = self._get_crops(uid)
            embedding = self.extractor.extract(crops)
            
            self.track_stats[uid] = {
                'frames': frames,
                'start': start_f,
                'end': end_f,
                'pos_end': last_pos,
                'pos_start': (track.iloc[0]['x'], track.iloc[0]['y']), # Needed for "next" track
                'area': avg_area,
                'embedding': embedding
            }

    def compute_cost_matrix(self):
        ids = list(self.track_stats.keys())
        ids.sort() # Fix order
        n = len(ids)
        # Initialize with a large finite cost (above any valid threshold)
        # linear_sum_assignment fails with np.inf if perfect match impossible
        LARGE_COST = 1000.0
        cost_matrix = np.full((n, n), LARGE_COST)
        
        print("Computing matching costs...")
        
        # We want to match existing track i (rows) to future track j (cols)
        # where End_i < Start_j
        
        for i_idx, id_i in enumerate(tqdm(ids)):
            stats_i = self.track_stats[id_i]
            
            for j_idx, id_j in enumerate(ids):
                if i_idx == j_idx: continue
                stats_j = self.track_stats[id_j]
                
                # 1. Temporal Constraints
                if not stats_i['frames'].isdisjoint(stats_j['frames']):
                    continue
                    
                # Check for Sequential vs Interleaved
                time_gap = stats_j['start'] - stats_i['end']
                
                # If time_gap is negative, it means tracks overlap in range (but are disjoint sets)
                # i.e., Interleaved (A-B-A-B or A surround B).
                # We ALLOW this but treat it as a "Gap Filling" merge.
                
                # 2. Visual Score
                if stats_i['embedding'] is None or stats_j['embedding'] is None:
                    continue
                    
                # Cosine Similarity
                sim = np.dot(stats_i['embedding'], stats_j['embedding']) / (
                    np.linalg.norm(stats_i['embedding']) * np.linalg.norm(stats_j['embedding'])
                )
                
                # TUNING:
                # For Interleaved/Negative Gap, we require HIGHER visual similarity to avoid accidents
                threshold = 0.85 if time_gap < 0 else 0.75
                
                if sim < threshold:
                    continue
                    
                vis_cost = 1 - sim
                
                # 3. Size Score (Z-Axis / Area Score)
                max_area = max(stats_i['area'], stats_j['area'])
                min_area = min(stats_i['area'], stats_j['area'])
                area_ratio = min_area / max_area
                
                # If the time gap is very short (under 10 frames), the size should be almost identical.
                if 0 < time_gap < 10:
                    if area_ratio < 0.80:  # Must be 80% similar in size
                        continue
                else:
                    # If they were lost for a long time, they might have swam closer/further from the lens
                    if area_ratio < 0.50:  
                        continue
                
                # 4. Motion Score
                # Only punish speed if positive gap
                if time_gap > 0:
                    gap_dist = np.linalg.norm(np.array(stats_i['pos_end']) - np.array(stats_j['pos_start']))
                    gap_speed = gap_dist / time_gap
                    if gap_speed > 30.0: # Relaxed slightly
                        continue
                else:
                    # Interleaved Tracks: We must find the chronological gap, not absolute ends.
                    # Approximation: If they interleave but are disjoint, the safest spatial check 
                    # is the minimum distance between their bounding box centers overall.
                    # If they are the same fish, their tracks should physically cross or touch.
                    gap_dist = np.linalg.norm(np.array(stats_i['pos_start']) - np.array(stats_j['pos_start']))
                    
                    # Apply a strict spatial gate for interleaved tracks
                    if gap_dist > 100: # Veto if the tracklets are nowhere near each other
                        continue
                    
                # Combined Cost
                # If time_gap > 0, penalty.
                # If time_gap < 0, 0 penalty (or slight penalty to prefer sequential if visual is equal)
                time_cost = max(0, time_gap) * 0.0001
                
                total_cost = vis_cost + (gap_dist * 0.005) + time_cost
                
                cost_matrix[i_idx, j_idx] = total_cost
                
        return ids, cost_matrix

    def run(self, thresholds=[0.20, 0.30, 0.50]):
        final_logs = []
        
        # Staged Passes + Convergence
        # We run the specific thresholds first.
        # For the LAST threshold, we repeat until convergence.
        
        pass_idx = 0
        while True:
            # Determine current threshold
            if pass_idx < len(thresholds):
                threshold = thresholds[pass_idx]
            else:
                # If we exhausted the list, keep using the last one until no moves
                threshold = thresholds[-1]
                
            print(f"\n--- Re-ID Pass {pass_idx + 1} (Threshold: {threshold}) ---")
            
            # 1. Update the Mapping
            # (In a real system we'd merge the embeddings. Here we just recalculate costs 
            # if we wanted to be super accurate, but since embeddings are static per ID,
            # we just re-run matching to see if new chains form).
            
            ids, costs = self.compute_cost_matrix()
            
            # 2. Hungarian Matching
            row_ind, col_ind = linear_sum_assignment(costs)
            
            moves = 0
            # Track which IDs have been mapped in THIS pass to prevent chains A->B->C 
            # resolving weirdly in a single pass.
            mapped_to = {} 
            
            for r, c in zip(row_ind, col_ind):
                if costs[r, c] < threshold:
                    id_a = ids[r] # Existing
                    id_b = ids[c] # Future
                    
                    # Ensure we aren't merging into something already merged
                    # Simplistic Union-Find approach for the dataframe:
                    
                    # Find true parent of A (in case A was merged in previous passes)
                    # Actually, we update the dataframe immediately, so id_a IS the current ID.
                    
                    parent_a = id_a
                    
                    if id_b in mapped_to: continue # B is already being consumed
                    
                    print(f"  Merge Candidate: {id_b} -> {parent_a} (Cost: {costs[r,c]:.4f})")
                    
                    # Execute Merge
                    self.df.loc[self.df['id'] == id_b, 'id'] = parent_a
                    
                    # Update stats for next pass (Merge the frames and recalculate ends)
                    # We don't recalculate embedding to save time, we just adopt A's embedding.
                    # Or better, we force a recompute of track stats for Parent A?
                    # For speed, we just combine temporal/spatial boundaries.
                    self.track_stats[parent_a]['frames'].update(self.track_stats[id_b]['frames'])
                    self.track_stats[parent_a]['end'] = max(self.track_stats[parent_a]['end'], self.track_stats[id_b]['end'])
                    
                    # Which position is the new end?
                    # Need to query DF to be safe.
                    track_a_full = self.df[self.df['id'] == parent_a].sort_values('frame')
                    self.track_stats[parent_a]['pos_end'] = (track_a_full.iloc[-1]['x'], track_a_full.iloc[-1]['y'])
                    
                    # Remove B from stats
                    del self.track_stats[id_b]
                    
                    mapped_to[id_b] = parent_a
                    moves += 1
                    final_logs.append(f"Pass {pass_idx+1}: Merged ID {id_b} -> {parent_a} (Cost: {costs[r,c]:.4f})")
                    
                    # We broke the index arrays by deleting from stats, so we must break and rebuild
                    break 
            
            print(f"Pass {pass_idx + 1} complete. {moves} merges.")
            
            # Convergence check
            if pass_idx >= len(thresholds) - 1 and moves == 0:
                print("Convergence reached. No more valid merges.")
                break
                
            pass_idx += 1
            
            # Safety break
            if pass_idx > 10:
                print("Max passes reached.")
                break
                
        # Final cleanup: re-number IDs to be continuous 1..N
        # (Optional, but good for MOT evaluation)
        # print("\nRe-indexing track IDs to be continuous...")
        # unique_final_ids = sorted(self.df['id'].unique())
        # id_map = {old: new+1 for new, old in enumerate(unique_final_ids)}
        # self.df['id'] = self.df['id'].map(id_map)
        
        return self.df, final_logs

# --- CLI Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_in", required=True, help="Input raw tracking CSV")
    parser.add_argument("--vid", required=True, help="Original Video for crops")
    parser.add_argument("--csv_out", default="reid_refined.csv", help="Output refined CSV")
    parser.add_argument("--log_out", default="reid_log.txt", help="Merge action log")
    parser.add_argument("--debug_pair", type=str, help="Comma separated ID pair to debug, e.g. '4,16'")
    
    args = parser.parse_args()

    if args.debug_pair:
        reid = AdvancedReID(args.csv_in, args.vid)
        p = args.debug_pair.split(',')
        if len(p) == 2:
            try:
                id_i = float(p[0])
                id_j = float(p[1])
                print(f"\n--- DEBUGGING MERGE {id_i} -> {id_j} ---")
                
                stats_i = reid.track_stats.get(id_i)
                stats_j = reid.track_stats.get(id_j)
                
                if not stats_i or not stats_j:
                    print("One or both IDs not found in tracks.")
                    sys.exit(0)
                    
                time_gap = stats_j['start'] - stats_i['end']
                print(f"  Time Gap: {time_gap} frames")
                
                # Visual
                sim = np.dot(stats_i['embedding'], stats_j['embedding']) / (
                    np.linalg.norm(stats_i['embedding']) * np.linalg.norm(stats_j['embedding'])
                )
                print(f"  Visual Sim: {sim:.4f} (Cost: {1-sim:.4f})")
                
                # Size
                area_ratio = min(stats_i['area'], stats_j['area']) / max(stats_i['area'], stats_j['area'])
                print(f"  Area i: {stats_i['area']:.1f}, Area j: {stats_j['area']:.1f}")
                print(f"  Size Ratio: {area_ratio:.4f}")
                
                # 4. Motion
                gap_dist = np.linalg.norm(np.array(stats_i['pos_end']) - np.array(stats_j['pos_start']))
                gap_speed = gap_dist / time_gap if time_gap > 0 else float('inf')
                print(f"  Gap Dist:   {gap_dist:.2f} px")
                print(f"  Gap Speed:  {gap_speed:.2f} px/frame")
                
                # Total
                vis_cost = 1 - sim
                total_cost = vis_cost + (gap_dist * 0.001)
                print(f"  TOTAL COST: {total_cost:.4f}")

            except ValueError:
                print(f"  Invalid pair format: {p}")
                
        sys.exit(0)

    reid = AdvancedReID(args.csv_in, args.vid)
    df_new, logs = reid.run()
    
    df_new.to_csv(args.csv_out, index=False)
    
    with open(args.log_out, 'w') as f:
        f.write("\n".join(logs))
        
    print(f"Advanced Re-ID Complete. {len(logs)} merges performed.")
    print(f"Results: {args.csv_out}")

def process_reid_pipeline(df, video_path):
    """
    Wrapper to run Advanced Re-ID on a DataFrame.
    Returns purified DataFrame.
    """
    print("\n--- Initializing Advanced Re-ID Pipeline ---")
    reid = AdvancedReID(df, video_path)
    # Tighter thresholds to prevent identity switches
    df_refined, logs = reid.run(thresholds=[0.12, 0.18, 0.22])
    
    print("\n[Re-ID Summary]")
    if logs:
        for log in logs:
            print(f"      {log}")
    else:
        print("      No confident merges found.")
        
    return df_refined