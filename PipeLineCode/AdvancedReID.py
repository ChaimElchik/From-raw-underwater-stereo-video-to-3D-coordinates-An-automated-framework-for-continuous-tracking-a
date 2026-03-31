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
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
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
            # Since we manually resized, we should check what self.transforms does.
            # Usually: Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize
            # To be safe and simple, let's just Normalize manually or trust the model.
            # Better: Apply the full transform chain to the PIL image if possible, 
            # but since we have numpy array, let's just stack the resized tensors 
            # and normalize using standard ImageNet stats if self.transforms fails.
            
            # Let's rely on self.transforms but apply it PER IMAGE to avoid stack error
            # BUT self.transforms usually expects PIL or Tensor. 
            # Let's try applying self.transforms to the tensor we just made.
            # Note: T.Resize in transforms expects [C, H, W] tensor.
            
            # RESET: The robust fix is to use the model's transforms on each item.
            # We convert numpy -> PIL -> transforms -> Tensor
            img_pil = T.ToPILImage()(img_rgb)
            tensor = self.transforms(img_pil) # Output is [3, 224, 224] normalized tensor
            batch.append(tensor)
            
        if not batch: return None
        
        # Stack into batch [B, 3, 224, 224]
        input_tensor = torch.stack(batch).to(DEVICE)
        
        with torch.no_grad():
            features = self.model(input_tensor)
        
        # Average pooling
        avg_feature = torch.mean(features, dim=0).cpu().numpy()
        return avg_feature

class AdvancedReID:
    def __init__(self, tracking_data, video_path):
        """
        tracking_data: str (path to csv) or pd.DataFrame
        video_path: str (path to video)
        """
        if isinstance(tracking_data, str):
            self.df = pd.read_csv(tracking_data)
        elif isinstance(tracking_data, pd.DataFrame):
            self.df = tracking_data.copy()
        else:
            raise ValueError("tracking_data must be a csv path or a pandas DataFrame")
        
        # Explicit type conversion to avoid string comparison bugs
        # Handle cases where column names might have spaces or issues (but here we assume 'frame', 'id')
        self.df['frame'] = pd.to_numeric(self.df['frame'], errors='coerce')
        self.df['id'] = pd.to_numeric(self.df['id'], errors='coerce')
        self.df.dropna(subset=['frame', 'id'], inplace=True)
        self.df['frame'] = self.df['frame'].astype(int)
        self.df['id'] = self.df['id'].astype(int)
        
        print(f"Data Loaded. Frame Type: {self.df['frame'].dtype}, ID Type: {self.df['id'].dtype}")

        self.video_path = video_path
        self.extractor = FeatureExtractor()
        
        # Output DataFrame
        self.result_df = self.df.copy()
        
        # Statistics per track
        self.track_stats = {} 
        self.id_mapping = {id: id for id in self.df['id'].unique()}

    def _get_crops(self, track_id, num_samples=5):
        """Extracts random crops for a track."""
        track_data = self.df[self.df['id'] == track_id]
        if len(track_data) == 0: return []
        
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
                
                # 3. Size Score
                area_ratio = min(stats_i['area'], stats_j['area']) / max(stats_i['area'], stats_j['area'])
                if area_ratio < 0.6:
                    continue
                
                # 4. Motion Score
                # For Interleaved, gap_dist might be misleading (end of A very far from start of B)
                # But we have no better metric without detailed segment analysis.
                # We assume if they identify as same fish visually, they are valid.
                gap_dist = np.linalg.norm(np.array(stats_i['pos_end']) - np.array(stats_j['pos_start']))
                
                # Only punish speed if positive gap
                if time_gap > 0:
                    gap_speed = gap_dist / time_gap
                    if gap_speed > 30.0: # Relaxed slightly
                        continue
                else:
                    # Interleaved: ignore speed (infinite/negative), reliance on spatial proximity is tricky
                    # We penalize distance linearly
                    pass
                    
                # Combined Cost
                # If time_gap > 0, penalty.
                # If time_gap < 0, 0 penalty (or slight penalty to prefer sequential if visual is equal)
                time_cost = max(0, time_gap) * 0.0001
                
                total_cost = vis_cost + (gap_dist * 0.005) + time_cost
                
                cost_matrix[i_idx, j_idx] = total_cost
                
        return ids, cost_matrix

    def run(self, thresholds=[0.12, 0.18, 0.22]):
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
            
            self.df = self.result_df.copy() 
            self.track_stats = {} 
            self.precompute_track_stats() 
            
            ids, costs = self.compute_cost_matrix()
            if len(ids) < 2: break
            
            row_ind, col_ind = linear_sum_assignment(costs)
            
            moves = 0
            for r, c in zip(row_ind, col_ind):
                if costs[r, c] < threshold: 
                    id_a = ids[r] 
                    id_b = ids[c] 
                    
                    if id_a == id_b: continue
                    
                    # 1. Resolve Root Parent
                    parent_a = self.id_mapping[id_a]
                    while parent_a != self.id_mapping[parent_a]:
                        parent_a = self.id_mapping[parent_a]
                    
                    if parent_a == id_b: continue
                    
                    # STRICT DISJOINT CHECK (Regression Fix for ID 43)
                    # Even if visual match is perfect, we CANNOT merge if they share frames.
                    # Note: We must check the RESLOVED PARENT's frames vs Child's frames.
                    if not self.track_stats[parent_a]['frames'].isdisjoint(self.track_stats[id_b]['frames']):
                        # print(f"Skipping overlap: {parent_a} vs {id_b}")
                        continue

                    # 2. Update Mapping
                    self.id_mapping[id_b] = parent_a
                    
                    # 3. Update DataFrame
                    self.result_df.loc[self.result_df['id'] == id_b, 'id'] = parent_a
                    
                    # 4. Update Stats
                    self._update_stats_post_merge(parent_a, id_b)
                    
                    final_logs.append(f"Pass {pass_idx+1}: Merged ID {id_b} -> {parent_a} (Cost: {costs[r,c]:.4f})")
                    moves += 1
            
            print(f"Pass {pass_idx + 1} complete. {moves} merges.")
            
            # Convergence Condition
            if pass_idx >= len(thresholds) - 1 and moves == 0:
                print("Convergence reached.")
                break
                
            pass_idx += 1
            
            # Safety break
            if pass_idx > 10:
                print("Max passes reached.")
                break

            
        return self.result_df, final_logs

    def _update_stats_post_merge(self, parent_id, child_id):
        """
        Updates parent stats with child's info to enable chain matching.
        """
        if parent_id not in self.track_stats or child_id not in self.track_stats:
            return

        p_stats = self.track_stats[parent_id]
        c_stats = self.track_stats[child_id]
        
        # 1. Update Temporal Extents
        if c_stats['end'] > p_stats['end']:
            p_stats['end'] = c_stats['end']
            p_stats['pos_end'] = c_stats['pos_end']
            if c_stats['embedding'] is not None:
                p_stats['embedding'] = c_stats['embedding']
        
        if c_stats['start'] < p_stats['start']:
            p_stats['start'] = c_stats['start']
            p_stats['pos_start'] = c_stats['pos_start']
            
        p_stats['frames'].update(c_stats['frames'])
        p_stats['area'] = (p_stats['area'] + c_stats['area']) / 2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_in", required=True)
    parser.add_argument("--vid", required=True)
    parser.add_argument("--csv_out", required=False)
    parser.add_argument("--log_out", default="merge_log.txt")
    parser.add_argument("--inspect", help="Space-separated pairs to check e.g. '1,2 3,4'")
    
    args = parser.parse_args()
    
    # --- INSPECTION MODE ---
    # Quick hack to inspect specific pairs without running full matching
    # Format: "idA,idB idC,idD"
    if hasattr(args, 'inspect') and args.inspect:
        reid = AdvancedReID(args.csv_in, args.vid)
        reid.precompute_track_stats()
        
        pairs = args.inspect.split()
        print(f"\n🔍 INSPECTING {len(pairs)} PAIRS:")
        
        for p in pairs:
            try:
                s_a, s_b = p.split(',')
                id_a, id_b = int(s_a), int(s_b)
                
                print(f"\n--- PAIR ({id_a}, {id_b}) ---")
                if id_a not in reid.track_stats or id_b not in reid.track_stats:
                    print("  One or both IDs not found in stats.")
                    continue
                    
                stats_i = reid.track_stats[id_a]
                stats_j = reid.track_stats[id_b]
                
                # 1. Temporal
                disjoint = stats_i['frames'].isdisjoint(stats_j['frames'])
                t_str = "OK (Disjoint)" if disjoint else "FAIL (Overlap)"
                print(f"  Temporal: {t_str}")
                if not disjoint:
                    overlap = stats_i['frames'].intersection(stats_j['frames'])
                    print(f"    Overlap Frames: {sorted(list(overlap))}")
                    
                time_gap = stats_j['start'] - stats_i['end']
                print(f"  Time Gap: {time_gap} frames")
                
                # 2. Visual
                sim = 0.0
                if stats_i['embedding'] is not None and stats_j['embedding'] is not None:
                    sim = np.dot(stats_i['embedding'], stats_j['embedding']) / (
                        np.linalg.norm(stats_i['embedding']) * np.linalg.norm(stats_j['embedding'])
                    )
                print(f"  Visual Sim: {sim:.4f} (Cost: {1-sim:.4f})")
                
                # 3. Size
                area_ratio = min(stats_i['area'], stats_j['area']) / max(stats_i['area'], stats_j['area'])
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
    reid = AdvancedReID(df, video_path)
    df_new, logs = reid.run()
    
    # Print logs to console since we are integrated
    if logs:
        print("\n    [Re-ID Summary]")
        for log in logs:
            print(f"      {log}")
    else:
        print("    [Re-ID] No merges performed.")
        
    return df_new
