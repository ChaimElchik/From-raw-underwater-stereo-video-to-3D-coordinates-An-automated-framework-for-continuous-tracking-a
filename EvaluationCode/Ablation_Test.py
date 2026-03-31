# import cv2
# import pandas as pd
# import numpy as np
# import os
# from tqdm import tqdm
# from scipy.optimize import linear_sum_assignment

# # =============================================================================
# # CONFIGURATION
# # =============================================================================
# # UPDATE THESE PATHS TO MATCH YOUR DATA
# VIDEO_PATH = "vids/406_1.mp4"       # Your video file
# RAW_TRACKS_CSV = "tracking_raw406.csv"      # The output from BoT-SORT *before* Re-ID
# GT_FILE = "mots/406_1_clean.txt"            # Your Ground Truth file          

# # =============================================================================
# # 1. RE-ID LOGIC (MODULAR GATES)
# # =============================================================================

# class VideoLoader:
#     def __init__(self, video_path):
#         self.cap = cv2.VideoCapture(video_path)
#         if not self.cap.isOpened():
#             print(f"Warning: Could not open {video_path}.")
#             self.total_frames = 0
#         else:
#             self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     def get_frame(self, frame_number):
#         if frame_number >= self.total_frames: return None
#         self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
#         success, frame = self.cap.read()
#         return frame if success else None

#     def release(self):
#         self.cap.release()

# def get_track_max_speed(track_df):
#     if len(track_df) < 2: return 0.0
#     dx = track_df['xmin'].diff()
#     dy = track_df['ymin'].diff()
#     dist = np.sqrt(dx**2 + dy**2)
#     max_speed = np.nanmax(dist)
#     return 200.0 if max_speed > 200 else max_speed

# def get_crop_similarity(img1, bbox1, img2, bbox2):
#     # GATE 3: Visual Appearance (HSV Histogram)
#     h, w, _ = img1.shape
#     x1, y1, x2, y2 = map(int, bbox1)
#     crop1 = img1[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    
#     x1, y1, x2, y2 = map(int, bbox2)
#     crop2 = img2[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

#     if crop1.size == 0 or crop2.size == 0: return 0.0

#     hist1 = cv2.calcHist([cv2.cvtColor(crop1, cv2.COLOR_BGR2HSV)], [0, 1], None, [180, 256], [0, 180, 0, 256])
#     hist2 = cv2.calcHist([cv2.cvtColor(crop2, cv2.COLOR_BGR2HSV)], [0, 1], None, [180, 256], [0, 180, 0, 256])

#     cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
#     cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

#     return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

# def solve_homography_and_check(img_a, img_b, center_a, center_b, max_pixel_dist):
#     # GATE 2: Homography (Geometric Consistency)
#     sift = cv2.SIFT_create()
#     kp1, des1 = sift.detectAndCompute(img_a, None)
#     kp2, des2 = sift.detectAndCompute(img_b, None)
#     if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4: return False, 0.0

#     flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
#     matches = flann.knnMatch(des1, des2, k=2)
#     good = [m for m, n in matches if m.distance < 0.75 * n.distance]

#     if len(good) < 4: return False, 0.0
#     pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
#     pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

#     H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
#     if H is None: return False, 0.0

#     p_original = np.array([center_a[0], center_a[1], 1.0]).reshape(3, 1)
#     p_transformed = np.dot(H, p_original)
#     p_transformed = p_transformed / (p_transformed[2] + 1e-8)

#     new_x, new_y = p_transformed[0][0], p_transformed[1][0]
#     dist = np.linalg.norm([new_x - center_b[0], new_y - center_b[1]])

#     return dist < max_pixel_dist, dist

# def run_reid_ablation(df_raw, video_path, use_homography=True, use_visual=True):
#     # Define experiment name
#     config = []
#     if use_homography: config.append("Homography")
#     if use_visual: config.append("Visual")
#     exp_name = "Kinematic + " + " + ".join(config) if config else "Kinematic Only"
    
#     print(f"\n[Running] {exp_name}...")
    
#     df_clean = df_raw.copy()
#     loader = VideoLoader(video_path)
#     if loader.total_frames == 0: return df_clean, 0

#     track_max_speeds = {}
#     for tid, group in df_clean.groupby('id'):
#         track_max_speeds[tid] = get_track_max_speed(group)
#     global_max = np.median([v for v in track_max_speeds.values() if v > 0]) if track_max_speeds else 20.0

#     df_starts = df_clean.sort_values('frame').groupby('id').first().reset_index()
#     df_ends = df_clean.sort_values('frame').groupby('id').last().reset_index()

#     id_map = {id: id for id in df_clean['id'].unique()}
#     def get_parent(i):
#         if id_map[i] != i: id_map[i] = get_parent(id_map[i])
#         return id_map[i]
#     def union(i, j):
#         root_i, root_j = get_parent(i), get_parent(j)
#         if root_i != root_j:
#             id_map[root_j] = root_i
#             return True
#         return False

#     ends_list = df_ends.to_dict('records')
#     starts_list = df_starts.to_dict('records')
#     valid_merges = []
#     frame_cache = {}

#     for track_a in tqdm(ends_list, desc="  Checking Pairs"):
#         max_speed_a = track_max_speeds.get(track_a['id'], global_max)
#         burst_potential = max(max_speed_a * 1.5, 1.0)

#         for track_b in starts_list:
#             time_gap = track_b['frame'] - track_a['frame']
#             if time_gap <= 0: continue 

#             possible_travel = burst_potential * time_gap
            
#             # --- GATE 1: KINEMATIC (Always On for efficiency) ---
#             raw_dist = np.sqrt((track_a['xmin']-track_b['xmin'])**2 + (track_a['ymin']-track_b['ymin'])**2)
#             if raw_dist > possible_travel: continue
            
#             if get_parent(track_a['id']) == get_parent(track_b['id']): continue

#             # Lazy Frame Loading
#             if use_homography or use_visual:
#                 if track_a['frame'] not in frame_cache:
#                     frame_cache[track_a['frame']] = loader.get_frame(track_a['frame'])
#                 if track_b['frame'] not in frame_cache:
#                     frame_cache[track_b['frame']] = loader.get_frame(track_b['frame'])
#                 img_a, img_b = frame_cache[track_a['frame']], frame_cache[track_b['frame']]
#                 if img_a is None or img_b is None: continue

#             # --- GATE 2: HOMOGRAPHY ---
#             homography_passed = True
#             if use_homography:
#                 center_a = ((track_a['xmin']+track_a['xmax'])/2, (track_a['ymin']+track_a['ymax'])/2)
#                 center_b = ((track_b['xmin']+track_b['xmax'])/2, (track_b['ymin']+track_b['ymax'])/2)
#                 is_consistent, _ = solve_homography_and_check(img_a, img_b, center_a, center_b, possible_travel)
#                 if not is_consistent:
#                     homography_passed = False
            
#             if not homography_passed: continue

#             # --- GATE 3: VISUAL ---
#             visual_passed = True
#             score = 1.0
#             if use_visual:
#                 bbox_a = [track_a['xmin'], track_a['ymin'], track_a['xmax'], track_a['ymax']]
#                 bbox_b = [track_b['xmin'], track_b['ymin'], track_b['xmax'], track_b['ymax']]
#                 score = get_crop_similarity(img_a, bbox_a, img_b, bbox_b)
#                 if score < 0.4:
#                     visual_passed = False
            
#             if visual_passed:
#                 valid_merges.append({'id_a': track_a['id'], 'id_b': track_b['id'], 'score': score})

#     loader.release()
    
#     valid_merges.sort(key=lambda x: -x['score'])
#     merge_count = 0
#     for m in valid_merges:
#         if union(m['id_a'], m['id_b']):
#             merge_count += 1
            
#     df_clean['id'] = df_clean['id'].apply(lambda x: get_parent(x))
#     return df_clean, merge_count, exp_name

# # =============================================================================
# # 2. EVALUATION METRICS
# # =============================================================================

# def calculate_iou(boxA, boxB):
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])
#     interArea = max(0, xB - xA) * max(0, yB - yA)
#     boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
#     boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
#     return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

# def evaluate_tracker(tracker_name, t_df, gt_df, merges_performed):
#     # Align Data
#     t_df = t_df.copy()
#     t_df['frame'] = t_df['frame'] + 1
#     if 'xmin' in t_df.columns:
#         t_df = t_df.rename(columns={'xmin': 'x1', 'ymin': 'y1', 'xmax': 'x2', 'ymax': 'y2'})

#     gt_matches = {gid: [] for gid in gt_df['id'].unique()}
#     all_frames = sorted(gt_df['frame'].unique())
    
#     matched_gt_boxes = 0
#     total_gt_boxes = len(gt_df)

#     for frame in all_frames:
#         gts = gt_df[gt_df['frame'] == frame]
#         trs = t_df[t_df['frame'] == frame]
        
#         gt_boxes = gts[['x1', 'y1', 'x2', 'y2']].values
#         gt_ids = gts['id'].values
#         tr_boxes = trs[['x1', 'y1', 'x2', 'y2']].values if len(trs) > 0 else []
#         tr_ids = trs['id'].values if len(trs) > 0 else []
        
#         if len(tr_boxes) == 0:
#             for gid in gt_ids: gt_matches[gid].append(None)
#             continue

#         iou_matrix = np.zeros((len(gt_boxes), len(tr_boxes)))
#         for i, gb in enumerate(gt_boxes):
#             for j, tb in enumerate(tr_boxes):
#                 iou_matrix[i, j] = calculate_iou(gb, tb)
        
#         row_ind, col_ind = linear_sum_assignment(-iou_matrix)
#         matched_gt_indices = set()
        
#         for r, c in zip(row_ind, col_ind):
#             if iou_matrix[r, c] >= 0.3:
#                 gt_matches[gt_ids[r]].append(tr_ids[c])
#                 matched_gt_indices.add(r)
#                 matched_gt_boxes += 1
        
#         for i in range(len(gt_boxes)):
#             if i not in matched_gt_indices:
#                 gt_matches[gt_ids[i]].append(None)

#     id_switches = 0
#     for gid, history in gt_matches.items():
#         clean_history = [x for x in history if x is not None]
#         if clean_history:
#             curr_id = clean_history[0]
#             for next_id in clean_history[1:]:
#                 if next_id != curr_id:
#                     id_switches += 1
#                     curr_id = next_id

#     recall = matched_gt_boxes / total_gt_boxes if total_gt_boxes > 0 else 0
    
#     return {
#         "Config": tracker_name,
#         "Merges": merges_performed,
#         "Final IDs": t_df['id'].nunique(),
#         "GT IDs": gt_df['id'].nunique(),
#         "ID Switches": id_switches,
#         "Recall": f"{recall:.1%}"
#     }

# # =============================================================================
# # MAIN EXECUTION
# # =============================================================================

# if __name__ == "__main__":
#     if not os.path.exists(RAW_TRACKS_CSV):
#         print(f"Error: {RAW_TRACKS_CSV} not found.")
#         exit()

#     gt_df = pd.read_csv(GT_FILE)
#     # GT Fix
#     gt_df['x1'] = gt_df['x']
#     gt_df['y1'] = gt_df['y']
#     gt_df['x2'] = gt_df['x'] + gt_df['x_offset']
#     gt_df['y2'] = gt_df['y'] + gt_df['y_offset']

#     raw_df = pd.read_csv(RAW_TRACKS_CSV)
    
#     print("--- 3-GATE ABLATION STUDY ---")
    
#     experiments = []
    
#     # 1. Kinematic Only
#     df_1, m_1, name_1 = run_reid_ablation(raw_df, VIDEO_PATH, use_homography=False, use_visual=False)
#     experiments.append(evaluate_tracker(name_1, df_1, gt_df, m_1))
    
#     # 2. Kinematic + Homography
#     df_2, m_2, name_2 = run_reid_ablation(raw_df, VIDEO_PATH, use_homography=True, use_visual=False)
#     experiments.append(evaluate_tracker(name_2, df_2, gt_df, m_2))
    
#     # 3. Kinematic + Visual (To see if Vis alone is better/worse than Hom alone)
#     df_3, m_3, name_3 = run_reid_ablation(raw_df, VIDEO_PATH, use_homography=False, use_visual=True)
#     experiments.append(evaluate_tracker(name_3, df_3, gt_df, m_3))

#     # 4. Full Pipeline
#     df_4, m_4, name_4 = run_reid_ablation(raw_df, VIDEO_PATH, use_homography=True, use_visual=True)
#     experiments.append(evaluate_tracker(name_4, df_4, gt_df, m_4))
    
#     # Output
#     results = pd.DataFrame(experiments)
#     print("\n" + "="*90)
#     print("ABLATION RESULTS: CONTRIBUTION OF EACH GATE")
#     print("="*90)
#     print(results.to_string(index=False))
import cv2
import pandas as pd
import numpy as np
import os
import torch
import torchvision.models as models
import torchvision.transforms as T
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

# =============================================================================
# CONFIGURATION
# =============================================================================
VIDEO_PATH = "vids/406_1.mp4"       
RAW_TRACKS_CSV = "tracking_raw406.csv"      
GT_FILE = "mots/406_1_clean.txt"            

SIMILARITY_THRESHOLD = 0.90  
TEMPORAL_BUFFER = 10         
MAX_GAP_FRAMES = 200         
MAX_SPEED_PX = 25            

# =============================================================================
# 1. HELPERS & DEEP LEARNING
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
resnet.eval()

transform = T.Compose([
    T.ToPILImage(), T.Resize((224, 224)), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_embedding(img_crop):
    if img_crop.size == 0: return None
    img_t = transform(img_crop).unsqueeze(0).to(device)
    with torch.no_grad(): return resnet(img_t)

def standardize_df(df):
    """Ensures centers (x, y) and corners (xmin, ymin, xmax, ymax) exist."""
    df = df.copy()
    # If format is x,y,w,h (Top-Left)
    if 'xmin' not in df.columns:
        df['xmin'] = df['x']
        df['ymin'] = df['y']
        df['xmax'] = df['x'] + df['w']
        df['ymax'] = df['y'] + df['h']
    
    # Always recalculate centers to be safe
    df['x'] = (df['xmin'] + df['xmax']) / 2
    df['y'] = (df['ymin'] + df['ymax']) / 2
    return df

# =============================================================================
# 2. THE FINAL RE-ID ENGINE (Averaging + Exclusion)
# =============================================================================

def run_reid_final(df_raw, video_path):
    print("\n--- RUNNING FINAL STRATEGY RE-ID ---")
    df = standardize_df(df_raw)
    track_ids = df['id'].unique()
    
    # Exclusion Gate: Map intervals
    track_intervals = {tid: (df[df['id']==tid]['frame'].min(), df[df['id']==tid]['frame'].max()) 
                       for tid in track_ids}
    
    loader = cv2.VideoCapture(video_path)
    
    # 1. Temporal Averaging (Denoising)
    print(f"Pre-calculating Temporal Embeddings (N={TEMPORAL_BUFFER})...")
    tail_embeddings = {} 
    head_embeddings = {} 
    
    for tid in tqdm(track_ids, desc="  Feature Extraction"):
        group = df[df['id'] == tid].sort_values('frame')
        
        # Tail (End of track)
        tail_rows = group.tail(TEMPORAL_BUFFER)
        tail_embs = []
        for _, row in tail_rows.iterrows():
            loader.set(cv2.CAP_PROP_POS_FRAMES, int(row['frame']))
            ret, frame = loader.read()
            if ret:
                crop = frame[int(row['ymin']):int(row['ymax']), int(row['xmin']):int(row['xmax'])]
                emb = get_embedding(crop)
                if emb is not None: tail_embs.append(emb)
        if tail_embs:
            tail_embeddings[tid] = torch.mean(torch.stack(tail_embs), dim=0)

        # Head (Start of track)
        head_rows = group.head(TEMPORAL_BUFFER)
        head_embs = []
        for _, row in head_rows.iterrows():
            loader.set(cv2.CAP_PROP_POS_FRAMES, int(row['frame']))
            ret, frame = loader.read()
            if ret:
                crop = frame[int(row['ymin']):int(row['ymax']), int(row['xmin']):int(row['xmax'])]
                emb = get_embedding(crop)
                if emb is not None: head_embs.append(emb)
        if head_embs:
            head_embeddings[tid] = torch.mean(torch.stack(head_embs), dim=0)

    # 2. Matching with Physical Exclusion
    id_map = {tid: tid for tid in track_ids}
    def get_p(i):
        if id_map[i] != i: id_map[i] = get_p(id_map[i])
        return id_map[i]
    
    potential_merges = []
    print("Evaluating Candidate Merges...")
    for tid_a in track_ids:
        for tid_b in track_ids:
            if tid_a == tid_b: continue
            
            start_a, end_a = track_intervals[tid_a]
            start_b, end_b = track_intervals[tid_b]
            
            # --- GATE 1: EXCLUSION GATE (Temporal Overlap) ---
            # If tracks co-exist, they are different fish.
            if max(start_a, start_b) <= min(end_a, end_b): continue 
            
            # Ensure A is before B
            if end_a > start_b: continue
            gap = start_b - end_a
            if gap > MAX_GAP_FRAMES: continue
            
            # --- GATE 2: KINEMATIC GATE ---
            # FIX: Get specifically centered coordinates
            pos_a = df[(df['id']==tid_a) & (df['frame']==end_a)].iloc[0]
            pos_b = df[(df['id']==tid_b) & (df['frame']==start_b)].iloc[0]
            dist = np.sqrt((pos_a['x'] - pos_b['x'])**2 + (pos_a['y'] - pos_b['y'])**2)
            
            if dist > (MAX_SPEED_PX * (gap + 1)): continue
            
            # --- GATE 3: TEMPORAL EMBEDDING GATE ---
            if tid_a in tail_embeddings and tid_b in head_embeddings:
                sim = cosine_similarity(tail_embeddings[tid_a], head_embeddings[tid_b]).item()
                if sim > SIMILARITY_THRESHOLD:
                    potential_merges.append((tid_a, tid_b, sim))

    # Apply Merges
    potential_merges.sort(key=lambda x: -x[2])
    merge_count = 0
    for tid_a, tid_b, _ in potential_merges:
        root_a, root_b = get_p(tid_a), get_p(tid_b)
        if root_a != root_b:
            id_map[root_b] = root_a
            merge_count += 1
            
    df['id'] = df['id'].apply(lambda x: get_p(x))
    loader.release()
    return df, potential_merges

# =============================================================================
# 3. METRICS
# =============================================================================

def calculate_iou(boxA, boxB):
    xA, yA, xB, yB = max(boxA[0], boxB[0]), max(boxA[1], boxB[1]), min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    return inter / float((boxA[2]-boxA[0])*(boxA[3]-boxA[1]) + (boxB[2]-boxB[0])*(boxB[3]-boxB[1]) - inter + 1e-6)

def get_ultimate_metrics(name, t_df, gt_df, merges_list):
    t_df = standardize_df(t_df)
    t_to_bio = {}
    bio_to_t = {gid: [] for gid in gt_df['id'].unique()}
    for tid in t_df['id'].unique():
        subset = t_df[t_df['id'] == tid]
        matches = []
        for _, row in subset.iterrows():
            gts = gt_df[gt_df['frame'] == (row['frame'] + 1)]
            for _, grow in gts.iterrows():
                if calculate_iou([row['xmin'], row['ymin'], row['xmax'], row['ymax']], 
                                 [grow['x1'], grow['y1'], grow['x2'], grow['y2']]) > 0.3:
                    matches.append(grow['id'])
        if matches:
            best_bio = max(set(matches), key=matches.count)
            t_to_bio[tid] = best_bio
            bio_to_t[best_bio].append(tid)

    cor, incor = 0, 0
    # Merges list is (id_a, id_b, score)
    for tid_a, tid_b, _ in merges_list:
        if t_to_bio.get(tid_a) == t_to_bio.get(tid_b) and t_to_bio.get(tid_a) is not None: cor += 1
        else: incor += 1
    perf = sum(1 for v in bio_to_t.values() if len(set(v)) == 1)
    return {"Config": name, "IDs": t_df['id'].nunique(), "Cor. Merges": cor, "Incor. Merges": incor, "Perf. Tracked": perf}

if __name__ == "__main__":
    gt_df = pd.read_csv(GT_FILE)
    gt_df['x1']=gt_df['x']; gt_df['y1']=gt_df['y']; gt_df['x2']=gt_df['x']+gt_df['x_offset']; gt_df['y2']=gt_df['y']+gt_df['y_offset']
    raw_df = pd.read_csv(RAW_TRACKS_CSV)
    
    # 1. Baseline
    res_base = get_ultimate_metrics("No Re-ID (Baseline)", standardize_df(raw_df), gt_df, [])
    
    # 2. Final Strategy
    df_final, m_final = run_reid_final(raw_df, VIDEO_PATH)
    res_final = get_ultimate_metrics("Final Strategy (Exclusion+Avg)", df_final, gt_df, m_final)
    
    print("\n" + "="*95)
    print(pd.DataFrame([res_base, res_final]).to_string(index=False))
    print("="*95)