import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import os
import sys
import torch
from tqdm import tqdm

# --- RE-ID HELPER CLASSES & FUNCTIONS ---

class VideoLoader:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_frame(self, frame_number):
        if frame_number >= self.total_frames: return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = self.cap.read()
        return frame if success else None

    def release(self):
        self.cap.release()

def get_track_max_speed(track_df):
    """Calculates the MAXIMUM speed (pixels/frame) observed for a track."""
    if len(track_df) < 2: return 0.0
    dx = track_df['xmin'].diff()
    dy = track_df['ymin'].diff()
    dist = np.sqrt(dx**2 + dy**2)
    max_speed = np.nanmax(dist)
    if max_speed > 200: max_speed = 200.0
    return max_speed

def get_crop_similarity(img1, bbox1, img2, bbox2):
    h, w, _ = img1.shape
    x1, y1, x2, y2 = map(int, bbox1)
    crop1 = img1[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

    x1, y1, x2, y2 = map(int, bbox2)
    crop2 = img2[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

    if crop1.size == 0 or crop2.size == 0: return 0.0

    hist1 = cv2.calcHist([cv2.cvtColor(crop1, cv2.COLOR_BGR2HSV)], [0, 1], None, [180, 256], [0, 180, 0, 256])
    hist2 = cv2.calcHist([cv2.cvtColor(crop2, cv2.COLOR_BGR2HSV)], [0, 1], None, [180, 256], [0, 180, 0, 256])

    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def solve_homography_and_check(img_a, img_b, center_a, center_b, max_pixel_dist):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_a, None)
    kp2, des2 = sift.detectAndCompute(img_b, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4: return False, 0.0

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good) < 4: return False, 0.0
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    if H is None: return False, 0.0

    p_original = np.array([center_a[0], center_a[1], 1.0]).reshape(3, 1)
    p_transformed = np.dot(H, p_original)
    p_transformed = p_transformed / (p_transformed[2] + 1e-8)

    new_x, new_y = p_transformed[0][0], p_transformed[1][0]
    dist = np.linalg.norm([new_x - center_b[0], new_y - center_b[1]])

    return dist < max_pixel_dist, dist

def process_reid(df_raw, video_path):
    """
    Main Re-ID logic that merges fragmented tracks based on visual appearance 
    and kinematic feasibility.
    """
    print(f"\n🧠 Running Re-ID Refinement on {len(df_raw)} detections...")

    df_clean = df_raw.copy()
    loader = VideoLoader(video_path)

    # 1. Calculate Per-ID MAX Speed
    track_max_speeds = {}
    for tid, group in df_clean.groupby('id'):
        track_max_speeds[tid] = get_track_max_speed(group)

    global_max = np.median([v for v in track_max_speeds.values() if v > 0]) if track_max_speeds else 20.0
    print(f"    Typical Max Fish Speed: {global_max:.2f} px/frame")

    # 2. Get Endpoints
    df_starts = df_clean.sort_values('frame').groupby('id').first().reset_index()
    df_ends = df_clean.sort_values('frame').groupby('id').last().reset_index()

    # Union-Find Setup
    id_map = {id: id for id in df_clean['id'].unique()}
    def get_parent(i):
        if id_map[i] != i: id_map[i] = get_parent(id_map[i])
        return id_map[i]
    def union(i, j):
        root_i, root_j = get_parent(i), get_parent(j)
        if root_i != root_j:
            id_map[root_j] = root_i
            return True
        return False

    ends_list = df_ends.to_dict('records')
    starts_list = df_starts.to_dict('records')
    valid_merges = []
    frame_cache = {}

    # 3. Iterate potential pairs
    for track_a in tqdm(ends_list, desc="    Checking Pairs", leave=False):
        max_speed_a = track_max_speeds.get(track_a['id'], global_max)
        if max_speed_a < 1.0: max_speed_a = 1.0
        burst_potential = max_speed_a * 1.5

        for track_b in starts_list:
            time_gap = track_b['frame'] - track_a['frame']
            if time_gap <= 0: continue # Only look forward in time

            possible_travel = burst_potential * time_gap
            
            # Simple Euclidean Check first
            raw_dist = np.sqrt((track_a['xmin']-track_b['xmin'])**2 + (track_a['ymin']-track_b['ymin'])**2)
            if raw_dist > possible_travel: continue

            if get_parent(track_a['id']) == get_parent(track_b['id']): continue

            # Load Frames
            if track_a['frame'] not in frame_cache:
                frame_cache[track_a['frame']] = loader.get_frame(track_a['frame'])
            if track_b['frame'] not in frame_cache:
                frame_cache[track_b['frame']] = loader.get_frame(track_b['frame'])
            img_a, img_b = frame_cache[track_a['frame']], frame_cache[track_b['frame']]
            if img_a is None or img_b is None: continue

            # Homography Check
            center_a = ((track_a['xmin']+track_a['xmax'])/2, (track_a['ymin']+track_a['ymax'])/2)
            center_b = ((track_b['xmin']+track_b['xmax'])/2, (track_b['ymin']+track_b['ymax'])/2)
            consistent, _ = solve_homography_and_check(img_a, img_b, center_a, center_b, possible_travel)

            if consistent:
                # Visual Check
                bbox_a = [track_a['xmin'], track_a['ymin'], track_a['xmax'], track_a['ymax']]
                bbox_b = [track_b['xmin'], track_b['ymin'], track_b['xmax'], track_b['ymax']]
                vis_score = get_crop_similarity(img_a, bbox_a, img_b, bbox_b)

                if vis_score > 0.4:
                    valid_merges.append({'id_a': track_a['id'], 'id_b': track_b['id'], 'score': vis_score})

    loader.release()

    # 4. Apply Merges
    valid_merges.sort(key=lambda x: -x['score'])
    count = 0
    for m in valid_merges:
        if union(m['id_a'], m['id_b']):
            count += 1

    print(f"    🔗 Merged {count} track pairs.")
    df_clean['id'] = df_clean['id'].apply(lambda x: get_parent(x))
    return df_clean

# --- MAIN TRACKING FUNCTIONS ---

def get_compute_device():
    """Detects available compute device: cuda, mps, or cpu."""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

def process_single_video(video_path, output_path, model, conf_thresh, device):
    if not os.path.exists(video_path):
        print(f"[Error] Video file not found: {video_path}")
        return False

    print(f"--> Processing: {video_path}")

    # Tracker Config
    tracker_config = 'CustomeBoTSORT.yaml' 
    if os.path.exists(tracker_config):
        tracker_arg = tracker_config
    else:
        tracker_arg = 'botsort.yaml'

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Error] Could not open video stream.")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detection_data = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = model.track(
            frame, persist=True, conf=conf_thresh, verbose=False, 
            tracker=tracker_arg, device=device 
        )

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            if boxes.id is not None:
                # Get coords
                coords = boxes.xywh.cpu().numpy()
                track_ids = boxes.id.int().cpu().tolist()
                
                for (x, y, w, h), track_id in zip(coords, track_ids):
                    # Save xywh AND xmin/ymin/xmax/ymax for Re-ID logic
                    detection_data.append({
                        'frame': frame_idx,
                        'id': track_id,
                        'x': float(x),
                        'y': float(y),
                        'w': float(w),
                        'h': float(h),
                        'xmin': float(x - w/2),
                        'ymin': float(y - h/2),
                        'xmax': float(x + w/2),
                        'ymax': float(y + h/2)
                    })

        if frame_idx % 50 == 0:
            sys.stdout.write(f"\r    Frame: {frame_idx}/{total_frames}")
            sys.stdout.flush()
            
        frame_idx += 1

    cap.release()
    print(f"\n    Tracking finished. Processing data...")

    # SAVE LOGIC
    if detection_data:
        df = pd.DataFrame(detection_data)
        
        # --- RUN RE-ID MODULE HERE ---
        try:
            df = process_reid(df, video_path)
        except Exception as e:
            print(f"Warning: Re-ID module failed ({e}). Saving raw tracking instead.")
        
        # Sort and Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Clean up extra columns (xmin, ymin...) before saving if you only want xywh
        # But keeping them might be useful. Let's keep strict format frame,id,x,y,w,h for downstream
        final_df = df[['frame', 'id', 'x', 'y', 'w', 'h']].sort_values(by=['frame', 'id'])
        
        final_df.to_csv(output_path, index=False)
        print(f"    Saved CLEAN tracking data to: {output_path}")
        return True
    else:
        print("    [Warning] No fish detected. No CSV created.")
        return False

def process_dual_videos(vid1_path, vid2_path, out1_path, out2_path, model_path, conf=0.25):
    device = get_compute_device()
    print(f"Compute Device Selected: {device.upper()}")

    print(f"Loading YOLO model: {model_path} ...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Critical Error loading model: {e}")
        return

    print("Model loaded. Starting batch processing...")
    print("="*60)
    print("CAMERA 1 SEQUENCE")
    process_single_video(vid1_path, out1_path, model, conf, device)
    print("-" * 40)
    print("CAMERA 2 SEQUENCE")
    process_single_video(vid2_path, out2_path, model, conf, device)
    print("="*60)
    print("Done! Both videos processed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vid1", required=True)
    parser.add_argument("--out1", required=True)
    parser.add_argument("--vid2", required=True)
    parser.add_argument("--out2", required=True)
    parser.add_argument("--model", default="yolov8n.pt")
    args = parser.parse_args()
    
    process_dual_videos(args.vid1, args.vid2, args.out1, args.out2, args.model)