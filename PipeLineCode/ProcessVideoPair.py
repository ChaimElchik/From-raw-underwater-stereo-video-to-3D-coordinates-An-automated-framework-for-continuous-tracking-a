import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import os
import sys
import torch
from tqdm import tqdm

# Import the superior Re-ID pipeline from your AdvancedReID script
from AdvancedReIDV2 import process_reid_pipeline

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

    # Force tracker reset by using a completely fresh YOLO model per video (handled in process_dual_videos)

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
        
        # --- RUN ADVANCED RE-ID MODULE HERE ---
        try:
            print("\n    🧠 Initializing Advanced ViT Re-ID...")
            df = process_reid_pipeline(df, video_path)
        except Exception as e:
            print(f"Warning: Advanced Re-ID module failed ({e}). Saving raw tracking instead.")
        
        # Sort and Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Clean up extra columns before saving strict format
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
    try:
        model1 = YOLO(model_path)
    except Exception as e:
        print(f"Critical Error loading model: {e}")
        return
    process_single_video(vid1_path, out1_path, model1, conf, device)
    
    print("-" * 40)
    print("CAMERA 2 SEQUENCE")
    # Fresh model initialization effectively resets the tracker's internal cache
    try:
        model2 = YOLO(model_path)
    except Exception as e:
        print(f"Critical Error loading model: {e}")
        return
    process_single_video(vid2_path, out2_path, model2, conf, device)
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