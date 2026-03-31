import cv2
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def generate_color(id_num):
    """Generates a consistent, unique color for each ID (BGR format)."""
    np.random.seed(int(id_num))
    return tuple(np.random.randint(0, 255, 3).tolist())

def draw_overlay(frame, df_frame):
    """Draws tracking info on a single frame."""
    for _, row in df_frame.iterrows():
        obj_id = int(row['id'])
        
        # --- COORDINATE FIX ---
        # Input (x, y) are CENTER coordinates from YOLO
        center_x = float(row['x'])
        center_y = float(row['y'])
        
        color = generate_color(obj_id)
        
        # Check if we have width/height for a bounding box
        if 'w' in row and 'h' in row:
            w = float(row['w'])
            h = float(row['h'])
            
            # Convert Center -> Top-Left for OpenCV
            x1 = int(center_x - (w / 2))
            y1 = int(center_y - (h / 2))
            x2 = int(center_x + (w / 2))
            y2 = int(center_y + (h / 2))
            
            # Draw Rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Label Position (Top-Left of the box)
            label_pos_x = x1
            label_pos_y = y1 - 10
        else:
            # Fallback: Draw a circle at the center if no W/H columns exist
            cx, cy = int(center_x), int(center_y)
            cv2.circle(frame, (cx, cy), 5, color, -1)
            cv2.circle(frame, (cx, cy), 10, color, 1) # Ring
            label_pos_x = cx + 10
            label_pos_y = cy - 5

        # Draw ID Label with background for readability
        label = f"ID: {obj_id}"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Ensure label doesn't go off-screen
        label_pos_y = max(label_pos_y, text_h + 5)
        
        cv2.rectangle(frame, 
                      (label_pos_x, label_pos_y - text_h - baseline), 
                      (label_pos_x + text_w, label_pos_y + baseline), 
                      color, -1)
        
        cv2.putText(frame, label, (label_pos_x, label_pos_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame

def create_annotated_video(video_path, csv_path, output_path):
    """
    Reads video and CSV, draws overlays, and saves result.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    # Load Data
    print(f"Loading tracking data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Open Video
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare Output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 'mp4v' for .mp4, 'XVID' for .avi
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing video: {video_path} -> {output_path}")
    
    frame_idx = 0
    with tqdm(total=total_frames) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get data for current frame
            if 'frame' in df.columns:
                df_current = df[df['frame'] == frame_idx]
                frame = draw_overlay(frame, df_current)
            
            out.write(frame)
            frame_idx += 1
            pbar.update(1)

    cap.release()
    out.release()
    print("Done.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--csv", required=True, help="Path to tracking CSV")
    parser.add_argument("--output", required=True, help="Path to output video")
    
    args = parser.parse_args()
    
    create_annotated_video(args.video, args.csv, args.output)