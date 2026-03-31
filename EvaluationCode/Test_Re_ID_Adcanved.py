import os
import argparse
import pandas as pd
from ultralytics import YOLO

# Import functions from your newly refactored modules
from ProcessVideoPair_Refactored import process_single_video, get_compute_device
from AdvancedReID import process_reid_pipeline
from OutPutVideoGenerater import create_annotated_video

def run_visual_test(video_path, model_path, tracker_config, out_dir, limit=None):
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Setup paths
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    default_botsort_csv = os.path.join(out_dir, f"{base_name}_default_botsort_tracks.csv")
    raw_csv = os.path.join(out_dir, f"{base_name}_raw_tracks.csv")
    raw_vid = os.path.join(out_dir, f"{base_name}_raw_visual.mp4")
    reid_csv = os.path.join(out_dir, f"{base_name}_reid_tracks.csv")
    reid_vid = os.path.join(out_dir, f"{base_name}_reid_visual.mp4")

    device = get_compute_device()
    print(f"\n[INIT] Using Compute Device: {device.upper()}")
    print(f"[INIT] Loading YOLO Model: {model_path}")
    model = YOLO(model_path)

    # ==========================================
    # STEP 0: Default BoT-SORT Tracking
    # ==========================================
    print("\n" + "="*50)
    print("STEP 0: Running Default BoT-SORT Tracking")
    print("="*50)
    
    process_single_video(
        video_path=video_path,
        output_path=default_botsort_csv,
        model=model,
        conf_thresh=0.3,
        device=device,
        tracker_config='botsort.yaml',
        use_reid=False, 
        limit=limit
    )

    # ==========================================
    # STEP 1: Baseline Tracking (No Re-ID)
    # ==========================================
    print("\n" + "="*50)
    print("STEP 1: Running Baseline BoT-SORT Tracking")
    print("="*50)
    
    # We pass use_reid=False so we can capture the raw fragmented tracks first
    df_raw, _, _ = process_single_video(
        video_path=video_path,
        output_path=raw_csv,
        model=model,
        conf_thresh=0.3,
        device=device,
        tracker_config=tracker_config,
        use_reid=False, 
        limit=limit
    )

    # ==========================================
    # STEP 2: Generate Baseline Video
    # ==========================================
    print("\n" + "="*50)
    print("STEP 2: Generating Baseline Visual Video")
    print("="*50)
    create_annotated_video(video_path, raw_csv, raw_vid)

    # ==========================================
    # STEP 3: Apply Advanced Re-ID
    # ==========================================
    print("\n" + "="*50)
    print("STEP 3: Running MegaDescriptor Re-ID Pipeline")
    print("="*50)
    
    # Pass the raw DataFrame and video through the Re-ID module
    df_reid = process_reid_pipeline(df_raw, video_path)
    
    # Save the purified DataFrame
    df_reid.to_csv(reid_csv, index=False)
    print(f"    Saved Refined tracking data to: {reid_csv}")

    # ==========================================
    # STEP 4: Generate Refined Video
    # ==========================================
    print("\n" + "="*50)
    print("STEP 4: Generating Refined Visual Video")
    print("="*50)
    create_annotated_video(video_path, reid_csv, reid_vid)

    # ==========================================
    # Summary
    # ==========================================
    print("\n" + "="*50)
    print("🎉 VISUAL TEST COMPLETE")
    print("="*50)
    print(f"Outputs generated in the '{out_dir}' directory:")
    print(f"  - Default BoT-SORT CSV:  {default_botsort_csv}")
    print(f"  - Custom Baseline Video: {raw_vid}")
    print(f"  - Custom Re-ID Video:    {reid_vid}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test and visualize Re-ID track merging.")
    parser.add_argument("--video", required=True, help="Path to the test video")
    parser.add_argument("--out_dir", default="ReID_Test_Results", help="Directory to save outputs")
    parser.add_argument("--model", default="ModelWeights/det_best_bgr29.pt", help="Path to YOLO weights")
    parser.add_argument("--tracker", default="CustomeBoTSORT.yaml", help="Path to tracker YAML config")
    parser.add_argument("--limit", type=int, default=None, help="Limit frames for a quick test (e.g., 500)")
    
    args = parser.parse_args()
    
    run_visual_test(
        video_path=args.video,
        model_path=args.model,
        tracker_config=args.tracker,
        out_dir=args.out_dir,
        limit=args.limit
    )