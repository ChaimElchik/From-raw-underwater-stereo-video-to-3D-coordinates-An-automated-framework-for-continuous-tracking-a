import os
import argparse
from pathlib import Path

# --- Import Custom Modules ---
# Ensure these .py files are in the same directory or PYTHONPATH
try:
    import ProcessVideoPair as tracking_module       # Step 1: Detection & Tracking
    import StereoMatching as matching_module                  # Step 2: Refractive Stereo Matching
    import ThreeDCordinate_Maker as triangulation_module # Step 3: 3D Triangulation
    import OutPutVideoGenerater as viz_module        # Step 4: Visualization
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import one of the required scripts. \n{e}")
    print("Ensure ProcessVideoPair.py, StereoMatching.py, ThreeDCordinate_Maker.py, "
          "and OutPutVideoGenerater.py are in the folder.")
    exit(1)

def run_pipeline(vid1_path, vid2_path, calibration_mat, model_path, output_root, 
                 correct_refraction=False, d_air=0.0, d_glass=0.0):
    """
    Executes the comprehensive 3D fish tracking framework.
    
    Workflow:
    1. Detection & BoT-SORT Tracking (ProcessVideoPair.py)
    2. Refractive Epipolar Matching (StereoMatching.py)
    3. 3D Triangulation (ThreeDCordinate_Maker.py)
    4. Visualization Overlay (OutPutVideoGenerater.py)
    """
    
    # --- Setup Directories ---
    root = Path(output_root)
    mots_dir = root / "mots"          # For intermediate CSVs
    video_out_dir = root / "videos"   # For final overlay videos
    
    for d in [mots_dir, video_out_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"=== Starting 3D Fish Tracking Pipeline ===")
    print(f"Output Directory: {root}")

    # Define File Paths
    raw_track_1 = mots_dir / "cam1_raw.csv"
    raw_track_2 = mots_dir / "cam2_raw.csv"
    
    mapped_track_1 = mots_dir / "cam1_mapped.csv"
    mapped_track_2 = mots_dir / "cam2_mapped.csv"
    
    trajectory_3d = mots_dir / "3d_trajectory.csv"

    # -------------------------------------------------------------------------
    # STEP 1: Detection & Temporal Linking (BoT-SORT)
    # -------------------------------------------------------------------------
    print("\n[Step 1/5] Running Modular Detection & BoT-SORT Tracking...")
    try:
        # Assuming ProcessVideoPair has a main processing function taking (vid1, vid2, out1, out2, model)
        # Note: Adjust function name 'process_dual_videos' if your script names it differently
        tracking_module.process_dual_videos(
            str(vid1_path), str(vid2_path),
            str(raw_track_1), str(raw_track_2),
            model_path=model_path,
            conf=0.3
        )
        print("✓ Tracking complete.")
    except Exception as e:
        print(f"❌ Error in Step 1: {e}")
        return

    # -------------------------------------------------------------------------
    # STEP 2: Epipolar Stereo Matching (Refractive Correction)
    # -------------------------------------------------------------------------
    print("\n[Step 2/5] Running Refractive Epipolar Matching...")
    try:
        # 1. Compute Geometric Matches using Refractive Matrix
        mapping_df, _ = matching_module.run_geometric_matching(
            str(raw_track_1), 
            str(raw_track_2), 
            str(calibration_mat),
            n_water=1.333,
            correct_refraction=correct_refraction,
            d_air=d_air,
            d_glass=d_glass
        )
        
        # 2. Apply Mapping and Renaming
        if not mapping_df.empty:
            matching_module.save_remapped_tracking(
                mapping_df,
                str(raw_track_1), str(raw_track_2),
                str(mapped_track_1), str(mapped_track_2)
            )
            print("✓ Matching complete. IDs synchronized.")
        else:
            print("❌ Error: No matches found between cameras. Aborting.")
            return
    except Exception as e:
        print(f"❌ Error in Step 2: {e}")
        return

    # -------------------------------------------------------------------------
    # STEP 3: Metric 3D Triangulation
    # -------------------------------------------------------------------------
    print("\n[Step 3/5] Triangulating 3D Coordinates...")
    try:
        df_3d = triangulation_module.cor_maker_3d(
            str(mapped_track_1), 
            str(mapped_track_2), 
            str(calibration_mat),
            n_water=1.333,
            correct_refraction=correct_refraction,
            d_air=d_air,
            d_glass=d_glass
        )
        
        if not df_3d.empty:
            df_3d.to_csv(trajectory_3d, index=False)
            print(f"✓ 3D Trajectory saved to: {trajectory_3d}")
        else:
            print("❌ Error: 3D reconstruction yielded empty results.")
            return
    except Exception as e:
        print(f"❌ Error in Step 3: {e}")
        return

    # -------------------------------------------------------------------------
    # STEP 4: Visualization
    # -------------------------------------------------------------------------
    print("\n[Step 4/4] Generating Overlay Videos...")
    try:
        # Camera 1 Overlay
        viz_module.create_annotated_video(
            str(vid1_path), str(mapped_track_1), 
            str(video_out_dir / "cam1_tracked.mp4")
        )
        # Camera 2 Overlay
        viz_module.create_annotated_video(
            str(vid2_path), str(mapped_track_2), 
            str(video_out_dir / "cam2_tracked.mp4")
        )
        print("✓ Videos generated.")
    except Exception as e:
        print(f"❌ Error in Step 5: {e}")

    print("\n=== Pipeline Finished Successfully ===")
    print(f"Results available at: {output_root}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="High-Precision 3D Fish Tracking Framework")
    
    # Inputs
    parser.add_argument("--vid1", required=True, help="Path to Left Camera Video")
    parser.add_argument("--vid2", required=True, help="Path to Right Camera Video")
    parser.add_argument("--calib", required=True, help="Path to Stereo Params .mat file")
    parser.add_argument("--model", default="yolov8n.pt", help="Path to YOLO weights")
    
    # Output
    
    # Refraction Options
    parser.add_argument("--correct_refraction", action="store_true", help="Enable refraction correction (if system calibrated in air)")
    parser.add_argument("--d_air", type=float, default=0.0, help="Distance from camera origin to glass (mm)")
    parser.add_argument("--d_glass", type=float, default=0.0, help="Thickness of the glass port (mm)")

    args = parser.parse_args()
    
    # Run
    run_pipeline(
        args.vid1, args.vid2, args.calib, args.model, args.output,
        correct_refraction=args.correct_refraction,
        d_air=args.d_air,
        d_glass=args.d_glass
    )