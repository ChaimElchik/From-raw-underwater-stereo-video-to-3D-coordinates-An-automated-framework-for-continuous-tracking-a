import os
import subprocess
import sys
from pathlib import Path

# --- Configuration ---
VIDEO_IDS = ['129'] # Focused test on Video 129
VIDEO_DIR = Path("/Users/chaim/Documents/Thesis_ReWrite/vids")
CODE_DIR = Path("/Users/chaim/Documents/Thesis_ReWrite/Final_Pipeline_Code")
OUTPUT_ROOT = Path("/Users/chaim/Documents/Thesis_ReWrite/eval_results")

# Model Path (Adjust if needed, assuming standard yolov8n or a custom one if found)
# Attempting to locate a custom model, otherwise default
weights_dir = CODE_DIR / "ModelWeights"
MODEL_PATH = str(weights_dir / "det_best_bgr29.pt")
if weights_dir.exists():
    # Try to find a .pt file
    pt_files = list(weights_dir.glob("*.pt"))
    if pt_files:
        MODEL_PATH = str(pt_files[0])
        print(f"Fnord: Using found model: {MODEL_PATH}")

# Tracker Configs
CUSTOM_BOTSORT = CODE_DIR / "CustomeBoTSORT.yaml"
DEFAULT_BOTSORT = CODE_DIR / "botsort.yaml" # Assuming this exists or is the default for YOLO

# If default yaml is not there, we might rely on YOLO's internal default, 
# but the script expects a path. Validating availability:
if not CUSTOM_BOTSORT.exists():
    print(f"WARNING: {CUSTOM_BOTSORT} not found.")

modes = [
    # {
    #     "name": "mode_1_default_noreid",
    #     "tracker": "botsort.yaml",
    #     "reid": False,
    #     "desc": "Default BotSORT, No Re-ID"
    # },
    # {
    #     "name": "mode_2_custom_noreid",
    #     "tracker": str(CUSTOM_BOTSORT),
    #     "reid": False,
    #     "desc": "Custom BotSORT, No Re-ID"
    # },
    # {
    #     "name": "mode_3_custom_reid",
    #     "tracker": str(CUSTOM_BOTSORT),
    #     "reid": True,
    #     "desc": "Custom BotSORT, Re-ID ON"
    # },
    {
        "name": "mode_5_advanced_reid",
        "tracker": str(CUSTOM_BOTSORT),
        "reid": False, # We run tracking WITHOUT internal ReID, then post-process
        "desc": "Advanced Re-ID (ViT + Trajectory)",
        "script": "ProcessVideoPair_Refactored.py", # Base tracking
        "post_process": True # Flag to run AdvancedReID.py
    }
]

def run_command(cmd):
    print(f"Running: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        print(f"Error executing command: {cmd}")
        return False
    return True

def generate_videos(vid_id, mode_name, track_csv, vid_path):
    output_vid = OUTPUT_ROOT / mode_name / vid_id / f"{vid_path.stem}_tracked.mp4"
    print(f"Generating video: {output_vid}")
    
    # Using OutPutVideoGenerater.py as a script or module
    # Since it has a name guard, we can run it via python command
    cmd = f"{sys.executable} {str(CODE_DIR / 'OutPutVideoGenerater.py')} --video {vid_path} --csv {track_csv} --output {output_vid}"
    
    # Wait, checking OutPutVideoGenerater.py args... it uses a hardcoded main block example.
    # I should import and call it if possible, or modify it to accept args.
    # Current file inspection showed it lacks arg parsing. I will call it as module here.
    
    try:
        from OutPutVideoGenerater import create_annotated_video
        create_annotated_video(str(vid_path), str(track_csv), str(output_vid))
        return True
    except ImportError:
        print("Failed to import Video Generator.")
        return False
    except Exception as e:
        print(f"Video Generation Error: {e}")
        return False

def main():
    print("=== Starting Tracking Evaluation Workflow ===")
    
    for mode in modes:
        print(f"\n--- Starting Mode: {mode['name']} ({mode['desc']}) ---")
        
        for vid_id in VIDEO_IDS:
            print(f"Processing Video ID: {vid_id}...")
            
            # Define Paths
            vid1 = VIDEO_DIR / f"{vid_id}_1.mp4"
            vid2 = VIDEO_DIR / f"{vid_id}_2.mp4"
            
            if not vid1.exists() or not vid2.exists():
                print(f"Skipping {vid_id}: Video files not found.")
                continue
                
            out_dir = OUTPUT_ROOT / mode['name'] / vid_id
            out1 = out_dir / "cam1_tracked.csv"
            out2 = out_dir / "cam2_tracked.csv"
            
            # Skip if already done? (Optional, but good for resuming)
            # if out1.exists() and out2.exists():
            #     print(f"Skipping {vid_id}: Results already exist.")
            #     continue
            
            # Construct Command
            script_name = mode.get("script", "ProcessVideoPair_Refactored.py")
            cmd = [
                sys.executable,
                str(CODE_DIR / script_name),
                f"--vid1 {vid1}",
                f"--vid2 {vid2}",
                f"--out1 {out1}",
                f"--out2 {out2}",
                f"--model {MODEL_PATH}",
                f"--tracker_config {mode['tracker']}"
            ]
            
            if mode['reid']:
                cmd.append("--use_reid")
                
            full_cmd = " ".join(cmd)
            
            if not run_command(full_cmd):
                print(f"Failed to process {vid_id} in {mode['name']}")
                continue
                
            # --- Post-Processing (Advanced Re-ID) ---
            if mode.get("post_process"):
                print(f"Running Advanced Re-ID Post-Processing...")
                
                # We need to run it for both camera outputs
                for cam_num, csv_in in [('1', out1), ('2', out2)]:
                    csv_final = out_dir / f"cam{cam_num}_tracked_advanced.csv"
                    log_file = out_dir / f"cam{cam_num}_merge_log.txt"
                    vid_src = vid1 if cam_num == '1' else vid2
                    
                    cmd_reid = [
                        sys.executable,
                        str(CODE_DIR / "AdvancedReID.py"),
                        f"--csv_in {csv_in}",
                        f"--vid {vid_src}",
                        f"--csv_out {csv_final}",
                        f"--log_out {log_file}"
                    ]
                    
                    if run_command(" ".join(cmd_reid)):
                        # Overwrite the 'out' variable so video generation uses the new file
                        if cam_num == '1': out1 = csv_final
                        else: out2 = csv_final
            
            # --- Video Generation ---
            # Generate videos for visual verification
            generate_videos(vid_id, mode['name'], out1, vid1)
            generate_videos(vid_id, mode['name'], out2, vid2)
                
    print("\n=== Workflow Completed ===")

if __name__ == "__main__":
    main()
