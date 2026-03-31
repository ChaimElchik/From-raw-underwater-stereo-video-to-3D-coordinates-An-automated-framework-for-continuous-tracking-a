import os
import subprocess

videos = [
    42, 43, 48, 54, 58, 61, 62, 116, 122, 134, 143, 168, 175, 183, 186, 187, 189, 
    191, 194, 198, 202, 204, 206, 214, 223, 236, 244, 268, 278, 280, 283, 289, 
    296, 301, 302, 303, 308, 312, 437, 328, 329, 330, 335, 336, 340, 341, 342, 
    343, 344, 347, 376, 378, 384, 406, 432
]

VIDS_DIR = "../vids"
MOTS_DIR = "../mots"

def main():
    success_count = 0
    failure_count = 0
    
    for vid in videos:
        # Assuming videos follow the {vid}_{cam} naming convention (e.g., 42_1 and 42_2)
        for cam in [1, 2]:
            vid_name = f"{vid}_{cam}"
            vid_path = os.path.join(VIDS_DIR, f"{vid_name}.mp4")
            gt_path = os.path.join(MOTS_DIR, f"{vid_name}_clean.txt")
            
            # Check if both video and ground truth exist before trying to run
            if os.path.exists(vid_path) and os.path.exists(gt_path):
                print(f"\n{'='*50}")
                print(f"Processing Video: {vid_name}")
                print(f"{'='*50}\n")
                
                # 1. Run tracking (Test_Re_ID_Adcanved.py)
                print(f"--> Running Test_Re_ID_Adcanved.py for {vid_name}...")
                try:
                    subprocess.run(
                        ["python", "Test_Re_ID_Adcanved.py", "--video", vid_path],
                        check=True
                    )
                except subprocess.CalledProcessError:
                    print(f"❌ Error running tracking for {vid_name}")
                    failure_count += 1
                    continue
                    
                # 2. Run merging evaluation (Check-Track-Merges.py)
                print(f"\n--> Running Check-Track-Merges.py for {vid_name}...")
                try:
                    subprocess.run(
                        ["python", "Check-Track-Merges.py", "--video", vid_name],
                        check=True
                    )
                    success_count += 1
                except subprocess.CalledProcessError:
                    print(f"❌ Error running tracking evaluation for {vid_name}")
                    failure_count += 1
                    continue
            else:
                pass # Silently skip missing files if you prefer, or handle missing _2 files nicely

    print("\n" + "="*50)
    print("BATCH EVALUATION COMPLETE")
    print(f"Successfully processed: {success_count} videos")
    print(f"Failed to process:      {failure_count} videos")
    print("="*50)

if __name__ == "__main__":
    main()
