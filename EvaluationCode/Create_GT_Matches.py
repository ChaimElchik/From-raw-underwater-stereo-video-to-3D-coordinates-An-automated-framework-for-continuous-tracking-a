import os
import glob
import json
import xml.etree.ElementTree as ET
import pandas as pd
import shutil

# --- Constants & Paths ---
BASE_DIR = "/Users/chaim/Documents/Thesis_ReWrite"
XML_PATH = os.path.join(BASE_DIR, "loom_video_annotations_sara 2.xml")
MATCHED_IDS_PATH = os.path.join(BASE_DIR, "Matched Fish_IDs.txt")
MOTS_DIR = os.path.join(BASE_DIR, "mots")
OUTPUT_DIR = os.path.join(BASE_DIR, "Final_Pipeline_Code", "Ground_Truth_Matches")

def parse_matched_fish_ids(path):
    """
    Parses the matched IDs text file and creates a structured mapping per video
    """
    print("Parsing Matched Fish_IDs.txt...")
    matches_per_video = {}
    
    with open(path, "r") as f:
        # Skip header
        header = f.readline()
        
        for line in f:
            if not line.strip(): continue
            parts = line.strip().split('\t')
            if len(parts) < 4: continue
            
            new_id = parts[0]
            cam = parts[1]
            cvat_id = parts[2]
            video = parts[3]
            
            if video not in matches_per_video:
                matches_per_video[video] = {}
            if new_id not in matches_per_video[video]:
                matches_per_video[video][new_id] = {"LC_ID": None, "RC_ID": None}
                
            matches_per_video[video][new_id][cam] = cvat_id
    
    # Filter out entries that don't have BOTH an LC_ID and RC_ID
    clean_matches = {}
    for video, pairs in matches_per_video.items():
        valid_pairs = []
        for pair_id, cams in pairs.items():
            if cams["LC_ID"] is not None and cams["RC_ID"] is not None:
                valid_pairs.append({
                    "pair_id": pair_id,
                    "LC_cvatID": cams["LC_ID"],
                    "RC_cvatID": cams["RC_ID"]
                })
        
        if len(valid_pairs) > 0:
            clean_matches[video] = valid_pairs
            
    return clean_matches

def load_cvat_xml(path):
    print(f"Loading CVAT XML at {path}...")
    tree = ET.parse(path)
    return tree.getroot()

def get_task_mapping_from_xml(cvat_root, target_videos):
    """
    Finds the mapping between pure video numbers (e.g., '42') and their internal XML task parameters.
    """
    print("Mapping video names to XML task IDs...")
    task_mapping = {}
    
    for task in cvat_root.findall(".//task"):
        task_id = task.find("id").text
        name = task.find("name").text
        
        # e.g., '42_cam12_stacked_short_20_240.mp4' -> '42'
        video_num = name.split('_')[0]
        
        if video_num in target_videos:
            task_mapping[video_num] = task_id
            
    return task_mapping

def find_track_box_signature(cvat_root, task_id, track_id):
    """
    Search the XML tree for a given task ID and track ID, and return the first valid box coordinate.
    """
    track_nodes = cvat_root.findall(f".//track[@task_id='{task_id}'][@id='{track_id}']")
    if not track_nodes:
        return None
    
    track_node = track_nodes[0]
    first_box = track_node.find("box")
    if first_box is None:
        return None
        
    frame = int(first_box.get("frame"))
    xtl = float(first_box.get("xtl"))
    ytl = float(first_box.get("ytl"))
    xbr = float(first_box.get("xbr"))
    ybr = float(first_box.get("ybr"))
    
    width = xbr - xtl
    height = ybr - ytl
    
    return {
        "frame": (frame % 260) + 1,  # CVAT strings sequence frames. MOT holds relative 1-[260] chunks.
        "xtl": xtl,
        "ytl": ytl,
        "width": width,
        "height": height
    }

def find_mot_id_by_signature(mot_file_path, signature, cam_idx):
    """
    Reads a MOT file and searches for a row matching the signature frame, and coordinates.
    Since precision might vary slightly, we use a reasonable epsilon.
    """
    if not os.path.exists(mot_file_path):
        print(f"Warning: Missing MOT file at {mot_file_path}")
        return None
        
    # Read MOT file: frame, id, x, y, width, height, conf, class, visibility
    try:
        df = pd.read_csv(mot_file_path, header=None, names=["frame", "id", "x", "y", "w", "h", "conf", "cls", "vis"])
    except Exception as e:
        print(f"Error reading {mot_file_path}: {e}")
        return None
        
    mot_frame = signature["frame"]
    mot_x = signature["xtl"] if cam_idx == 1 else signature["xtl"] - 1920.0 # Adjust RC offset
    mot_y = signature["ytl"]
    
    # Filter by frame
    frame_matches = df[df["frame"] == mot_frame]
    
    if frame_matches.empty:
        return None
        
    # Find closest match by bounding box proximity
    # Allow some epsilon distance because values might be slightly rounded
    best_match_id = None
    min_dist = float('inf')
    
    for _, row in frame_matches.iterrows():
        dist = abs(row["x"] - mot_x) + abs(row["y"] - mot_y)
        if dist < min_dist:
            min_dist = dist
            best_match_id = int(row["id"])
            
    # If the closest coordinate is within 5 pixels, accept it
    if min_dist < 5.0 and best_match_id is not None:
        return best_match_id
        
    return None

def process_video_matches(video, pairs, cvat_root, task_id):
    print(f"Processing Ground Truth Matches for Video {video} ... (Total Pairs: {len(pairs)})")
    
    mot_left_path = os.path.join(MOTS_DIR, f"{video}_1.txt")
    mot_right_path = os.path.join(MOTS_DIR, f"{video}_2.txt")
    
    final_matches = []
    
    for pair in pairs:
        cvat_lc = pair["LC_cvatID"]
        cvat_rc = pair["RC_cvatID"]
        
        # 1. Look up the CVAT box signature
        sig_lc = find_track_box_signature(cvat_root, task_id, cvat_lc)
        sig_rc = find_track_box_signature(cvat_root, task_id, cvat_rc)
        
        if not sig_lc or not sig_rc:
            print(f"  [Warning] Missing XML track info for {video} pairs ({cvat_lc}, {cvat_rc})")
            continue
            
        # 2. Look up the corresponding ID in the MOT text outputs
        mot_lc_idx = find_mot_id_by_signature(mot_left_path, sig_lc, cam_idx=1)
        mot_rc_idx = find_mot_id_by_signature(mot_right_path, sig_rc, cam_idx=2)
        
        if mot_lc_idx is not None and mot_rc_idx is not None:
            # We found a definitive link!
            final_matches.append({
                "pair_id": pair["pair_id"],         # The reference pair integer (e.g. Fish 1)
                "cam_1_id": mot_lc_idx,             # MOT trajectory ID in *_1.txt
                "cam_2_id": mot_rc_idx,             # MOT trajectory ID in *_2.txt
                "cvat_lc_track": cvat_lc,
                "cvat_rc_track": cvat_rc
            })
            
    return final_matches

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # 1. Parse Matched File
    matches = parse_matched_fish_ids(MATCHED_IDS_PATH)
    target_videos = list(matches.keys())
    print(f"Target videos to process matches for: {target_videos}")
    
    # 2. Load CVAT XML
    cvat_root = load_cvat_xml(XML_PATH)
    
    # 3. Associate Target videos with Task IDs
    task_map = get_task_mapping_from_xml(cvat_root, target_videos)
    
    global_stats = {"total_videos": 0, "total_pairs_extracted": 0}
    
    # 4. Generate the Ground Truth
    for video, pairs in matches.items():
        if video not in task_map:
            print(f"Skipping {video}, could not resolve Task ID inside CVAT XML.")
            continue
            
        task_id = task_map[video]
        final_video_matches = process_video_matches(video, pairs, cvat_root, task_id)
        
        if len(final_video_matches) > 0:
            # A dictionary mapping cam 1 ID -> cam 2 ID directly for evaluation ease
            simplified_mapping_dict = {
                str(m["cam_1_id"]): int(m["cam_2_id"]) for m in final_video_matches
            }
            
            output_data = {
                "video": video,
                "task_id": task_id,
                "total_pairs": len(final_video_matches),
                "pairs": final_video_matches,
                "mapping_dict": simplified_mapping_dict
            }
            
            # Save the JSON manifest
            out_json = os.path.join(OUTPUT_DIR, f"{video}_gt_matches.json")
            with open(out_json, "w") as f:
                json.dump(output_data, f, indent=4)
                
            # Filter MOT files to exclusively contain confirmed GT pairs
            valid_cam1 = set([m["cam_1_id"] for m in final_video_matches])
            valid_cam2 = set([m["cam_2_id"] for m in final_video_matches])
            
            for src_suffix, valid_ids in [("1.txt", valid_cam1), ("2.txt", valid_cam2)]:
                src_path = os.path.join(MOTS_DIR, f"{video}_{src_suffix}")
                dst_path = os.path.join(OUTPUT_DIR, f"{video}_{src_suffix}")
                if os.path.exists(src_path):
                    df_mot = pd.read_csv(src_path, header=None, names=["frame", "id", "x", "y", "w", "h", "conf", "cls", "vis"])
                    df_filtered = df_mot[df_mot["id"].isin(valid_ids)]
                    df_filtered.to_csv(dst_path, header=False, index=False)
            
            global_stats["total_videos"] += 1
            global_stats["total_pairs_extracted"] += len(final_video_matches)
        else:
            print(f"Failed to find matched MOT indexes for {video}.")
            
    print("\n============ DONE ============")
    print(f"Extracted Ground Truth pairings for {global_stats['total_videos']} videos.")
    print(f"Total matched tracked IDs mapped: {global_stats['total_pairs_extracted']}")
    print(f"Data saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
