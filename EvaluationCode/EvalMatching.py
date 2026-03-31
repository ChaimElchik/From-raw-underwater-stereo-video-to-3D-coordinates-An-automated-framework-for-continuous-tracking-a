import pandas as pd
import os
import sys

# Import your matching scripts
# Ensure strV2.py, strV3.py, strV4.py are in the same directory as this script
import strV5
import strV3
import strV4

def compare_stereo_matching(video_names, ground_truth_dict, mat_file_path='stereoParams_Dep4.mat', mots_folder='mots'):
    """
    Runs 3 versions of stereo matching and compares against a ground truth dictionary.
    
    Args:
        video_names (list): List of video prefixes (e.g., ['129', '406'])
        ground_truth_dict (dict): { 'videoname': [[file1_ids], [file2_ids]] }
        mat_file_path (str): Path to the .mat calibration file
        mots_folder (str): Folder containing the .txt tracking files
        
    Returns:
        pd.DataFrame: Summary of results
    """
    
    results = []
    versions = {
        'Version 5': strV5,
        'Version 3': strV3,
        'Version 4': strV4
    }

    print(f"Starting evaluation on {len(video_names)} videos...")
    print(f"Looking for tracking files in: {mots_folder}/")

    for video in video_names:
        print(f"\nProcessing Video: {video}")
        
        # 1. Construct File Paths
        # Looks for 'mots/8_1_clean.txt' etc.
        f1_path = os.path.join(mots_folder, f"{video}_1_clean.txt")
        f2_path = os.path.join(mots_folder, f"{video}_2_clean.txt")
        
        # Check if files exist
        if not os.path.exists(f1_path) or not os.path.exists(f2_path):
            print(f"  [Error] Files not found: {f1_path} or {f2_path}. Skipping.")
            continue
            
        # 2. Retrieve Ground Truth
        # Handles keys both as strings ("8") and ints (8) just in case
        if video in ground_truth_dict:
            gt_entry = ground_truth_dict[video]
        elif int(video) in ground_truth_dict:
            gt_entry = ground_truth_dict[int(video)]
        else:
            print(f"  [Warning] No ground truth found for {video}. Skipping evaluation.")
            continue
            
        gt_ids1 = gt_entry[0]
        gt_ids2 = gt_entry[1]
        
        # Create a lookup dictionary for Ground Truth: {id1: expected_id2}
        gt_map = dict(zip(gt_ids1, gt_ids2))
        total_gt_pairs = len(gt_map)
        
        # 3. Run Each Version
        for v_name, module in versions.items():
            print(f"  Running {v_name}...", end=" ", flush=True)
            try:
                # Run the algorithm
                # Note: We discard the second return value (the renamed df)
                pred_mapping, _ = module.run_geometric_matching(f1_path, f2_path, mat_file_path)
                
                # Convert prediction to dictionary: {id1: predicted_id2}
                pred_map = dict(zip(pred_mapping['id1'], pred_mapping['id2']))
                
                # 4. Calculate Metrics
                correct_count = 0
                missing_count = 0
                wrong_count = 0
                
                # Compare every ID in Ground Truth against Prediction
                for id1, expected_id2 in gt_map.items():
                    if id1 not in pred_map:
                        missing_count += 1
                    elif pred_map[id1] == expected_id2:
                        correct_count += 1
                    else:
                        wrong_count += 1
                
                accuracy = (correct_count / total_gt_pairs) * 100 if total_gt_pairs > 0 else 0.0
                
                results.append({
                    'Video': video,
                    'Version': v_name,
                    'Accuracy (%)': round(accuracy, 2),
                    'Correct': correct_count,
                    'Wrong': wrong_count,
                    'Missing': missing_count,
                    'Total GT': total_gt_pairs
                })
                print(f"Done. Accuracy: {accuracy:.1f}%")
                
            except Exception as e:
                print(f"Failed! Error: {e}")
                results.append({
                    'Video': video,
                    'Version': v_name,
                    'Accuracy (%)': 0.0,
                    'Correct': 0,
                    'Wrong': 0,
                    'Missing': total_gt_pairs,
                    'Total GT': total_gt_pairs,
                    'Error': str(e)
                })

    # Create Final DataFrame
    df_results = pd.DataFrame(results)
    
    # Reorder columns for readability
    cols = ['Video', 'Version', 'Accuracy (%)', 'Correct', 'Wrong', 'Missing', 'Total GT']
    if 'Error' in df_results.columns:
        cols.append('Error')
        
    return df_results[cols]

# ==========================================
#              USER CONFIGURATION
# ==========================================
if __name__ == "__main__":
    
    # 1. SETUP: List of Video Names (File Prefixes)
    my_videos = ["8", "10", "13", "14", "15", "16", "23", "129", "406"]

    # 2. SETUP: Path to your Camera Parameters .mat file
    my_mat_file = "stereoParams_Dep1.mat" 

    # 3. SETUP: Your Ground Truth Dictionary
    # Ensure this matches your actual data structure
    my_ground_truth = {
        "8": [
            [8,7,17,2,5,18,6,4,3],  # File 1 IDs
            [14,13,16,1,11,15,12,9,10]   # File 2 IDs (Matches)
        ],
        "10": [
            [7,6,3,4,5,16,17,13,2],
            [12,11,8,9,10,18,15,14,1]
        ],
        "13": [
            [11,12,13,10,9,8,7,2,1,4,3,5],
            [16,15,14,25,24,17,22,21,18,19,20,23]
        ],
        "14": [
            [4,3,2,25,5,11,9,10,6,1,7,8],
            [15,13,12,28,16,18,20,22,19,17,21,14]
        ],
        "15": [
            [13,12,11,1,14,4,2,3,6,7,17,16,8],
            [31,29,30,32,28,34,19,20,22,23,33,26,24]
        ],
        "16": [
            [4,7,32,10,2,1,24,3,5,6,8,11,9,30],
            [18,19,33,22,17,16,25,15,13,14,12,21,23,27]
        ],
        "23": [
            [5,4,3,2,22,17,16,15,7,1,6],
            [12,11,10,9,21,18,20,19,14,8,13]
        ],
        "129": [
            [9,7,8,2,1,4,3,6,5,11,10,12],
            [5,8,9,7,6,1,4,2,3,11,10,12]
        ],
        "406": [
            [9,6,7,1,2,3,5,4,10,8],
            [7,8,2,1,4,3,5,9,10,6]
        ]
        # Add more videos here...
        # "406": [ [1, 2], [5, 9] ]
    }

    # Run the comparison
    # Note: 'mots_folder' argument points to where your .txt files are
    final_report = compare_stereo_matching(
        my_videos, 
        my_ground_truth, 
        mat_file_path=my_mat_file,
        mots_folder='mots'
    )

    # Print Report
    print("\n" + "="*40)
    print("      PERFORMANCE COMPARISON REPORT      ")
    print("="*40)
    print(final_report.to_string(index=False))
    
    # Save to CSV for easy viewing
    final_report.to_csv("comparison_results.csv", index=False)