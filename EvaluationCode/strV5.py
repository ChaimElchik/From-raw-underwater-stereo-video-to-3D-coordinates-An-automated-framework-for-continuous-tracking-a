import pandas as pd
import numpy as np
import scipy.io
import cv2
from scipy.optimize import linear_sum_assignment

def refract_points(pts_pixel, K, D, n_air=1.0, n_water=1.333):
    """
    Corrects pixel coordinates for underwater refraction.
    
    1. Undistorts standard lens distortion (Pinhole model).
    2. Applies Flat Port Refraction correction (Snell's Law approximation).
       Transforms 'Air Angles' seen by the camera into 'Water Angles' where the object actually is.
    
    Args:
        pts_pixel: Numpy array of shape (N, 2)
        K: Intrinsic Matrix (3x3)
        D: Distortion Coefficients (1x5 or 1x4)
        n_air: Refractive index of air (inside housing)
        n_water: Refractive index of water (outside housing)
        
    Returns:
        pts_norm_water: Normalized coordinates (x, y) on the unit plane corresponding to the 
                        direction of the ray in water.
    """
    if len(pts_pixel) == 0:
        return np.array([])

    # 1. Undistort points to get Normalized Air Coordinates (x_a, y_a)
    # cv2.undistortPoints returns (N, 1, 2)
    pts_norm_air = cv2.undistortPoints(np.expand_dims(pts_pixel, 1), K, D)
    pts_norm_air = pts_norm_air.squeeze(1) # (N, 2)

    # 2. Apply Refraction Correction (Snell's Law)
    # Calculate radius from optical center (r = tan(theta_air))
    r_air = np.linalg.norm(pts_norm_air, axis=1)
    
    # Avoid division by zero
    mask = r_air > 1e-8
    
    pts_norm_water = pts_norm_air.copy()
    
    if np.any(mask):
        # theta_air = arctan(r_air)
        theta_air = np.arctan(r_air[mask])
        
        # Snell's Law: n_air * sin(theta_air) = n_water * sin(theta_water)
        # sin(theta_water) = (n_air / n_water) * sin(theta_air)
        sin_theta_water = (n_air / n_water) * np.sin(theta_air)
        
        # Check for Total Internal Reflection (unlikely going Air->Water, but good practice)
        valid_snell = np.abs(sin_theta_water) <= 1.0
        
        # theta_water = arcsin(...)
        theta_water = np.arcsin(sin_theta_water[valid_snell])
        
        # r_water = tan(theta_water)
        r_water = np.tan(theta_water)
        
        # Scale the original normalized points
        # New_pos = Old_pos * (r_new / r_old)
        scale_factor = r_water / r_air[mask][valid_snell]
        
        # Apply scaling
        pts_norm_water[mask] = pts_norm_air[mask] * scale_factor[:, np.newaxis]

    return pts_norm_water

def run_geometric_matching(file1_path, file2_path, mat_path, 
                                      n_air=1.0, n_water=1.333, 
                                      glass_dist_mm=None):
    """
    Runs stereo matching with underwater refraction correction.
    
    CRITICAL: 'glass_dist_mm' (Lens center to glass distance) is ideally required for 
    perfect ray tracing. This implementation uses the 'Angular Correction' method 
    which assumes the optical center is close to the glass or the object is far away 
    (Point at Infinity assumption) to avoid guessing the distance.
    """
    
    # Check Assumptions
    if glass_dist_mm is None:
        print("WARNING: 'glass_dist_mm' (lens-to-port distance) not provided.")
        print("         Using Angular Correction (Snell's Law only).") 
        print("         This assumes objects are far away relative to the housing size.")
    
    # 1. Load Camera Parameters
    try:
        mat_data = scipy.io.loadmat(mat_path)
        K1 = mat_data['intrinsicMatrix1']
        K2 = mat_data['intrinsicMatrix2']
        D1 = mat_data['distortionCoefficients1']
        D2 = mat_data['distortionCoefficients2']
        R = mat_data['rotationOfCamera2']
        t = mat_data['translationOfCamera2'].flatten()
    except KeyError as e:
        raise ValueError(f"Missing key in MAT file: {e}. Check your file structure.")

    # 2. Compute Essential Matrix (E) instead of Fundamental Matrix (F)
    # We work in Normalized Coordinates now to handle refraction cleanly.
    t_skew = np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])
    E = t_skew @ R
    
    # 3. Load Tracking Data
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)
    common_frames = sorted(list(set(df1['frame']) & set(df2['frame'])))
    
    # 4. Helper: Get Epipolar Line in Normalized Space
    def get_epipolar_line_norm(p_norm, E_mat):
        # p_norm is (x, y), append 1
        p_hom = np.array([p_norm[0], p_norm[1], 1.0])
        # Line l = E * p1
        return E_mat @ p_hom

    def point_to_line_dist_norm(p_norm, line):
        # Distance of point p2 from line l1 in normalized space (approx radians)
        a, b, c = line
        return abs(a * p_norm[0] + b * p_norm[1] + c) / np.sqrt(a**2 + b**2)

    # 5. Iterate through frames
    all_frame_matches = []
    
    for frame in common_frames:
        pts1 = df1[df1['frame'] == frame]
        pts2 = df2[df2['frame'] == frame]
        
        if pts1.empty or pts2.empty: continue
        
        ids1, ids2 = pts1['id'].values, pts2['id'].values
        coords1_pix = pts1[['x', 'y']].values.astype(np.float32)
        coords2_pix = pts2[['x', 'y']].values.astype(np.float32)
        
        # --- REFRACTION FIX ---
        # Convert Pixels -> Normalized Water Angles
        norm_coords1 = refract_points(coords1_pix, K1, D1, n_air, n_water)
        norm_coords2 = refract_points(coords2_pix, K2, D2, n_air, n_water)
        # ----------------------

        n1, n2 = len(ids1), len(ids2)
        cost_matrix = np.zeros((n1, n2))
        
        for i in range(n1):
            # Line in Camera 2 corresponding to Point in Camera 1
            line = get_epipolar_line_norm(norm_coords1[i], E)
            for j in range(n2):
                cost_matrix[i, j] = point_to_line_dist_norm(norm_coords2[j], line)
        
        # Hungarian Algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        for r, c in zip(row_ind, col_ind):
            all_frame_matches.append({
                'id1': ids1[r], 
                'id2': ids2[c], 
                'dist': cost_matrix[r, c] 
            })

    # 6. Consensus Stats
    if not all_frame_matches:
        print("No matches found.")
        return pd.DataFrame(), pd.DataFrame()

    match_df = pd.DataFrame(all_frame_matches)
    consensus = match_df.groupby(['id1', 'id2']).agg(
        count=('dist', 'count'),
        avg_dist=('dist', 'mean')
    ).reset_index()

    # 7. Sorting & Assignment
    consensus = consensus.sort_values(
        by=['count', 'avg_dist'], 
        ascending=[False, True]
    )
    
    final_matches = []
    assigned_1 = set()
    assigned_2 = set()
    
    for _, row in consensus.iterrows():
        i1, i2 = row['id1'], row['id2']
        if i1 not in assigned_1 and i2 not in assigned_2:
            final_matches.append(row)
            assigned_1.add(i1)
            assigned_2.add(i2)
            
    best_mapping = pd.DataFrame(final_matches)
    
    # 8. Rename DF2
    mapping_dict = dict(zip(best_mapping['id2'], best_mapping['id1']))
    df2_renamed = df2.copy()
    df2_renamed['id'] = df2_renamed['id'].map(mapping_dict)
    
    return best_mapping, df2_renamed


def save_remapped_tracking(mapping_df, file1_path, file2_path, out1_path, out2_path):
    """
    Applies the geometric mapping to the original tracking files.
    Renames IDs to the lowest possible integers (starting at 1).
    Preserves unmatched IDs by assigning them new unique integers.
    """
    print("Applying ID remapping...")
    
    # 1. Load original data
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # 2. Initialize Maps
    new_id_map_cam1 = {}
    new_id_map_cam2 = {}
    next_id = 1

    # 3. Process Confirmed Matches (Shared IDs)
    # These get the first batch of low IDs (e.g., 1, 2, 3...)
    for _, row in mapping_df.iterrows():
        id1, id2 = int(row['id1']), int(row['id2'])
        new_id_map_cam1[id1] = next_id
        new_id_map_cam2[id2] = next_id
        next_id += 1

    # 4. Process Unmatched IDs from Camera 1
    # Assign new unique IDs to objects seen ONLY in Cam 1
    for id1 in df1['id'].unique():
        if id1 not in new_id_map_cam1:
            new_id_map_cam1[id1] = next_id
            next_id += 1

    # 5. Process Unmatched IDs from Camera 2
    # Assign new unique IDs to objects seen ONLY in Cam 2
    for id2 in df2['id'].unique():
        if id2 not in new_id_map_cam2:
            new_id_map_cam2[id2] = next_id
            next_id += 1

    # 6. Apply Mapping
    # We use .map() and fillna with the original ID (or handle errors) if something goes wrong,
    # though our logic above covers all unique IDs found in the files.
    df1['id'] = df1['id'].map(new_id_map_cam1).astype(int)
    df2['id'] = df2['id'].map(new_id_map_cam2).astype(int)

    # 7. Save to CSV
    df1.to_csv(out1_path, index=False)
    df2.to_csv(out2_path, index=False)

    print(f"Success! Files saved:\n - {out1_path}\n - {out2_path}")
    print(f"Total unique entities (Matched + Unmatched): {next_id - 1}")

if __name__ == "__main__":
    # Define Inputs
    cam1_in = 'mots/406_1_clean.txt'
    cam2_in = 'mots/406_2_clean.txt'
    mat_params = 'stereoParams_Dep1.mat'
    
    # Define Outputs
    cam1_out = 'mots/406_1_final.csv'
    cam2_out = 'mots/406_2_final.csv'

    try:
        # Step 1: Compute Geometric Matches
        print("Running geometric matching...")
        mapping_df, _ = run_geometric_matching(
            cam1_in, cam2_in, mat_params, n_water=1.333 
        )

        if mapping_df.empty:
            print("No matches found. Skipping save.")
        else:
            # Step 2: Remap IDs and Save Files
            save_remapped_tracking(
                mapping_df, 
                cam1_in, cam2_in, 
                cam1_out, cam2_out
            )

    except Exception as e:
        print(f"An error occurred: {e}")