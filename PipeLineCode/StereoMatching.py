import pandas as pd
import numpy as np
import scipy.io
import cv2
from scipy.optimize import linear_sum_assignment

def refract_points(pts_pixel, K, D, n_air=1.0, n_water=1.333, n_glass=1.5, d_air=0.0, d_glass=0.0, correct_refraction=False):
    """
    Converts Pixel Coordinates -> 3D Rays (Origins + Directions) in Water.
    Handles Air -> Glass -> Water interface.
    
    Returns:
        tuple: (ray_origins, ray_directions)
        - ray_origins: (N, 3) offset points on the glass-water surface.
        - ray_directions: (N, 3) normalized direction vectors in water.
    """
    if len(pts_pixel) == 0:
        return np.array([]), np.array([])

    # 1. Undistort to Normalized Coordinates (Ray direction in Air)
    pts_norm = cv2.undistortPoints(np.expand_dims(pts_pixel, 1), K, D).squeeze(1)
    
    # Directions in Air (u, v, 1) normalized
    rays_air = np.column_stack((pts_norm, np.ones(len(pts_norm))))
    norms = np.linalg.norm(rays_air, axis=1, keepdims=True)
    rays_air /= norms
    
    if not correct_refraction:
        # Underwater calibration -> rays start at 0,0,0
        return np.zeros_like(rays_air), rays_air

    # 2. Refraction
    r_air = np.linalg.norm(pts_norm, axis=1) # tan(theta_air)
    mask = r_air > 1e-8
    
    rays_water = rays_air.copy()
    origins_water = np.zeros_like(rays_air)
    
    if np.any(mask):
        theta_air = np.arctan(r_air[mask])
        
        # Angle Correction (Snell's Law 1->3)
        sin_theta_water = (n_air / n_water) * np.sin(theta_air)
        valid = np.abs(sin_theta_water) <= 1.0
        theta_water = np.arcsin(sin_theta_water[valid])
        
        # Lateral Shift (Snell's Law 1->2 for Angle 2)
        sin_theta_glass = (n_air / n_glass) * np.sin(theta_air[valid])
        theta_glass = np.arcsin(sin_theta_glass)
        
        # Radial Exit Point at Water Interface (Z = d_air + d_glass)
        r_exit = d_air * np.tan(theta_air[valid]) + d_glass * np.tan(theta_glass)
        z_exit = d_air + d_glass
        
        # Start Point (Origin):
        # We define the ray origin at the glass-water interface.
        u_r = pts_norm[mask][valid] / r_air[mask][valid][:, np.newaxis]
        ox = r_exit[:, np.newaxis] * u_r
        oz = np.full((len(ox), 1), z_exit)
        
        # Sub-origins
        sub_origins = np.hstack((ox, oz))
        
        # Directions
        vz = np.cos(theta_water)
        vr = np.sin(theta_water)
        vx = vr[:, np.newaxis] * u_r
        sub_dirs = np.hstack((vx, vz[:, np.newaxis]))
        
        # Map back
        temp_idxs = np.where(mask)[0]
        final_idxs = temp_idxs[valid]
        origins_water[final_idxs] = sub_origins
        rays_water[final_idxs] = sub_dirs

    return origins_water, rays_water

def run_geometric_matching(file1_path, file2_path, mat_path, 
                                      n_air=1.0, n_water=1.333, 
                                      n_glass=1.5, d_air=0.0, d_glass=0.0,
                                      glass_dist_mm=None, # Legacy, mapped to d_air if d_air is 0?
                                      correct_refraction=False,
                                      max_epipolar_dist=0.12):
    """
    Runs stereo matching, optionally with underwater refraction correction.
    Only matches if the epipolar distance is below max_epipolar_dist.
    """
    
    # Handle legacy parameter
    if glass_dist_mm is not None and d_air == 0.0:
        d_air = glass_dist_mm
    
    # Check Assumptions
    if correct_refraction and d_air == 0.0 and d_glass == 0.0 and glass_dist_mm is None:
        pass # It assumes 0 shift (Angular correction only), which is fine.
    
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

    # 2. Compute Essential Matrix (E)
    # NOTE: E assumes central projection.
    # Refraction shifts the center.
    # Ideally we should assume the "Virtual Center" is close to 0,0,0 or use the Directions.
    # Since we can't easily change the epipolar solver to generic rays, 
    # we use the Refracted DIRECTIONS as if they came from 0,0,0.
    
    t_skew = np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])
    E = t_skew @ R
    
    # 3. Load Tracking Data
    names = ["frame", "id", "x", "y", "w", "h", "conf", "cls", "vis"]
    df1 = pd.read_csv(file1_path, header=None, names=names)
    df2 = pd.read_csv(file2_path, header=None, names=names)
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
        coords1_pix = (pts1[['x', 'y']].values + pts1[['w', 'h']].values / 2.0).astype(np.float32)
        coords2_pix = (pts2[['x', 'y']].values + pts2[['w', 'h']].values / 2.0).astype(np.float32)
        
        # --- REFRACTION FIX ---
        # Convert Pixels -> Normalized Coordinates (Water or Air)
        # refract_points returns (origins, directions)
        _, dirs1 = refract_points(coords1_pix, K1, D1, n_air, n_water, n_glass, d_air, d_glass, correct_refraction)
        _, dirs2 = refract_points(coords2_pix, K2, D2, n_air, n_water, n_glass, d_air, d_glass, correct_refraction)
        
        # Convert Directions (x,y,z) to Normalized (x/z, y/z) for Epipolar
        # directions are (N, 3)
        z1 = dirs1[:, 2]
        z1[z1 == 0] = 1e-8
        norm_coords1 = (dirs1[:, :2] / z1[:, np.newaxis])
        
        z2 = dirs2[:, 2]
        z2[z2 == 0] = 1e-8
        norm_coords2 = (dirs2[:, :2] / z2[:, np.newaxis])
        # ----------------------

        n1, n2 = len(ids1), len(ids2)
        cost_matrix = np.zeros((n1, n2))
        
        for i in range(n1):
            # Line in Camera 2 corresponding to Point in Camera 1
            line = get_epipolar_line_norm(norm_coords1[i], E)
            for j in range(n2):
                dist = point_to_line_dist_norm(norm_coords2[j], line)
                cost_matrix[i, j] = dist if dist <= max_epipolar_dist else 1e9
        
        # Hungarian Algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        for r, c in zip(row_ind, col_ind):
            cost = cost_matrix[r, c]
            if cost < 1e9:
                all_frame_matches.append({
                    'id1': ids1[r], 
                    'id2': ids2[c], 
                    'dist': cost
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

    # Save tracking info of unmatched IDs
    unmatched_records = []
    
    # Extract records for Unmatched Cam1 IDs
    for id1, new_id in new_id_map_cam1.items():
        if new_id not in new_id_map_cam2.values(): # It's not in Cam2
            rows = df1[df1['id'] == id1].copy()
            rows['id'] = new_id
            rows['camera'] = 'cam1'
            unmatched_records.append(rows)
            
    # Extract records for Unmatched Cam2 IDs
    for id2, new_id in new_id_map_cam2.items():
        if new_id not in new_id_map_cam1.values(): # It's not in Cam1
            rows = df2[df2['id'] == id2].copy()
            rows['id'] = new_id
            rows['camera'] = 'cam2'
            unmatched_records.append(rows)

    if unmatched_records:
        unmatched_df = pd.concat(unmatched_records, ignore_index=True)
        import os
        out_dir = os.path.dirname(out1_path)
        unmatched_out = os.path.join(out_dir, "Unmatched_IDs_Tracking.csv")
        unmatched_df.to_csv(unmatched_out, index=False)
        print(f"Saved Unmatched ID Tracking to: {unmatched_out}")
    else:
        print("No unmatched IDs found.")

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
