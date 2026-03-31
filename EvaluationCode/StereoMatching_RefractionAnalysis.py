import pandas as pd
import numpy as np
import scipy.io
import cv2
from scipy.io import loadmat
import argparse
import sys
from pathlib import Path
from scipy.optimize import linear_sum_assignment

# --------------------------------------------------------------------------
# REFRACTION LOGIC
# --------------------------------------------------------------------------

def refract_points_current(pts_pixel, K, D, n_air=1.0, n_water=1.333):
    """
    CURRENT IMPLEMENTATION in StereoMatching.py
    """
    if len(pts_pixel) == 0:
        return np.array([])

    pts_norm_air = cv2.undistortPoints(np.expand_dims(pts_pixel, 1), K, D).squeeze(1)

    r_air = np.linalg.norm(pts_norm_air, axis=1)
    mask = r_air > 1e-8
    
    pts_norm_water = pts_norm_air.copy()
    
    if np.any(mask):
        theta_air = np.arctan(r_air[mask])
        sin_theta_water = (n_air / n_water) * np.sin(theta_air)
        valid = np.abs(sin_theta_water) <= 1.0
        
        theta_water = np.arcsin(sin_theta_water[valid])
        r_water = np.tan(theta_water)
        
        scale = r_water / r_air[mask][valid]
        pts_norm_water[mask] = pts_norm_air[mask] * scale[:, np.newaxis]

    return pts_norm_water

def refract_points_none(pts_pixel, K, D):
    """
    NO REFRACTION CORRECTION.
    """
    if len(pts_pixel) == 0:
        return np.array([])
    pts_norm = cv2.undistortPoints(np.expand_dims(pts_pixel, 1), K, D).squeeze(1)
    return pts_norm

# --------------------------------------------------------------------------
# STEREO MATCHING LOGIC (Simplified from StereoMatching.py)
# --------------------------------------------------------------------------

def compute_matches(df1, df2, K1, K2, D1, D2, R, T):
    """
    Finds stereo matches using Epipolar Geometry (Geometry-Only).
    Returns a DataFrame with [frame, id1, id2].
    """
    # Essential Matrix
    t_skew = np.array([
        [0, -T[2], T[1]],
        [T[2], 0, -T[0]],
        [-T[1], T[0], 0]
    ])
    E = t_skew @ R
    
    common_frames = sorted(list(set(df1['frame']) & set(df2['frame'])))
    matches = []
    
    # Pre-compute refraction for matching to be fair (using current method)
    # The matching itself shouldn't change much between experiments, 
    # we want to test the RECONSTRUCTION accuracy difference.
    # So we use standard "current" refraction for finding matches.
    
    print(f"Finding matches in {len(common_frames)} frames...")
    
    for frame in common_frames:
        pts1 = df1[df1['frame'] == frame]
        pts2 = df2[df2['frame'] == frame]
        
        if pts1.empty or pts2.empty: continue
        
        ids1 = pts1['id'].values
        ids2 = pts2['id'].values
        
        # Use Current Refraction for Matching
        norm1 = refract_points_current(pts1[['x','y']].values.astype(np.float32), K1, D1)
        norm2 = refract_points_current(pts2[['x','y']].values.astype(np.float32), K2, D2)
        
        n1, n2 = len(ids1), len(ids2)
        cost_matrix = np.zeros((n1, n2))
        
        for i in range(n1):
            # l = E * p1
            p1_hom = np.array([norm1[i,0], norm1[i,1], 1.0])
            line = E @ p1_hom
            
            for j in range(n2):
                # dist = |ax+by+c| / sqrt(a^2+b^2)
                p2 = norm2[j]
                dist = abs(line[0]*p2[0] + line[1]*p2[1] + line[2]) / np.sqrt(line[0]**2 + line[1]**2)
                cost_matrix[i, j] = dist
                
        # Assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        for r, c in zip(row_ind, col_ind):
            # Threshold: 0.1 normalized distance (approx small angle)
            if cost_matrix[r, c] < 0.1: 
                matches.append({
                    'frame': frame,
                    'id1': ids1[r],
                    'id2': ids2[c],
                    'x1': pts1.iloc[r]['x'], 'y1': pts1.iloc[r]['y'],
                    'x2': pts2.iloc[c]['x'], 'y2': pts2.iloc[c]['y']
                })
                
    return pd.DataFrame(matches)

# --------------------------------------------------------------------------
# MAIN COMPARISON LOOP
# --------------------------------------------------------------------------

def run_compare(df1_path, df2_path, mat_path, output_dir):
    print(f"Loading data...")
    df1 = pd.read_csv(df1_path)
    df2 = pd.read_csv(df2_path)
    
    try:
        dd = loadmat(mat_path)
    except Exception as e:
        print(f"Error loading MAT: {e}")
        return

    K1, K2 = dd["intrinsicMatrix1"], dd["intrinsicMatrix2"]
    D1, D2 = dd["distortionCoefficients1"], dd["distortionCoefficients2"]
    R, T = dd["rotationOfCamera2"], dd["translationOfCamera2"].flatten()
    
    # 1. FIND MATCHES (Cross-Camera Association)
    matched_df = compute_matches(df1, df2, K1, K2, D1, D2, R, T)
    
    if matched_df.empty:
        print("No stereo matches found! Cannot compare 3D reconstruction.")
        return

    print(f"Found {len(matched_df)} stereo matches.")
    
    # 2. RECONSTRUCT 3D ("With" vs "Without")
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((R, T.reshape(3, 1)))
    
    results_curr = []
    results_none = []
    
    # Vectorized Reconstruction would be faster, but loop is clearer for dual-mode
    pts1_pix = matched_df[['x1', 'y1']].values.astype(np.float32)
    pts2_pix = matched_df[['x2', 'y2']].values.astype(np.float32)
    
    # MODE 1: WITH Refraction
    p1_c = refract_points_current(pts1_pix, K1, D1)
    p2_c = refract_points_current(pts2_pix, K2, D2)
    p4d_c = cv2.triangulatePoints(P1, P2, p1_c.T, p2_c.T)
    p3d_c = (p4d_c[:3] / p4d_c[3]).T
    
    # MODE 2: WITHOUT Refraction
    p1_n = refract_points_none(pts1_pix, K1, D1)
    p2_n = refract_points_none(pts2_pix, K2, D2)
    p4d_n = cv2.triangulatePoints(P1, P2, p1_n.T, p2_n.T)
    p3d_n = (p4d_n[:3] / p4d_n[3]).T
    
    # Save & Compare
    for i in range(len(matched_df)):
        row = matched_df.iloc[i]
        results_curr.append({'frame': row['frame'], 'id1': row['id1'], 'x': p3d_c[i,0], 'y': p3d_c[i,1], 'z': p3d_c[i,2]})
        results_none.append({'frame': row['frame'], 'id1': row['id1'], 'x': p3d_n[i,0], 'y': p3d_n[i,1], 'z': p3d_n[i,2]})
        
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df_c = pd.DataFrame(results_curr)
    df_n = pd.DataFrame(results_none)
    
    df_c.to_csv(out_dir / "3d_with_refraction.csv", index=False)
    df_n.to_csv(out_dir / "3d_NO_refraction.csv", index=False)
    
    print("\n--- Z-Depth Statistics (mm) ---")
    print(f"WITH Refraction: Mean Z={df_c['z'].mean():.2f}, Std={df_c['z'].std():.2f}, Min={df_c['z'].min():.2f}, Max={df_c['z'].max():.2f}")
    print(f"NO Refraction  : Mean Z={df_n['z'].mean():.2f}, Std={df_n['z'].std():.2f}, Min={df_n['z'].min():.2f}, Max={df_n['z'].max():.2f}")
    
    diff = np.sqrt(np.sum((df_c[['x','y','z']].values - df_n[['x','y','z']].values)**2, axis=1))
    print(f"\nAverage 3D Shift: {diff.mean():.2f} mm")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--c1', required=True)
    parser.add_argument('--c2', required=True)
    parser.add_argument('--mat', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    
    run_compare(args.c1, args.c2, args.mat, args.out)
