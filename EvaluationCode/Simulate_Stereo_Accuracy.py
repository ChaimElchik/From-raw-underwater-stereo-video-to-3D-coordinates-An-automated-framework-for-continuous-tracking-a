
import numpy as np
import cv2
import scipy.io
from scipy.optimize import minimize
import pandas as pd
import argparse
import sys
import os

# --- Import Pipeline Functions ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/Final_Pipeline_Code")
# Attempt imports (assuming script is in root or adjacent)
try:
    from Final_Pipeline_Code.ThreeDCordinate_Maker import refract_points, triangulate_rays
except ImportError:
    # If running from inside Final_Pipeline_Code
    try:
        from ThreeDCordinate_Maker import refract_points, triangulate_rays
    except ImportError:
        print("Could not import pipeline functions. Check paths.")
        sys.exit(1)

GAME_MAT = "stereoParams_Dep4.mat"

def load_camera_params(mat_file):
    if not os.path.exists(mat_file):
        # Fallback for mocked environment if file missing
        print(f"Warning: {mat_file} not found. Using dummy params.")
        K = np.array([[2000, 0, 1000], [0, 2000, 1000], [0, 0, 1]])
        D = np.zeros(5)
        R = np.eye(3)
        t = np.array([-100, 0, 0]).astype(float) # 100mm baseline
        return K, D, K, D, R, t

    mat = scipy.io.loadmat(mat_file)
    K1 = mat['intrinsicMatrix1']
    D1 = mat['distortionCoefficients1']
    K2 = mat['intrinsicMatrix2']
    D2 = mat['distortionCoefficients2']
    R = mat['rotationOfCamera2']
    t = mat['translationOfCamera2'].flatten()
    return K1, D1, K2, D2, R, t

def forward_project_refracted(point_3d, K, D, n_air=1.0, n_glass=1.5, n_water=1.333, d_air=0.0, d_glass=0.0, R_cam_to_world=np.eye(3), t_cam_to_world=np.zeros(3)):
    """
    Finds the pixel (u,v) in the camera that sees point_3d, accounting for refraction.
    The camera is at (0,0,0) in its local frame.
    point_3d is in World Frame (or Camera Frame if R=I, t=0).
    
    We treat this as an optimization problem:
    Find the initial ray direction (u_n, v_n) in Normalized Camera coords (Z=1)
    such that the refracted ray passes through P_local.
    """
    
    # Transform P to local camera frame
    # P_local = R (P_world) + t  (Standard CV convention for extrinsic is P_cam = R P_w + t)
    # But usually we pass Extrinsics that transform World->Cam.
    # In this script, we'll assume point_3d is already in Cam Frame for simplicity of function,
    # or handle transform if R/t provided.
    
    # Let's assume point_3d is already transformed to Local Camera Frame for this function calculation.
    Target = point_3d 
    
    def objective(uv):
        # uv is [u_n, v_n] in normalized image plane (z=1)
        # Construct ray in air
        ray_air = np.array([uv[0], uv[1], 1.0])
        ray_air = ray_air / np.linalg.norm(ray_air)
        
        # Refract Air -> Glass
        # Normal is (0,0,1)
        # Snell's Law implementation explicitly here to be safe
        
        # 1. Intersection with Glass Surface (Z = d_air)
        # P0 = (0,0,0)
        # distance t1 = d_air / ray_air[2]
        P1 = ray_air * (d_air / ray_air[2])
        
        # Refract at P1
        # n1 sin1 = n2 sin2
        # Vector math form: r2 = r1*mu + n*(...)
        # Assume normal is n = [0, 0, 1]
        
        def refract_vec(v_in, n_surf, n1, n2):
            mu = n1/n2
            dot = np.dot(v_in, n_surf)
            k = 1.0 - mu*mu * (1.0 - dot*dot)
            if k < 0: return np.zeros_like(v_in) # TIR
            return mu * v_in - (mu*dot + np.sqrt(k)) * n_surf

        # Normal should point against the ray (towards Air/Camera)
        normal = np.array([0, 0, -1.0])
        ray_glass = refract_vec(ray_air, normal, n_air, n_glass)
        ray_glass /= np.linalg.norm(ray_glass)
        
        # 2. Intersection with Water Surface (Z = d_air + d_glass)
        # dist t2. Delta Z is d_glass.
        # t2 = d_glass / ray_glass[2]
        P2 = P1 + ray_glass * (d_glass / ray_glass[2])
        
        # Refract Glass -> Water
        ray_water = refract_vec(ray_glass, normal, n_glass, n_water)
        ray_water /= np.linalg.norm(ray_water)
        
        # 3. Distance to target Z plane
        # Target Z is Target[2]
        # dist t3 = (Target[2] - P2[2]) / ray_water[2]
        P3 = P2 + ray_water * ((Target[2] - P2[2]) / ray_water[2])
        
        # Error: Distance between P3 and Target in XY
        err = np.linalg.norm(P3[:2] - Target[:2])
        return err

    # Initial guess: Pinhole projection of Target
    uv_init = Target[:2] / Target[2]
    
    res = minimize(objective, uv_init, method='Nelder-Mead', tol=1e-6)
    
    # Debug info for first few failures
    if not res.success and calculate_final_err:
        pass
             
    # Calculate final pixel
    uv_final = res.x
    
    # pt_norm = np.array([[res.x[0], res.x[1], 0.0]]) # Not needed
    
    pt_3d_norm = np.array([[res.x[0], res.x[1], 1.0]])
    pix_tmp, _ = cv2.projectPoints(pt_3d_norm, np.zeros(3), np.zeros(3), K, D)
    return pix_tmp[0][0] # [x, y]

calculate_final_err = True

def run_simulation(N=2, glass_d=50.0, glass_th=10.0):
    print(f"--- Running Simulation (N={N} points) ---", flush=True)
    print(f"Params: d_air={glass_d}mm, d_glass={glass_th}mm", flush=True)
    
    K1, D1, K2, D2, R, t = load_camera_params(GAME_MAT)
    print("Camera Params Loaded:", flush=True)
    print(f"t (Translation): {t}", flush=True)
    print(f"R (Rotation):\n{R}", flush=True)
    print(f"K1:\n{K1}", flush=True)

    # 1. Generate Ground Truth (World Frame = Cam 1 Frame)
    
    # 1. Generate Ground Truth (World Frame = Cam 1 Frame)
    # X: -500 to 500 mm
    # Y: -500 to 500 mm
    # Z: 1000 to 4000 mm (1m to 4m)
    pts_gt = []
    for _ in range(N):
        x = np.random.uniform(-400, 400)
        y = np.random.uniform(-400, 400)
        z = np.random.uniform(1000, 4000)
        pts_gt.append([x, y, z])
    pts_gt = np.array(pts_gt)
    
    errors_pinhole = []
    errors_rayray = []
    
    for idx, P_world in enumerate(pts_gt):
        # Cam 1 sees P_world directly (Identity transform)
        pix1 = forward_project_refracted(P_world, K1, D1, d_air=glass_d, d_glass=glass_th)
        
        # Cam 2 sees P_world. Transform P_world to Cam 2 Frame.
        # P_cam2 = R * P_world + t
        # Note: loadmat usually gives rotation of CAM2.
        # Check standard stereo calibration: P2 = R*P1 + T.
        P_cam2 = R @ P_world + t
        pix2 = forward_project_refracted(P_cam2, K2, D2, d_air=glass_d, d_glass=glass_th)
        
        if pix1 is None or pix2 is None:
            continue
            
        # Reconstruct
        # Prepare inputs as (1, 2) arrays
        pts1_pix = np.array([pix1])
        pts2_pix = np.array([pix2])
        
        # --- DEBUG SINGLE RAY (First point only) ---
        if idx == 0:
            print(f"\n--- DEBUG RAY (Point {idx}) ---")
            print(f"Target World: {P_world}")
            print(f"Pix1: {pix1}")
            
            # Check Cam 1 Ray
            O1_r, D1_r = refract_points(pts1_pix, K1, D1, d_air=glass_d, d_glass=glass_th, correct_refraction=True)
            # Distance from Line to Point
            # Line: O + t*D
            # Vector O->P: v = P - O
            # Projection of v on D: t = dot(v, D)
            # Closest Point on Line: C = O + t*D
            # Dist = norm(P - C)
            v = P_world - O1_r[0]
            t_proj = np.dot(v, D1_r[0])
            C = O1_r[0] + t_proj * D1_r[0]
            dist = np.linalg.norm(P_world - C)
            print(f"Ray 1 (Cam1) Dist to Target: {dist:.2f} mm")
            
            # Check Cam 2 Ray
            O2_r, D2_r = refract_points(pts2_pix, K2, D2, d_air=glass_d, d_glass=glass_th, correct_refraction=True)
            # P_cam2 is the target in Cam 2 frame
            v2 = P_cam2 - O2_r[0]
            t_proj2 = np.dot(v2, D2_r[0])
            C2 = O2_r[0] + t_proj2 * D2_r[0]
            dist2 = np.linalg.norm(P_cam2 - C2)
            print(f"Ray 2 (Cam2 Frame) Dist to Target (Cam2 Frame): {dist2:.2f} mm")
            
            if dist > 50 or dist2 > 50:
                print("!! CRITICAL: Ray generation invalid !!")
        # -------------------------------------------
        
        # --- METHOD A: PINHOLE (Old) ---
        # Simulate 'old' method by calling refraction with 0 params or just standard undistort
        # We can use the exposed refract_points with correct_refraction=False
        O1_p, D1_p = refract_points(pts1_pix, K1, D1, correct_refraction=False)
        O2_p, D2_p = refract_points(pts2_pix, K2, D2, correct_refraction=False)
        
        # Standard Triangulation (using CV2 for fairness as it was the old way)
        # But we can approximate using our ray-ray with origin 0, it's mathematically similar to DLT
        # Let's use the code's logic manually to match ThreeDCordinate_Maker.
        
        # Cam 1 Rays
        # ... derived inside cor_maker_3d's "if not use_ray_tracing" block ...
        # Let's just use triangulate_rays with Origins=0. It generalizes Pinhole.
        O1_global_p = np.zeros_like(D1_p)
        O2_global_p = (np.zeros_like(D2_p) - t) @ R # Transform O2(0,0,0) to world
        D1_global_p = D1_p
        D2_global_p = D2_p @ R
        
        P_est_pinhole = triangulate_rays(O1_global_p, D1_global_p, O2_global_p, D2_global_p)[0]
        
        # --- METHOD B: RAY-RAY (New) ---
        O1_r, D1_r = refract_points(pts1_pix, K1, D1, d_air=glass_d, d_glass=glass_th, correct_refraction=True)
        O2_r, D2_r = refract_points(pts2_pix, K2, D2, d_air=glass_d, d_glass=glass_th, correct_refraction=True)
        
        O1_global_r = O1_r
        # Transform Cam2 Rays to World(Cam1)
        # O2_global = (O2_local - t) @ R
        O2_global_r = (O2_r - t) @ R
        D2_global_r = D2_r @ R
        D1_global_r = D1_r
        
        P_est_ray = triangulate_rays(O1_global_r, D1_global_r, O2_global_r, D2_global_r)[0]
        
        # Errors
        err_a = np.linalg.norm(P_est_pinhole - P_world)
        err_b = np.linalg.norm(P_est_ray - P_world)
        
        errors_pinhole.append(err_a)
        errors_rayray.append(err_b)
        
    rmse_a = np.sqrt(np.mean(np.square(errors_pinhole)))
    rmse_b = np.sqrt(np.mean(np.square(errors_rayray)))
    
    print(f"\nRESULTS (N={len(errors_pinhole)} valid points):")
    print(f"Method A (Pinhole/Old): RMSE = {rmse_a:.2f} mm")
    print(f"Method B (Ray-Ray/New): RMSE = {rmse_b:.2f} mm")
    
    improvement = rmse_a - rmse_b
    print(f"Improvement: {improvement:.2f} mm")
    
    # Save to CSV for plotting if needed
    df = pd.DataFrame({
        'Error_Pinhole': errors_pinhole,
        'Error_New': errors_rayray
    })
    df.to_csv("simulation_results.csv", index=False)
    print("Results saved to simulation_results.csv")

if __name__ == "__main__":
    run_simulation()
