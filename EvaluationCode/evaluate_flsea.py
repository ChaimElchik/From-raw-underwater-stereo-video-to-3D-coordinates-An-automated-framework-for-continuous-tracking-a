import cv2
import numpy as np
import glob
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ThreeDCordinate_Maker import refract_points, triangulate_rays

def load_flsea_yaml(yaml_path):
    """Loads FLSea Stereo camera parameters from OpenCV YAML."""
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Could not open {yaml_path}")

    # Camera 1 (Left)
    lft_fx = fs.getNode("camera_lft.fx").real()
    lft_fy = fs.getNode("camera_lft.fy").real()
    lft_cx = fs.getNode("camera_lft.cx").real()
    lft_cy = fs.getNode("camera_lft.cy").real()
    
    K1 = np.array([
        [lft_fx, 0.0, lft_cx],
        [0.0, lft_fy, lft_cy],
        [0.0, 0.0, 1.0]
    ])
    
    # Distortion: The yaml gives k1, k2, p1, p2. 
    D1 = np.array([
        fs.getNode("camera_lft.k1").real(),
        fs.getNode("camera_lft.k2").real(),
        fs.getNode("camera_lft.p1").real(),
        fs.getNode("camera_lft.p2").real(),
        0.0  # k3 usually 0 if not provided
    ])

    # Camera 2 (Right)
    rgt_fx = fs.getNode("camera_rgt.fx").real()
    rgt_fy = fs.getNode("camera_rgt.fy").real()
    rgt_cx = fs.getNode("camera_rgt.cx").real()
    rgt_cy = fs.getNode("camera_rgt.cy").real()
    
    K2 = np.array([
        [rgt_fx, 0.0, rgt_cx],
        [0.0, rgt_fy, rgt_cy],
        [0.0, 0.0, 1.0]
    ])
    
    D2 = np.array([
        fs.getNode("camera_rgt.k1").real(),
        fs.getNode("camera_rgt.k2").real(),
        fs.getNode("camera_rgt.p1").real(),
        fs.getNode("camera_rgt.p2").real(),
        0.0
    ])

    # Extrinsics (T_c1_c2 is Cam 2 -> Cam 1)
    T = fs.getNode("Stereo.T_c1_c2").mat()
    R = T[:3, :3]
    t = T[:3, 3] / 1000.0 # Translation vector in meters

    fs.release()
    return K1, D1, K2, D2, R, t

# ---------------------------------------------------------------------------
# HOUSING / REFRACTION PARAMETERS
# ---------------------------------------------------------------------------

# CRITICAL NOTE: The FLSea Stereo dataset was captured using Hugyfot HFN-D810 
# housings equipped with Dome Ports, not flat ports. Because dome ports minimize 
# the 1-3 cm lateral physical shift (r_exit) caused by flat glass, running the 
# 3D coordinate maker on this dataset effectively tests its fallback Pinhole/Linear-LS 
# logic rather than proving the flat-port skew-ray intersection fixes.
# We set d_air=0, d_glass=0, n_water=1.0 to disable flat-port refraction correction.
d_air = 0.0
d_glass = 0.0
n_air = 1.0
n_water = 1.0
n_glass = 1.0

def evaluate_flsea_stereo(left_img_dir, right_img_dir, calib_yaml, output_dir="flsea_eval_plots"):
    """
    Evaluates stereo calibration accuracy by triangulating checkerboard corners.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find matching image pairs based on timestamps
    left_images = sorted(glob.glob(os.path.join(left_img_dir, '*.png')) + glob.glob(os.path.join(left_img_dir, '*.tiff')) + glob.glob(os.path.join(left_img_dir, '*.tif')))
    right_images = sorted(glob.glob(os.path.join(right_img_dir, '*.png')) + glob.glob(os.path.join(right_img_dir, '*.tiff')) + glob.glob(os.path.join(right_img_dir, '*.tif')))
    
    if len(left_images) == 0 or len(right_images) == 0:
        print(f"No images found in {left_img_dir} or {right_img_dir}")
        return

    # Load parameters
    K1, D1, K2, D2, R, t = load_flsea_yaml(calib_yaml)
    
    # Store detected objective points (mostly for visualization/reprojection)
    all_points_3d = []
    
    # Store metrics for global statistics
    frame_metrics = []
    
    paired = min(len(left_images), len(right_images))
    print(f"Testing on {paired} image pairs...")

    for i in range(paired):
        imgL_path = left_images[i]
        imgR_path = right_images[i]
        
        # We need identical or matched timestamps. 
        # For FLSea, the filenames are the unix timestamps in nanoseconds.
        # This simple zip assumes they align exactly or are already synced.
        
        imgL = cv2.imread(imgL_path, cv2.IMREAD_GRAYSCALE)
        imgR = cv2.imread(imgR_path, cv2.IMREAD_GRAYSCALE)
        
        if imgL is None or imgR is None:
            continue
            
        # The user requested to evaluate using only images that have the full checkerboard in view.
        # Based on visual confirmations, the full inner board size is 9x6.
        grid_sizes = [(9, 6)]
        found_size = None
        
        for size in grid_sizes:
            retL, cornersL = cv2.findChessboardCorners(imgL, size, None)
            if retL:
                retR, cornersR = cv2.findChessboardCorners(imgR, size, None)
                if retR:
                    found_size = size
                    break
        
        if found_size:
            w, h = found_size
            print(f"[{i}] Found {w}x{h} corners in both images: {Path(imgL_path).name}")
            
            # Subpixel refinement
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            cornersL = cv2.cornerSubPix(imgL, cornersL, (11, 11), (-1, -1), criteria)
            cornersR = cv2.cornerSubPix(imgR, cornersR, (11, 11), (-1, -1), criteria)
            
            # Format (N, 2)
            ptsL = cornersL.reshape(-1, 2)
            ptsR = cornersR.reshape(-1, 2)
            
            # --- 1. THE ADVANCED METHOD (Your Custom Ray Intersection) ---
            # Using correct_refraction=False because they are behind dome ports
            O1_local, D1_local = refract_points(ptsL, K1, D1, 
                                                n_water=n_water, n_glass=n_glass, 
                                                d_air=d_air, d_glass=d_glass, correct_refraction=False)
                                                
            O2_local, D2_local = refract_points(ptsR, K2, D2, 
                                                n_water=n_water, n_glass=n_glass, 
                                                d_air=d_air, d_glass=d_glass, correct_refraction=False)
                                                
            # Transformations from Cam 2 to Cam 1 (YAML T_c1_c2 is Cam 1 -> Cam 2)
            # Row Vectors: X_1 = (X_2 - t) @ R
            O2_global = (O2_local - t) @ R
            D2_global = D2_local @ R
            
            # --- 1. ADVANCED METHOD: Geometric midterm ---
            def triangulate_local(O1, D1, O2, D2):
                w0 = O1 - O2
                b = np.sum(D1 * D2, axis=1)
                d = np.sum(D1 * w0, axis=1)
                e = np.sum(D2 * w0, axis=1)
                denom = 1.0 - b*b
                denom[np.abs(denom) < 1e-8] = 1e-8
                t1 = (b*e - d) / denom
                t2 = (e - b*d) / denom
                P1 = O1 + t1[:, np.newaxis] * D1
                P2 = O2 + t2[:, np.newaxis] * D2
                return (P1 + P2) / 2.0
            
            points_3d_adv = triangulate_local(O1_local, D1_local, O2_global, D2_global)
            
            # --- 2. SIMPLE METHOD: Algebraic DLT ---
            # Undistort the points to normalized image coordinates (z=1 plane)
            pts1_norm = cv2.undistortPoints(cornersL, K1, D1).reshape(-1, 2).T
            pts2_norm = cv2.undistortPoints(cornersR, K2, D2).reshape(-1, 2).T
            
            # Projection Matrix P = K[R|t] where [R|t] is Cam 1 -> Cam 2
            P1_simple = np.eye(3, 4)
            P2_simple = np.hstack((R, t.reshape(3, 1)))
            
            points_4d = cv2.triangulatePoints(P1_simple, P2_simple, pts1_norm, pts2_norm)
            points_3d_simple = (points_4d / points_4d[3])[:3].T 
            
            if i == 4 and Path(imgL_path).name == "LFT_cal_000030.tif":
                print(f"DEBUG Frame 30:")
                print(f"  t: {t}")
                print(f"  Sample Adv Point 0: {points_3d_adv[0]}")
                print(f"  Sample Sim Point 0: {points_3d_simple[0]}")
            
            # 3. Calculate grid spacing (Ground Truth is 0.038m per square)
            def get_grid_stats(pts_3d, w, h):
                grid = pts_3d.reshape((h, w, 3))
                edges = []
                for r in range(h):
                    for c in range(w-1): edges.append(np.linalg.norm(grid[r, c] - grid[r, c+1]))
                for r in range(h-1):
                    for c in range(w): edges.append(np.linalg.norm(grid[r, c] - grid[r+1, c]))
                return np.mean(edges), np.std(edges)
                    
            mean_adv, std_adv = get_grid_stats(points_3d_adv, w, h)
            mean_simp, std_simp = get_grid_stats(points_3d_simple, w, h)
            
            # Save for global report
            frame_metrics.append({
                'img_name': Path(imgL_path).name,
                'mean_adv': mean_adv * 1000.0,
                'std_adv': std_adv * 1000.0,
                'mean_simp': mean_simp * 1000.0,
                'std_simp': std_simp * 1000.0,
                'error_adv': (mean_adv - 0.038) * 1000.0,
                'error_simp': (mean_simp - 0.038) * 1000.0
            })
            
            print(f"  -> [ADV] Mean: {mean_adv*1000:.2f}mm | STD: {std_adv*1000:.2f}mm")
            print(f"  -> [SIM] Mean: {mean_simp*1000:.2f}mm | STD: {std_simp*1000:.2f}mm")
            all_points_3d.append(points_3d_adv)

            # Visualization (using Advanced)
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            points_grid = points_3d_adv.reshape((h, w, 3))
            ax.scatter(points_3d_adv[:, 0], points_3d_adv[:, 1], points_3d_adv[:, 2], c='b', marker='o')
            for r in range(h): ax.plot(points_grid[r, :, 0], points_grid[r, :, 1], points_grid[r, :, 2], 'r-', alpha=0.5)
            for c in range(w): ax.plot(points_grid[:, c, 0], points_grid[:, c, 1], points_grid[:, c, 2], 'r-', alpha=0.5)
            ax.set_title(f"3D Triangulation: {Path(imgL_path).name}\n[Adv] Mean: {mean_adv*1000:.2f}mm | [Sim] Mean: {mean_simp*1000:.2f}mm")
            ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Depth Z (m)')

            plot_path = os.path.join(output_dir, f"plot_{Path(imgL_path).stem}.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"  -> Saved visualization to {plot_path}")
        else:
            print(f"[{i}] Missing corners.")

    print(f"\nSuccessfully processed {len(all_points_3d)} image pairs out of {paired}.")
    
    # Generate the requested metrics report
    if frame_metrics:
        report_path = os.path.join(output_dir, "flsea_ablation_metrics.md")
        with open(report_path, "w") as f:
            f.write("# FLSea Ablation Study: Advanced vs Simple Triangulation\n\n")
            f.write("## Global Comparison\n")
            f.write(f"- **Total Frames:** {len(frame_metrics)}\n")
            f.write(f"- **[ADV] Avg Error:** {np.mean([fm['error_adv'] for fm in frame_metrics]):.2f} mm\n")
            f.write(f"- **[SIM] Avg Error:** {np.mean([fm['error_simp'] for fm in frame_metrics]):.2f} mm\n")
            f.write(f"- **[ADV] Avg StdDev:** {np.mean([fm['std_adv'] for fm in frame_metrics]):.2f} mm\n")
            f.write(f"- **[SIM] Avg StdDev:** {np.mean([fm['std_simp'] for fm in frame_metrics]):.2f} mm\n\n")
            
            f.write("## Per-Frame Metrics\n")
            f.write("| Image | Adv Err (mm) | Adv STD | Sim Err (mm) | Sim STD |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- |\n")
            for fm in frame_metrics:
                f.write(f"| {fm['img_name']} | {fm['error_adv']:.2f} | {fm['std_adv']:.2f} | {fm['error_simp']:.2f} | {fm['std_simp']:.2f} |\n")
        
        print(f"Saved ablation report to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 3D Triangulation on FLSea Stereo Data")
    parser.add_argument("--left_dir", type=str, required=True, help="Path to left camera images")
    parser.add_argument("--right_dir", type=str, required=True, help="Path to right camera images")
    parser.add_argument("--calib", type=str, required=True, help="Path to OpenCV calibration YAML file")
    parser.add_argument("--output_dir", type=str, default="flsea_eval_plots", help="Directory to save 3D plots")
    
    args = parser.parse_args()
    
    evaluate_flsea_stereo(args.left_dir, args.right_dir, args.calib, args.output_dir)
