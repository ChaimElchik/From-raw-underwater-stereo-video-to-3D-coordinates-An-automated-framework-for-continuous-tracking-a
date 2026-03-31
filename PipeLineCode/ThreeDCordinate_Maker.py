import pandas as pd
import numpy as np
import scipy.io
import cv2
from scipy.io import loadmat

def refract_points(pts_pixel, K, D, n_air=1.0, n_water=1.333, n_glass=1.5, d_air=0.0, d_glass=0.0, correct_refraction=False):
    """
    Converts Pixel Coordinates -> 3D Rays (Origins + Directions) in Water.
    Handles Air -> Glass -> Water interface.
    
    Returns:
        tuple: (ray_origins, ray_directions)
        - ray_origins: (N, 3) offset points on the glass-water surface (or virtual center).
        - ray_directions: (N, 3) normalized direction vectors in water.
    """
    if len(pts_pixel) == 0:
        return np.array([]), np.array([])

    # 1. Undistort to Normalized Coordinates (Ray direction in Air)
    # Pinhole Model: Z=1, so vector is (u, v, 1)
    # pts_norm shape (N, 2) -> (u, v)
    pts_norm = cv2.undistortPoints(np.expand_dims(pts_pixel, 1), K, D).squeeze(1)
    
    # Convert to 3D vectors (u, v, 1)
    # Normalize them? No, keep Z=1 for easy math with depths initially
    # But usually directions are unit vectors.
    rays_air = np.column_stack((pts_norm, np.ones(len(pts_norm))))
    norms = np.linalg.norm(rays_air, axis=1, keepdims=True)
    rays_air /= norms # Unit vectors
    
    if not correct_refraction:
        # If no correction (underwater calibration), assume rays start at (0,0,0) with these directions
        return np.zeros_like(rays_air), rays_air

    # 2. Refraction (Air -> Glass -> Water)
    # We essentially need the exit angle and the exit point shift.
    # Angle depends only on n_air -> n_water (Invariant to glass index for parallel plates)
    
    # Compute angles relative to Optical Axis (Z)
    # dot(v, [0,0,1]) = vz = cos(theta_air)
    # But we want sin(theta).
    # r_air = sqrt(x^2 + y^2) / z. Since z component is vz/vz...
    # Let's use the radial distance in the normalized plane (z=1) for simplicity as before
    
    r_air = np.linalg.norm(pts_norm, axis=1) # tan(theta_air)
    mask = r_air > 1e-8
    
    rays_water = rays_air.copy()
    origins_water = np.zeros_like(rays_air)
    
    if np.any(mask):
        theta_air = np.arctan(r_air[mask])
        
        # --- ANGLE CORRECTION ---
        # n1 sin1 = n3 sin3
        sin_theta_water = (n_air / n_water) * np.sin(theta_air)
        valid = np.abs(sin_theta_water) <= 1.0
        theta_water = np.arcsin(sin_theta_water[valid])
        
        # --- LATERAL SHIFT CORRECTION ---
        # Snell's Law for Glass intermediate: n1 sin1 = n2 sin2
        sin_theta_glass = (n_air / n_glass) * np.sin(theta_air[valid])
        theta_glass = np.arcsin(sin_theta_glass)
        
        # Radial offsets: r = d * tan(theta)
        # Shift in R (radial) from the central axis at the Water interface
        # R_exit = d_air * tan(theta_air) + d_glass * tan(theta_glass)
        # Wait! The "Camera Center" is usually 0.
        # So at Z=0 (Center), R=0.
        # At Z=d_air (Glass Start): R1 = d_air * tan(theta_air)
        # At Z=d_air+d_glass (Water Start): R2 = R1 + d_glass * tan(theta_glass)
        
        # The ray in water starts at (R2, Z_exit) and has angle theta_water.
        # We need its Origin.
        # It's convenient to define Origin at the Glass-Water interface?
        # Or shift it back to where it WOULD intersect Z=0? (Virtual Center).
        # Virtual Center R_virtual = R2 - (d_air+d_glass) * tan(theta_water).
        # Shift_radial = R_virtual (since it should be 0).
        
        r_exit = d_air * np.tan(theta_air[valid]) + d_glass * np.tan(theta_glass)
        z_exit = d_air + d_glass
        
        # Virtual shift back to Z=0 plane (optional, but helps keeping Z~0 based)
        # OR just allow non-zero origins. Let's return Origins at the Water Surface (Z = z_exit)
        # Then triangulation handles it.
        
        # Scale unit vectors in X,Y to match R_exit direction
        # The direction in X,Y plane is same as pts_norm (normalized x,y).
        # Unit radial vector u_r = (x,y) / r_air.
        
        u_r = pts_norm[mask][valid] / r_air[mask][valid][:, np.newaxis]
        
        # Org_x = R_exit * u_r_x
        # Org_y = R_exit * u_r_y
        # Org_z = z_exit
        
        ox = r_exit[:, np.newaxis] * u_r
        oz = np.full((len(ox), 1), z_exit)
        
        origins_water[mask] = 0 # Default 0
        
        # Re-assign valid ones.
        # We need to handle the double masking.
        # Create temp arrays for the masked subset
        sub_origins = np.hstack((ox, oz))
        
        # Direction:
        # tan(theta_water) is slope r/z.
        # z component of unit vector = cos(theta_water)
        # r component = sin(theta_water)
        # x = sin * u_r_x
        
        vz = np.cos(theta_water)
        vr = np.sin(theta_water)
        vx = vr[:, np.newaxis] * u_r
        sub_dirs = np.hstack((vx, vz[:, np.newaxis]))
        
        # Map back to full array
        # This is tricky with double masking (mask then valid).
        # Let's assume all passed snell check (usually true for Air->Water).
        # Simpler:
        final_mask = np.zeros(len(pts_pixel), dtype=bool)
        temp_idxs = np.where(mask)[0]
        final_idxs = temp_idxs[valid]
        
        origins_water[final_idxs] = sub_origins
        rays_water[final_idxs] = sub_dirs

    return origins_water, rays_water

def triangulate_rays(O1, D1, O2, D2):
    """
    Triangulates 3D points from skewed rays using Least Squares midpoint.
    O1, D1: Origins and Unit Directions for Camera 1 (N, 3)
    O2, D2: Origins and Unit Directions for Camera 2 (N, 3) (Already in Cam 1 Frame?)
    
    NOTE: D2, O2 must be in Global (Cam 1) Frame!
    """
    # Ray 1: P1 = O1 + t1 * D1
    # Ray 2: P2 = O2 + t2 * D2
    # Minimize || P1 - P2 ||^2
    # This is a standard line-line intersection problem.
    # Cross product approach or linear system.
    
    # Vector w0 = O1 - O2
    # a = D1 . D1 = 1
    # b = D1 . D2
    # c = D2 . D2 = 1
    # d = D1 . w0
    # e = D2 . w0
    
    # System:
    # t1 - b*t2 = -d
    # b*t1 - t2 = -e  => t2 = b*t1 + e
    # Subst: t1 - b(b*t1 + e) = -d
    # t1 (1 - b^2) = be - d
    # t1 = (be - d) / (1 - b^2)
    
    w0 = O1 - O2
    b = np.sum(D1 * D2, axis=1)
    d = np.sum(D1 * w0, axis=1)
    e = np.sum(D2 * w0, axis=1)
    
    denom = 1.0 - b*b
    # Avoid parallel lines (denom ~ 0)
    denom[np.abs(denom) < 1e-6] = 1e-6
    
    t1 = (b*e - d) / denom
    t2 = (e - b*d) / denom # Derived similarly
    
    # Points
    P1 = O1 + t1[:, np.newaxis] * D1
    P2 = O2 + t2[:, np.newaxis] * D2
    
    return (P1 + P2) / 2.0

def cor_maker_3d(df1_path, df2_path, camera_data_file, n_water=1.333, n_glass=1.5, d_air=0.0, d_glass=0.0, correct_refraction=False):
    """
    Generates 3D coordinates from matched stereo files.
    SAFE MODE: Enforces strict 1-to-1 matching per frame to prevent crashes.
    """
    print(f"Processing 3D reconstruction for {df1_path} and {df2_path}...")
    
    # 1. Load Data
    df1 = pd.read_csv(df1_path)
    df2 = pd.read_csv(df2_path)
    
    # 2. Safety: Remove duplicate IDs in the same frame (tracker glitches)
    df1 = df1.drop_duplicates(subset=['frame', 'id'], keep='first')
    df2 = df2.drop_duplicates(subset=['frame', 'id'], keep='first')
    
    # 3. Rename cols for merging
    df1 = df1.rename(columns={'x': 'x_1', 'y': 'y_1', 'w': 'w_1', 'h': 'h_1'})
    df2 = df2.rename(columns={'x': 'x_2', 'y': 'y_2', 'w': 'w_2', 'h': 'h_2'})
    
    # 4. Merge on Frame AND ID (Assuming ID matches, or use geometric matching input)
    # The input DFs here might be raw tracking OR matched.
    # If ProcessVideoPair calls this, it passes outputs from run_geometric_matching IF it saves them.
    # Wait, run_geometric_matching returns (df, df). ProcessVideoPair saves them to disk?
    # No, ProcessVideoPair calls cor_maker_3d with PATHS.
    # Checks ProcessVideoPair logic...
    
    # MERGE:
    merged = pd.merge(df1, df2, on=['frame', 'id'], how='inner')
    
    if merged.empty:
        print("No matching frames/IDs found for 3D reconstruction.")
        return pd.DataFrame()

    # 5. Load Camera Params
    mat = loadmat(camera_data_file)
    K1, D1 = mat['intrinsicMatrix1'], mat['distortionCoefficients1']
    K2, D2 = mat['intrinsicMatrix2'], mat['distortionCoefficients2']
    R, t = mat['rotationOfCamera2'], mat['translationOfCamera2'].flatten()
    
    # Project Matrices (Normalized Space)
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    # Removed K1 multiplication since triangulatePoints receives normalized points!
    
    # P2 = [R|t]
    P2 = np.hstack((R, t.reshape(3, 1)))
    # Removed K2 multiplication
    
    # 6. Process
    frames = merged['frame'].unique()
    results = []
    
    for frame in frames:
        subset = merged[merged['frame'] == frame]
        if subset.empty: continue
            
        pts1_pix = subset[['x_1', 'y_1']].values.astype(np.float32)
        pts2_pix = subset[['x_2', 'y_2']].values.astype(np.float32)
        ids = subset['id'].values
        
        # Get Rays (Origin, Direction)
        O1_local, D1_local = refract_points(pts1_pix, K1, D1, 
                                            n_water=n_water, n_glass=n_glass, d_air=d_air, d_glass=d_glass, 
                                            correct_refraction=correct_refraction)
        O2_local, D2_local = refract_points(pts2_pix, K2, D2, 
                                            n_water=n_water, n_glass=n_glass, d_air=d_air, d_glass=d_glass, 
                                            correct_refraction=correct_refraction)
        
        # HYBRID LOGIC: Check Shift Magnitude
        # Shift is the distance of the origin from (0,0,0).
        # We check the maximum shift in the batch.
        max_shift_1 = np.max(np.linalg.norm(O1_local, axis=1)) if len(O1_local) > 0 else 0
        max_shift_2 = np.max(np.linalg.norm(O2_local, axis=1)) if len(O2_local) > 0 else 0
        max_shift = max(max_shift_1, max_shift_2)
        
        # Threshold for switching to Ray Tracing (e.g., 20mm)
        # If shift is small, Pinhole Triangulation is "good enough" and faster/standard.
        RAY_TRACING_THRESHOLD_MM = 20.0 
        
        use_ray_tracing = False
        if correct_refraction:
            if max_shift > RAY_TRACING_THRESHOLD_MM:
                print(f"  [Refraction] Significant shift detected ({max_shift:.2f} mm > {RAY_TRACING_THRESHOLD_MM} mm). Using Ray Triangulation.")
                use_ray_tracing = True
            else:
                 # Optional: Warn if it's non-zero but we are ignoring it?
                 if max_shift > 1.0:
                     print(f"  [Refraction] Shift detected ({max_shift:.2f} mm) but below threshold. Using Pinhole Triangulation.")
        
        if not use_ray_tracing:
             # Standard Triangulation (Pinhole, Ray Origins assumed 0)
             # Use OpenCV optimized function
             
             # Note: If correct_refraction=True but shift is small, O1_local is non-zero but we IGNORE it.
             # We effectively project the ray back to (0,0,0) preserving its Direction.
             # D1_local IS the refracted direction.
             
             # We need (2, N) for OpenCV
             # D1_local is (N, 3). We need (u, v) in normalized plane (z=1).
             # u = x/z, v = y/z.
             
             z1 = D1_local[:, 2]
             z1[z1 == 0] = 1e-8
             pts1_norm_T = (D1_local[:, :2] / z1[:, np.newaxis]).T
             
             z2 = D2_local[:, 2]
             z2[z2 == 0] = 1e-8
             pts2_norm_T = (D2_local[:, :2] / z2[:, np.newaxis]).T
             
             points_4d = cv2.triangulatePoints(P1, P2, pts1_norm_T, pts2_norm_T)
             
             # Convert Homogeneous -> Euclidean (X/W, Y/W, Z/W)
             w = points_4d[3]
             w[w == 0] = 1e-10
             points_3d = points_4d[:3] / w
             points_3d = points_3d.T # (N, 3)
             
        else:
             # Advanced Ray Triangulation
             # Transform Camera 2 rays to Camera 1 Frame (Global)
             # P2_local = R_2to1 * P_global + t_2to1
             # P_global = R^T (P_local - t)
             # D_global = R^T * D_local
             
             # Origins:
             O2_global = (O2_local - t) @ R 
             
             # Directions:
             D2_global = D2_local @ R
             
             O1_global = O1_local # Cam1 is Origin
             D1_global = D1_local
             
             points_3d = triangulate_rays(O1_global, D1_global, O2_global, D2_global)

        for i, obj_id in enumerate(ids):
            results.append({
                'frame': frame,
                'id': int(obj_id),
                'x': points_3d[i, 0],
                'y': points_3d[i, 1],
                'z': points_3d[i, 2]
            })

    return pd.DataFrame(results)