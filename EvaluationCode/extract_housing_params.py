import cv2
import glob
import numpy as np
from scipy.optimize import minimize

def get_node_data(filepath, node_name):
    fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_READ)
    node = fs.getNode(node_name)
    val = node.mat()
    fs.release()
    return val

cameraMatrixL = get_node_data('/Users/chaim/Documents/Thesis_ReWrite/underwater-datasets-main/calib_result/matlab_cameraMatrixL.xml', 'matlab_cameraMatrixL')
cameraMatrixR = get_node_data('/Users/chaim/Documents/Thesis_ReWrite/underwater-datasets-main/calib_result/matlab_cameraMatrixR.xml', 'matlab_cameraMatrixR')
distCoeffL = get_node_data('/Users/chaim/Documents/Thesis_ReWrite/underwater-datasets-main/calib_result/matlab_distCoeffL.xml', 'matlab_distCoeffL')
distCoeffR = get_node_data('/Users/chaim/Documents/Thesis_ReWrite/underwater-datasets-main/calib_result/matlab_distCoeffR.xml', 'matlab_distCoeffR')
R_stereo = get_node_data('/Users/chaim/Documents/Thesis_ReWrite/underwater-datasets-main/calib_result/matlab_R.xml', 'matlab_R')
T_stereo = get_node_data('/Users/chaim/Documents/Thesis_ReWrite/underwater-datasets-main/calib_result/matlab_T.xml', 'matlab_T')
# T_stereo in XML is 3x1 vector

def refract_points(pts_pixel, K, D, n_air=1.0, n_water=1.333, n_glass=1.49, d_air=0.0, d_glass=0.0):
    if len(pts_pixel) == 0:
        return np.array([]), np.array([])

    pts_norm = cv2.undistortPoints(np.expand_dims(pts_pixel, 1), K, D).squeeze(1)
    rays_air = np.column_stack((pts_norm, np.ones(len(pts_norm))))
    norms = np.linalg.norm(rays_air, axis=1, keepdims=True)
    rays_air /= norms 
    
    r_air = np.linalg.norm(pts_norm, axis=1) 
    mask = r_air > 1e-8
    
    rays_water = rays_air.copy()
    origins_water = np.zeros_like(rays_air)
    
    if np.any(mask):
        theta_air = np.arctan(r_air[mask])
        sin_theta_water = (n_air / n_water) * np.sin(theta_air)
        valid = np.abs(sin_theta_water) <= 1.0
        theta_water = np.arcsin(sin_theta_water[valid])
        
        sin_theta_glass = (n_air / n_glass) * np.sin(theta_air[valid])
        theta_glass = np.arcsin(sin_theta_glass)
        
        r_exit = d_air * np.tan(theta_air[valid]) + d_glass * np.tan(theta_glass)
        z_exit = d_air + d_glass
        
        u_r = pts_norm[mask][valid] / r_air[mask][valid][:, np.newaxis]
        
        ox = r_exit[:, np.newaxis] * u_r
        oz = np.full((len(ox), 1), z_exit)
        
        origins_water[mask] = 0 
        
        sub_origins = np.hstack((ox, oz))
        
        vz = np.cos(theta_water)
        vr = np.sin(theta_water)
        vx = vr[:, np.newaxis] * u_r
        sub_dirs = np.hstack((vx, vz[:, np.newaxis]))
        
        origins_water[mask] = sub_origins  # Because "valid" is all elements (Snell check almost always passes)
        rays_water[mask] = sub_dirs

    return origins_water, rays_water

def triangulate_rays(O1, D1, O2, D2):
    w0 = O1 - O2
    b = np.sum(D1 * D2, axis=1)
    d = np.sum(D1 * w0, axis=1)
    e = np.sum(D2 * w0, axis=1)
    
    denom = 1.0 - b*b
    denom[np.abs(denom) < 1e-6] = 1e-6
    
    t1 = (b*e - d) / denom
    t2 = (e - b*d) / denom 
    
    P1 = O1 + t1[:, np.newaxis] * D1
    P2 = O2 + t2[:, np.newaxis] * D2
    
    return (P1 + P2) / 2.0

# Extract corners from a few images
left_images = sorted(glob.glob('/Users/chaim/Documents/Thesis_ReWrite/underwater-datasets-main/calib_image/left/*.bmp'))
right_images = sorted(glob.glob('/Users/chaim/Documents/Thesis_ReWrite/underwater-datasets-main/calib_image/right/*.bmp'))

pattern_size = (8, 7)
pts_list_L = []
pts_list_R = []

# Use up to 10 image pairs for optimization
for i, (l, r) in enumerate(zip(left_images, right_images)):
    if i >= 10: break
    imgL = cv2.imread(l, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(r, cv2.IMREAD_GRAYSCALE)
    
    ret_l, corners_l = cv2.findChessboardCorners(imgL, pattern_size, None)
    ret_r, corners_r = cv2.findChessboardCorners(imgR, pattern_size, None)

    if ret_l and ret_r:
        corners_l = cv2.cornerSubPix(imgL, corners_l, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        corners_r = cv2.cornerSubPix(imgR, corners_r, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        pts_list_L.append(corners_l.reshape(-1, 2))
        pts_list_R.append(corners_r.reshape(-1, 2))

print(f"Loaded {len(pts_list_L)} stereo pairs for optimization.")

expected_side = 19.8003 # mm

def objective(x):
    d_air, d_glass = x
    error_sum = 0
    err_count = 0
    
    for pts_L, pts_R in zip(pts_list_L, pts_list_R):
        O1_local, D1_local = refract_points(pts_L, cameraMatrixL, distCoeffL, d_air=d_air, d_glass=d_glass)
        O2_local, D2_local = refract_points(pts_R, cameraMatrixR, distCoeffR, d_air=d_air, d_glass=d_glass)
        
        # P_1 = R^T (P_2 - T) -> Row vecs: P_1.T = (P_2 - T).T * R
        O2_global = (O2_local - T_stereo.flatten()) @ R_stereo
        D2_global = D2_local @ R_stereo
        
        P3D = triangulate_rays(O1_local, D1_local, O2_global, D2_global)
        
        # Calculate expected spacing
        P3D_grid = P3D.reshape(7, 8, 3)
        diff_h = np.linalg.norm(P3D_grid[:, 1:, :] - P3D_grid[:, :-1, :], axis=2)
        diff_v = np.linalg.norm(P3D_grid[1:, :, :] - P3D_grid[:-1, :, :], axis=2)
        
        errs = np.concatenate((diff_h.flatten(), diff_v.flatten()))
        error_sum += np.sum((errs - expected_side)**2)
        err_count += len(errs)
        
    return error_sum / max(1, err_count)

# Bounds for d_air and d_glass
bounds = ((0, 200), (0.1, 100))
x0 = np.array([10.0, 5.0]) # Initial guess

print("Starting optimization. Initial MSE:", objective(x0))
res = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')

print("Optimization result:", res.message)
print(f"Optimal d_air = {res.x[0]:.4f} mm, d_glass = {res.x[1]:.4f} mm")
print("Final MSE:", objective(res.x))
