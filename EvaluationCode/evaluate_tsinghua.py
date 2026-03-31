import cv2
import numpy as np
import glob
import os
from scipy.spatial import cKDTree

from plyfile import PlyData

def read_ply(filename):
    plydata = PlyData.read(filename)
    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']
    return np.vstack([x, y, z]).T

def icp(source, target, max_iterations=50, tolerance=1e-5):
    if len(source) == 0: return source, float('inf')
    src = np.copy(source)
    target_tree = cKDTree(target)
    
    prev_error = float('inf')
    for i in range(max_iterations):
        distances, indices = target_tree.query(src)
        error = np.mean(distances)
        if abs(prev_error - error) < tolerance:
            break
        prev_error = error
        
        closest_points = target[indices]
        centroid_src = np.mean(src, axis=0)
        centroid_tgt = np.mean(closest_points, axis=0)
        H = (src - centroid_src).T @ (closest_points - centroid_tgt)
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1,:] *= -1
            R = Vt.T @ U.T
        t = centroid_tgt - R @ centroid_src
        src = (R @ src.T).T + t
        
    return src, prev_error

# Load calibration
def get_node(filepath, node_name):
    fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_READ)
    val = fs.getNode(node_name).mat()
    fs.release()
    return val

calib_dir = '/Users/chaim/Documents/Thesis_ReWrite/EvaluationCode/underwater-datasets-main/calib_result/'
K1 = get_node(calib_dir + 'matlab_cameraMatrixL.xml', 'matlab_cameraMatrixL')
K2 = get_node(calib_dir + 'matlab_cameraMatrixR.xml', 'matlab_cameraMatrixR')
D1 = get_node(calib_dir + 'matlab_distCoeffL.xml', 'matlab_distCoeffL')
D2 = get_node(calib_dir + 'matlab_distCoeffR.xml', 'matlab_distCoeffR')
R_stereo = get_node(calib_dir + 'matlab_R.xml', 'matlab_R')
T_stereo = get_node(calib_dir + 'matlab_T.xml', 'matlab_T').flatten()

P1_pinhole = K1 @ np.hstack((np.eye(3), np.zeros((3,1))))
P2_pinhole = K2 @ np.hstack((R_stereo, T_stereo.reshape(3,1)))

# Use Pipeline logic
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'PipeLine'))
try:
    from ThreeDCordinate_Maker import refract_points, triangulate_rays
except ImportError as e:
    print(f"Could not import pipeline functions from PipeLine. Error: {e}")
    sys.exit(1)

sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

mesh_dir = '/Users/chaim/Documents/Thesis_ReWrite/EvaluationCode/underwater-datasets-main/mesh/'
img_pairs = [
    ('fish1', '1.bmp'),
    ('shell', '2.bmp'),
    ('startfish', '3.bmp'),
    ('stone1', '4.bmp'),
    ('stone2', '5.bmp')
]

# We will use the extracted optimal (d_air=50, d_glass=1) or (d_air=15, d_glass=10)
# to see which is better! Let's test 15/10.
d_air = 15.0
d_glass = 10.0

print(f"{'Object':<10} | {'Matches':<8} | {'Pinhole RMSE':<15} | {'Refraction RMSE':<15}")
print("-" * 55)

for mesh_name, img_name in img_pairs:
    l_path = f'/Users/chaim/Documents/Thesis_ReWrite/EvaluationCode/underwater-datasets-main/image/left/{img_name}'
    r_path = f'/Users/chaim/Documents/Thesis_ReWrite/EvaluationCode/underwater-datasets-main/image/right/{img_name}'
    mesh_path = f'{mesh_dir}{mesh_name}.ply'
    
    imgL = cv2.imread(l_path, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(r_path, cv2.IMREAD_GRAYSCALE)
    
    kp1, des1 = sift.detectAndCompute(imgL, None)
    kp2, des2 = sift.detectAndCompute(imgR, None)
    
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
            
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    
    # 1. Pinhole Triangulation
    pts1_norm = cv2.undistortPoints(np.expand_dims(pts1, 1), K1, D1).reshape(-1, 2).T
    pts2_norm = cv2.undistortPoints(np.expand_dims(pts2, 1), K2, D2).reshape(-1, 2).T
    P4D = cv2.triangulatePoints(P1_pinhole, P2_pinhole, pts1_norm, pts2_norm)
    pts3D_pinhole = (P4D[:3] / P4D[3]).T
    
    # Filter depth outliers > 5000 mm or < 100 mm
    valid_mask = (pts3D_pinhole[:, 2] > 100) & (pts3D_pinhole[:, 2] < 5000)
    pts3D_pinhole = pts3D_pinhole[valid_mask]
    pts1_r = pts1[valid_mask]
    pts2_r = pts2[valid_mask]
    
    # 2. Refractive Triangulation
    O1, D1_vec = refract_points(pts1_r, K1, D1, d_air=d_air, d_glass=d_glass, correct_refraction=True, n_glass=1.49)
    O2, D2_vec = refract_points(pts2_r, K2, D2, d_air=d_air, d_glass=d_glass, correct_refraction=True, n_glass=1.49)
    O2_global = (O2 - T_stereo) @ R_stereo
    D2_global = D2_vec @ R_stereo
    pts3D_refract = triangulate_rays(O1, D1_vec, O2_global, D2_global)
    
    # ICP
    mesh_pts = read_ply(mesh_path)
    if len(mesh_pts) == 0:
        continue
        
    # Standardize mesh scale? The mesh is supposedly in arbitrary or Kinect mm scale. ICP handles rigid transform.
    # Kinect mesh might be in meters! So let's scale it to matched points magnitude.
    # Actually, rigid ICP does NOT scale! If the scales don't match, we need Procrustes or scale-ICP.
    # We will compute the scale difference first:
    mesh_size = np.max(mesh_pts, axis=0) - np.min(mesh_pts, axis=0)
    pinhole_size = np.max(pts3D_pinhole, axis=0) - np.min(pts3D_pinhole, axis=0)
    
    # If the scales are off by ~1000, we apply scaling to the mesh.
    if np.mean(pinhole_size) > 100 * np.mean(mesh_size):
        mesh_pts *= 1000.0 # m to mm
        
    _, err_p = icp(pts3D_pinhole, mesh_pts)
    _, err_r = icp(pts3D_refract, mesh_pts)
    
    print(f"{mesh_name:<10} | {len(pts1_r):<8} | {err_p:<15.4f} | {err_r:<15.4f}")
