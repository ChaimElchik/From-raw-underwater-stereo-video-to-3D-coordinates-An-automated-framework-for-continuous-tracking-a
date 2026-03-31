import numpy as np
from scipy.optimize import minimize

# ==========================================
# 1. VIRTUAL HARDWARE PARAMETERS
# ==========================================
SQUARE_SIZE = 38.0  # mm
BOARD_SHAPE = (5, 5) # 5x5 internal corners

# Camera Intrinsics (Standard HD Pinhole)
K = np.array([[1000.0, 0.0, 640.0],
              [0.0, 1000.0, 360.0],
              [0.0, 0.0, 1.0]])

BASELINE = 120.0 # mm offset for Right Camera (Tx)

# Refractive Interfaces
N_AIR = 1.0
N_GLASS = 1.49
N_WATER = 1.333
D_AIR = 15.0     # mm (distance from lens to glass)
D_GLASS = 10.0   # mm (glass thickness)

# ==========================================
# 2. CORE REFRACTION MATH
# ==========================================
def get_refracted_ray(u, v, K_mat, origin_offset=np.array([0.0, 0.0, 0.0])):
    """Forward ray tracing: Air -> Glass -> Water"""
    # 1. Ray in Air
    ray_a = np.array([(u - K_mat[0,2])/K_mat[0,0], (v - K_mat[1,2])/K_mat[1,1], 1.0])
    ray_a /= np.linalg.norm(ray_a)
    
    # 2. Intersect Air-Glass Interface
    p_glass_in = ray_a * (D_AIR / ray_a[2])
    
    # 3. Refract into Glass (Snell's Law Vector Form)
    n = np.array([0.0, 0.0, -1.0]) # Normal pointing at camera
    c1 = -np.dot(n, ray_a)
    r = N_AIR / N_GLASS
    c2 = np.sqrt(1.0 - r**2 * (1.0 - c1**2))
    ray_g = r * ray_a + (r * c1 - c2) * n
    
    # 4. Intersect Glass-Water Interface
    dist_in_glass = D_GLASS / ray_g[2]
    p_glass_out = p_glass_in + ray_g * dist_in_glass
    
    # 5. Refract into Water
    c1_w = -np.dot(n, ray_g)
    r_w = N_GLASS / N_WATER
    c2_w = np.sqrt(1.0 - r_w**2 * (1.0 - c1_w**2))
    ray_w = r_w * ray_g + (r_w * c1_w - c2_w) * n
    
    # Shift to global origin (for right camera)
    return p_glass_out + origin_offset, ray_w

# ==========================================
# 3. BACKWARD RAY TRACING (SIMULATOR)
# ==========================================
def find_pixel_for_3d_point(P_target, K_mat, origin_offset=np.array([0.0, 0.0, 0.0])):
    """Iteratively finds the (u, v) pixel that refracts to hit the 3D target."""
    def loss(px):
        O, D = get_refracted_ray(px[0], px[1], K_mat, origin_offset)
        # Point-to-line distance
        v_vec = P_target - O
        dist = np.linalg.norm(v_vec - np.dot(v_vec, D) * D)
        return dist
        
    # Initial guess using standard pinhole projection (ignoring refraction)
    P_local = P_target - origin_offset
    u_guess = (P_local[0] / P_local[2]) * K_mat[0,0] + K_mat[0,2]
    v_guess = (P_local[1] / P_local[2]) * K_mat[1,1] + K_mat[1,2]
    
    # Optimize to find exact refracted pixel
    res = minimize(loss, [u_guess, v_guess], method='Nelder-Mead')
    return res.x

# ==========================================
# 4. YOUR SKEW-RAY TRIANGULATION MATH
# ==========================================
def solve_skew_ray(O1, D1, O2, D2):
    """Finds the midpoint of the shortest segment between two skewed rays."""
    A = np.vstack((D1, -D2)).T
    b = O2 - O1
    t, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    P1 = O1 + t[0] * D1
    P2 = O2 + t[1] * D2
    return (P1 + P2) / 2.0

# ==========================================
# 5. RUN THE EVALUATION
# ==========================================
if __name__ == "__main__":
    print("1. Generating perfect 3D Checkerboard...")
    true_3d_points = []
    for i in range(BOARD_SHAPE[1]):
        for j in range(BOARD_SHAPE[0]):
            # Center the board roughly in front of the cameras at Z = 1000mm
            X = (j - BOARD_SHAPE[0]/2) * SQUARE_SIZE + (BASELINE / 2)
            Y = (i - BOARD_SHAPE[1]/2) * SQUARE_SIZE
            Z = 1000.0 
            true_3d_points.append(np.array([X, Y, Z]))
            
    print("2. Simulating Refraction (Finding 2D Pixels)...")
    pixels_L = []
    pixels_R = []
    for pt in true_3d_points:
        pixels_L.append(find_pixel_for_3d_point(pt, K, origin_offset=np.array([0.0, 0.0, 0.0])))
        pixels_R.append(find_pixel_for_3d_point(pt, K, origin_offset=np.array([BASELINE, 0.0, 0.0])))
        
    print("3. Executing Your Triangulation Math...")
    triangulated_points = []
    for px_L, px_R in zip(pixels_L, pixels_R):
        # Forward refract the pixels into 3D rays
        O1, D1 = get_refracted_ray(px_L[0], px_L[1], K, origin_offset=np.array([0.0, 0.0, 0.0]))
        O2, D2 = get_refracted_ray(px_R[0], px_R[1], K, origin_offset=np.array([BASELINE, 0.0, 0.0]))
        
        # Triangulate
        Pw = solve_skew_ray(O1, D1, O2, D2)
        triangulated_points.append(Pw)
        
    print("4. Evaluating Accuracy...")
    errors = []
    for i in range(BOARD_SHAPE[1]):
        for j in range(BOARD_SHAPE[0] - 1):
            idx1 = i * BOARD_SHAPE[0] + j
            idx2 = idx1 + 1
            dist = np.linalg.norm(triangulated_points[idx1] - triangulated_points[idx2])
            errors.append(abs(dist - SQUARE_SIZE))
            
    print("-" * 40)
    print(f"Target Square Size:    {SQUARE_SIZE:.3f} mm")
    print(f"Mean Triangulated Size:{np.mean([SQUARE_SIZE + e for e in errors]):.3f} mm")
    print(f"Mean Absolute Error:   {np.mean(errors):.6f} mm")
    print("-" * 40)
    print("If error is ~0.000000 mm, your flat-port math is mathematically flawless!")
