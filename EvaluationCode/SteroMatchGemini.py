import pandas as pd
import numpy as np
import scipy.io
from scipy.optimize import linear_sum_assignment

def run_geometric_matching(file1_path, file2_path, mat_path):
    # 1. Load Camera Parameters
    mat_data = scipy.io.loadmat(mat_path)
    K1 = mat_data['intrinsicMatrix1']
    K2 = mat_data['intrinsicMatrix2']
    R = mat_data['rotationOfCamera2']
    t = mat_data['translationOfCamera2'].flatten()

    # 2. Compute the Global Fundamental Matrix (F)
    # F = K2^-T * [t]_x * R * K1^-1
    t_skew = np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])
    E = t_skew @ R
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)

    # 3. Load and Pre-process Tracking Data
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # Find frames common to both files
    common_frames = sorted(list(set(df1['frame']) & set(df2['frame'])))
    
    # 4. Matching functions
    def get_epipolar_line(p, F_mat):
        """Returns the coefficients (a, b, c) of the line ax + by + c = 0"""
        return F_mat @ np.array([p[0], p[1], 1.0])

    def point_to_line_dist(p, line):
        """Calculates perpendicular distance from point p to line"""
        a, b, c = line
        return abs(a * p[0] + b * p[1] + c) / np.sqrt(a**2 + b**2)

    # 5. Iterate through frames and match IDs
    all_frame_matches = []

    for frame in common_frames:
        pts1 = df1[df1['frame'] == frame]
        pts2 = df2[df2['frame'] == frame]
        
        ids1 = pts1['id'].values
        ids2 = pts2['id'].values
        coords1 = pts1[['x', 'y']].values
        coords2 = pts2[['x', 'y']].values
        
        n1, n2 = len(ids1), len(ids2)
        if n1 == 0 or n2 == 0: continue

        # Build Cost Matrix (Distances to Epipolar Lines)
        cost_matrix = np.zeros((n1, n2))
        for i in range(n1):
            line = get_epipolar_line(coords1[i], F)
            for j in range(n2):
                cost_matrix[i, j] = point_to_line_dist(coords2[j], line)
        
        # Use Hungarian Algorithm for optimal assignment in this frame
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        for r, c in zip(row_ind, col_ind):
            all_frame_matches.append({
                'id1': ids1[r], 
                'id2': ids2[c], 
                'dist': cost_matrix[r, c]
            })

    # 6. Global Consensus (Majority Vote)
    match_df = pd.DataFrame(all_frame_matches)
    
    # Count occurrences of each (id1, id2) pair
    consensus = match_df.groupby(['id1', 'id2']).size().reset_index(name='count')
    
    # For each ID in File 1, find the most frequent corresponding ID in File 2
    final_mapping = consensus.sort_values(['id1', 'count'], ascending=[True, False])
    best_mapping = final_mapping.groupby('id1').head(1)

    # 7. Create Renamed DataFrame
    mapping_dict = dict(zip(best_mapping['id2'], best_mapping['id1']))
    df2_renamed = df2.copy()
    df2_renamed['id'] = df2_renamed['id'].map(mapping_dict)
    
    return best_mapping, df2_renamed

# Execution
if __name__ == "__main__":
    mapping, renamed_df = run_geometric_matching(
        'mots/129_1_clean.txt', 
        'mots/129_2_clean.txt', 
        'stereoParams_Dep1.mat'
    )
    
    print("Final ID Mapping (File 1 ID -> File 2 ID):")
    print(mapping)
    
    # Save the synchronized result
    renamed_df.to_csv('129__synchronized.csv', index=False)