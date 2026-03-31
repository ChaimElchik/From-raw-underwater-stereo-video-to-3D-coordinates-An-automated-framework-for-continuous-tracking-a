import pandas as pd
import numpy as np
import scipy.io
from scipy.optimize import linear_sum_assignment

def run_geometric_matching(file1_path, file2_path, mat_path):
    # 1. Load Camera Parameters
    mat_data = scipy.io.loadmat(mat_path)
    K1, K2 = mat_data['intrinsicMatrix1'], mat_data['intrinsicMatrix2']
    R, t = mat_data['rotationOfCamera2'], mat_data['translationOfCamera2'].flatten()

    # 2. Compute the Global Fundamental Matrix (F)
    t_skew = np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])
    E = t_skew @ R
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)

    # 3. Load Tracking Data
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)
    common_frames = sorted(list(set(df1['frame']) & set(df2['frame'])))
    
    # 4. Matching functions
    def get_epipolar_line(p, F_mat):
        return F_mat @ np.array([p[0], p[1], 1.0])

    def point_to_line_dist(p, line):
        a, b, c = line
        return abs(a * p[0] + b * p[1] + c) / np.sqrt(a**2 + b**2)

    # 5. Iterate through frames
    all_frame_matches = []
    
    for frame in common_frames:
        pts1 = df1[df1['frame'] == frame]
        pts2 = df2[df2['frame'] == frame]
        
        ids1, ids2 = pts1['id'].values, pts2['id'].values
        coords1, coords2 = pts1[['x', 'y']].values, pts2[['x', 'y']].values
        
        n1, n2 = len(ids1), len(ids2)
        if n1 == 0 or n2 == 0: continue

        cost_matrix = np.zeros((n1, n2))
        for i in range(n1):
            line = get_epipolar_line(coords1[i], F)
            for j in range(n2):
                cost_matrix[i, j] = point_to_line_dist(coords2[j], line)
        
        # Hungarian Algorithm forces a 1-to-1 match for the frame
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        for r, c in zip(row_ind, col_ind):
            all_frame_matches.append({
                'id1': ids1[r], 
                'id2': ids2[c], 
                'dist': cost_matrix[r, c]
            })

    # 6. Consensus Stats
    match_df = pd.DataFrame(all_frame_matches)
    consensus = match_df.groupby(['id1', 'id2']).agg(
        count=('dist', 'count'),
        avg_dist=('dist', 'mean')
    ).reset_index()

    # 7. Intelligent Sorting & Greedy Assignment
    # Prioritize: 
    #   1. High Frequency (count)
    #   2. Low Error (avg_dist)
    
    # Sort all potential candidates by quality
    consensus = consensus.sort_values(
        by=['count', 'avg_dist'], 
        ascending=[False, True]
    )
    
    final_matches = []
    assigned_1 = set()
    assigned_2 = set()
    
    # Iterate through the sorted candidates.
    # This naturally implements the logic:
    # - The "Best" global options get picked first.
    # - If ID1's best option is taken, it is skipped here.
    # - ID1 will eventually appear again in the list with its *second best* option.
    # - If that is available, it gets assigned. If not, it falls to the third, etc.
    for _, row in consensus.iterrows():
        i1, i2 = row['id1'], row['id2']
        if i1 not in assigned_1 and i2 not in assigned_2:
            final_matches.append(row)
            assigned_1.add(i1)
            assigned_2.add(i2)
            
    best_mapping = pd.DataFrame(final_matches)
    
    # 8. Create Renamed DataFrame
    mapping_dict = dict(zip(best_mapping['id2'], best_mapping['id1']))
    df2_renamed = df2.copy()
    df2_renamed['id'] = df2_renamed['id'].map(mapping_dict)
    
    return best_mapping, df2_renamed

if __name__ == "__main__":
    mapping, renamed_df = run_geometric_matching(
        'mots/406_1_clean.txt', 
        'mots/406_2_clean.txt', 
        'stereoParams_Dep4.mat'
    )
    
    print("Final ID Mapping (Iterative Best Fit for Losers):")
    print(mapping[['id1', 'id2', 'count', 'avg_dist']].sort_values('id1'))
    
    renamed_df.to_csv('406_2_synchronized.csv', index=False)