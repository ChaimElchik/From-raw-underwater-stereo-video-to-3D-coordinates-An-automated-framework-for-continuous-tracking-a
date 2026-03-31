import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# Load comparison results from Final Pipeline Code
results_df = pd.read_csv('Final_Pipeline_Code/evaluation_results.csv')

# Ensure metrics are in percentage so they match the expected format (or just plot them as 0-1 if they are already)
if results_df['F1_Score'].max() <= 1.0:
    results_df['F1-Score'] = results_df['F1_Score']
    results_df['Precision (%)'] = results_df['Precision']
    results_df['Recall (%)'] = results_df['Recall']
else:
    results_df['F1-Score'] = results_df['F1_Score']
    results_df['Precision (%)'] = results_df['Precision']
    results_df['Recall (%)'] = results_df['Recall']

# Rename 'video' to 'Video' to match logic
results_df.rename(columns={'video': 'Video'}, inplace=True)
results_df['Version'] = 'Best Version (Final_Pipeline_Code)'

# Unique videos
videos = results_df['Video'].unique()

def get_gt_stats(video_id):
    mot_dir = 'mots'
    cam1_file = os.path.join(mot_dir, f"{video_id}_1_clean.txt")
    cam2_file = os.path.join(mot_dir, f"{video_id}_2_clean.txt")
    
    id_counts = []
    tracks_per_frame = []
    
    for f in [cam1_file, cam2_file]:
        if os.path.exists(f):
            try:
                # MOT format: frame, id, bb_left, bb_top, width, height, conf, x, y, z
                df = pd.read_csv(f, header=None)
                if len(df.columns) >= 2:
                    frame_col, id_col = 0, 1
                    # unique ids
                    n_ids = df[id_col].nunique()
                    id_counts.append(n_ids)
                    
                    # tracks per frame: number of rows per frame
                    tpf = df.groupby(frame_col).size().mean()
                    tracks_per_frame.append(tpf)
            except Exception as e:
                print(f"Error reading {f}: {e}")
                
    if id_counts:
        return np.mean(id_counts), np.mean(tracks_per_frame)
    else:
        return None, None

stats = []
for v in videos:
    n_ids, tpf = get_gt_stats(v)
    if n_ids is not None:
        stats.append({'Video': v, 'Avg_IDs': n_ids, 'Avg_Tracks_Per_Frame': tpf})

stats_df = pd.DataFrame(stats)
print("Abundance Stats:")
print(stats_df.describe())

print("\nMin IDs:", stats_df['Avg_IDs'].min())
print("Max IDs:", stats_df['Avg_IDs'].max())
print("Average IDs:", stats_df['Avg_IDs'].mean())

# Merge with results
merged = pd.merge(results_df, stats_df, on='Video')

plt.figure(figsize=(12, 4))
metrics = ['F1-Score', 'Precision (%)', 'Recall (%)']

versions = merged['Version'].unique()
colors = ['b']

for i, metric in enumerate(metrics):
    plt.subplot(1, 3, i+1)
    for j, version in enumerate(versions):
        v_data = merged[merged['Version'] == version].sort_values('Avg_IDs')
        v_data['Avg_IDs_jittered'] = v_data['Avg_IDs'] + np.random.uniform(-0.5, 0.5, size=len(v_data))
        plt.scatter(v_data['Avg_IDs'], v_data[metric], label=version, c=colors[j], alpha=0.7)
    plt.title(f"{metric} vs Amount of IDs")
    plt.xlabel("Average IDs per Video")
    plt.ylabel(metric)
    if i == 0:
        plt.legend()
plt.tight_layout()
plt.savefig('performance_vs_ids_final.png', dpi=300)
print("Saved plot to performance_vs_ids_final.png")

plt.figure(figsize=(12, 4))
for i, metric in enumerate(metrics):
    plt.subplot(1, 3, i+1)
    for j, version in enumerate(versions):
        v_data = merged[merged['Version'] == version].sort_values('Avg_Tracks_Per_Frame')
        plt.scatter(v_data['Avg_Tracks_Per_Frame'], v_data[metric], label=version, c=colors[j], alpha=0.7)
    plt.title(f"{metric} vs Tracks per Frame")
    plt.xlabel("Average Tracks per Frame")
    plt.ylabel(metric)
    if i == 0:
        plt.legend()
plt.tight_layout()
plt.savefig('performance_vs_tpf_final.png', dpi=300)
print("Saved plot to performance_vs_tpf_final.png")
