import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import os
import cv2

# --- ANALYSIS FUNCTIONS ---

def Fish_Trajectories_Graphs(df, output_dir):
    for fish_id, group in df.groupby('id'):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(group['x'], group['y'], group['z'], label=f'Fish ID: {fish_id}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Fish ID: {fish_id} Trajectory')
        ax.legend()
        plt.savefig(os.path.join(output_dir, f"{fish_id}_Fish_Trajectory_plot.png"))
        plt.close()

def Fish_Speed_Graphs(df, output_dir):
    for fish_id, group in df.groupby('id'):
        plt.figure()
        plt.plot(group['frame'], group['speed'], label=f'Fish ID: {fish_id}')
        plt.xlabel('Frame')
        plt.ylabel('Speed (units/frame)')
        plt.title(f'Speed Analysis - Fish ID: {fish_id}')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"{fish_id}_Fish_Speed_plot.png"))
        plt.close()

def Fish_Acceleration_Graphs(df, output_dir):
    for fish_id, group in df.groupby('id'):
        plt.figure()
        plt.plot(group['frame'], group['acceleration'], label=f'Fish ID: {fish_id}')
        plt.xlabel('Frame')
        plt.ylabel('Acceleration')
        plt.title(f'Acceleration Analysis - Fish ID: {fish_id}')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"{fish_id}_Fish_Acceleration_plot.png"))
        plt.close()

def Acceleration_Statistics(df, output_dir):
    acceleration_stats = df.groupby('id')['acceleration'].describe()
    with open(os.path.join(output_dir, "acceleration_analysis.txt"), "w") as f:
        f.write("Summary statistics of acceleration for each fish ID:\n")
        f.write(acceleration_stats.to_string())

def Fish_Path_length_Graph(df, output_dir):
    path_lengths = df.groupby('id')['speed'].sum()
    plt.figure()
    plt.bar(path_lengths.index, path_lengths.values)
    plt.xlabel('Fish ID')
    plt.ylabel('Total Path Length')
    plt.title('Path Length Analysis')
    plt.savefig(os.path.join(output_dir, "Fish_Path_Length_Plot.png"))
    plt.close()

def AVG_Fish_Path_Length(df, output_dir):
    path_lengths = df.groupby('id')['speed'].sum()
    mean_path_length = str(path_lengths.mean())
    
    with open(os.path.join(output_dir, "AVG_Fish_Path_Length.txt"), "w") as f:
        f.write("Average Path Length across all fish:\n")
        f.write(mean_path_length)

def Heatmap_of_Fish_Density_Graph(df, output_dir):
    plt.figure()
    plt.hist2d(df['x'], df['y'], bins=50, cmap='hot')
    plt.colorbar(label='Density')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Heatmap of Fish Density (Top Down)')
    plt.savefig(os.path.join(output_dir, "Heatmap_of_Fish_Density_Graph.png"))
    plt.close()

def Temporal_Patterns_Of_Movement(df, output_dir, fps):
    time_interval_seconds = 10 
    df['time_interval'] = (df['time_seconds'] // time_interval_seconds) * time_interval_seconds
    
    mean_speed_by_interval = df.groupby('time_interval')['speed'].mean()

    plt.figure()
    plt.plot(mean_speed_by_interval.index, mean_speed_by_interval.values)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Mean Speed of Population')
    plt.title('Temporal Patterns of Movement')
    plt.savefig(os.path.join(output_dir, "Temporal_Patterns_Of_Movement_Graph.png"))
    plt.close()

def Fish_speeds_changes(df, output_dir):
    df['speed_diff'] = df.groupby('id')['speed'].diff()
    threshold = 0.5
    increased_speed_frames = df[df['speed_diff'] > threshold]

    with open(os.path.join(output_dir, "Fish_speeds_changes.txt"), "w") as f:
        for fish_id, group in increased_speed_frames.groupby('id'):
            f.write(f"Fish ID {fish_id}: Frames with speed increase > {threshold} - {group['frame'].tolist()}\n")

def Spatial_Distribution(df, output_dir):
    plt.figure()
    sns.scatterplot(x='x', y='y', data=df, hue='id', palette='tab10')
    plt.title("Spatial Distribution (Top Down)")
    plt.savefig(os.path.join(output_dir, "Spatial_Distribution_Graph.png"))
    plt.close()

def Depth_Analysis(df, output_dir):
    plt.figure()
    for fish_id, group in df.groupby('id'):
        plt.plot(group['frame'], group['z'], label=f'Fish {fish_id}')

    plt.xlabel('Frame')
    plt.ylabel('Depth (Z)')
    plt.title('Depth Trajectories for Each Fish')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "Depth_Analysis_Graph.png"))
    plt.close()

# --- MAIN EXECUTION ---

def get_video_fps(video_path):
    """Attempt to read the exact FPS from the video file using OpenCV."""
    if video_path and os.path.exists(video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if fps > 0:
                print(f"Detected video FPS: {fps:.2f}")
                return fps
        except Exception as e:
            print(f"Warning: Could not read video FPS: {e}")
            
    # Fallback default if video reading fails or video_path is None
    print("Using default FPS: 240.0")
    return 240.0

def run_all_analysis(input_csv_path, output_folder, video_path=None):
    """
    Wrapper to run all analysis functions on the 3D data.
    """
    print(f"Loading 3D data from: {input_csv_path}")
    if not os.path.exists(input_csv_path):
        print("Error: Input file does not exist.")
        return

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    # Load Data
    df = pd.read_csv(input_csv_path)

    # Validate Columns
    required_cols = ['id', 'x', 'y', 'z', 'frame']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV missing required columns. Found: {df.columns}")
        print(f"Expected: {required_cols}")
        return

    # 0. Pre-processing core metrics once
    print("Pre-calculating velocities and acceleration...")
    
    # Sort by ID and Frame to ensure diff() calculations are correct
    df = df.sort_values(by=['id', 'frame']).reset_index(drop=True)

    # Calculate Speed (distance traveled per frame)
    df['speed'] = ((df.groupby('id')['x'].diff()**2 + 
                    df.groupby('id')['y'].diff()**2 + 
                    df.groupby('id')['z'].diff()**2) ** 0.5).fillna(0)
                    
    # Calculate Acceleration (change in speed per frame)
    df['acceleration'] = df.groupby('id')['speed'].diff().fillna(0)
    
    # Calculate Time in seconds statically based on dynamic FPS discovery
    fps = get_video_fps(video_path)
    df['time_seconds'] = df['frame'] / fps
    
    # Save the consolidated pre-processed dataframe
    preprocessed_path = os.path.join(output_folder, "Preprocessed_Trajectory_Data.csv")
    df.to_csv(preprocessed_path, index=False)
    print(f"Saved pre-processed data with kinematics to: {preprocessed_path}")

    print("Running Analysis Graphs...")
    
    # 1. Trajectories
    Fish_Trajectories_Graphs(df, output_folder)
    print("- Trajectories plotted.")

    # 2. Kinematics (Speed/Accel)
    Fish_Speed_Graphs(df, output_folder)
    Fish_Acceleration_Graphs(df, output_folder)
    Acceleration_Statistics(df, output_folder)
    Fish_speeds_changes(df, output_folder)
    print("- Kinematics analyzed.")

    # 3. Spatial & Temporal
    Fish_Path_length_Graph(df, output_folder)
    AVG_Fish_Path_Length(df, output_folder)
    Heatmap_of_Fish_Density_Graph(df, output_folder)
    Spatial_Distribution(df, output_folder)
    Depth_Analysis(df, output_folder)
    Temporal_Patterns_Of_Movement(df, output_folder, fps)
    print("- Spatial/Temporal analysis complete.")
    
    print(f"Done! All results saved to: {output_folder}")