
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def analyze_depth(csv_old, csv_new, label):
    print(f"\n--- Analyzing {label} ---")
    
    if not os.path.exists(csv_old):
        print(f"File not found: {csv_old}")
        return
    if not os.path.exists(csv_new):
        print(f"File not found: {csv_new}")
        return
        
    df_old = pd.read_csv(csv_old)
    df_new = pd.read_csv(csv_new)
    
    if df_old.empty or df_new.empty:
        print("One of the dataframes is empty.")
        return

    z_old = df_old['z'].values
    z_new = df_new['z'].values
    
    print(f"Old Method (Pinhole): N={len(z_old)}")
    print(f"  Mean Z: {np.mean(z_old):.2f} mm")
    print(f"  Median Z: {np.median(z_old):.2f} mm")
    print(f"  Std Z: {np.std(z_old):.2f} mm")
    print(f"  Min/Max: {np.min(z_old):.2f} / {np.max(z_old):.2f}")
    
    print(f"New Method (Ray-Ray): N={len(z_new)}")
    print(f"  Mean Z: {np.mean(z_new):.2f} mm")
    print(f"  Median Z: {np.median(z_new):.2f} mm")
    print(f"  Std Z: {np.std(z_new):.2f} mm")
    print(f"  Min/Max: {np.min(z_new):.2f} / {np.max(z_new):.2f}")
    
    diff_mean = np.mean(z_new) - np.mean(z_old)
    print(f"Shift: New methods puts points {diff_mean:.2f} mm {'FURTHER' if diff_mean > 0 else 'CLOSER'}")
    
    # Save Histogram plot
    plt.figure(figsize=(10, 6))
    plt.hist(z_old, bins=20, alpha=0.5, label='Old (Pinhole)', color='blue')
    plt.hist(z_new, bins=20, alpha=0.5, label='New (Ray-Ray)', color='orange')
    plt.xlabel('Depth Z (mm)')
    plt.ylabel('Count')
    plt.title(f'Depth Distribution Comparison - {label}')
    plt.legend(loc='upper right')
    
    out_img = f"comparison_{label}.png"
    plt.savefig(out_img)
    print(f"Saved histogram to {out_img}")

if __name__ == "__main__":
    analyze_depth("Process_Vid8_Old/3D_points.csv", "Process_Vid8_New/3D_points.csv", "Video8")
    analyze_depth("Process_Vid10_Old/3D_points.csv", "Process_Vid10_New/3D_points.csv", "Video10")
