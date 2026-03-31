# 3D Fish Tracking & Re-ID Pipeline

A robust, high-precision 3D tracking framework designed to analyze fish movement in stereo underwater video datasets. This pipeline integrates object detection, multi-object tracking, visual Re-Identification (Re-ID) using Vision Transformers, Refractive Stereo Matching, and 3D Triangulation into an automated workflow.

## Features

- **2D Detection & Tracking**: Utilizes YOLO object detection paired with BoT-SORT for resilient temporal tracking.
- **Advanced Visual Re-ID**: A custom Vision Transformer (ViT) module that "heals" fragmented tracks by extracting visual embeddings and intelligently matching broken tracklets across time, factoring in temporal and kinematic constraints.
- **Refractive Stereo Matching**: Epipolar geometric matching module that handles complex **Air-Glass-Water** refractive interfaces, drastically improving 3D mapping accuracy over standard pinhole camera assumption.
- **3D Triangulation**: Computes metric 3D coordinates (X, Y, Z) from matched 2D stereo trajectories.
- **Visualization**: Generates custom annotated overlay videos natively.

## Implementation Overview

The main execution sequence is handled by `Run_PipeLine.py`, divided into 4 clear steps:
1. **Detection & Tracking** (`ProcessVideoPair.py` and `AdvancedReID.py`): Performs YOLO detection followed by tracking and Re-ID to generate temporally consistent 2D IDs.
2. **Refractive Epipolar Matching** (`StereoMatching.py`): Matches tracks between the left and right cameras using epipolar geometry and refractive correction.
3. **3D Triangulation** (`ThreeDCordinate_Maker.py`): Reconstructs 3D real-world coordinates from stereo matching results.
4. **Visualization Overlay** (`OutPutVideoGenerater.py`): Annotates original video files with final tracking data.

## Installation

Ensure you have a working Python environment (Python 3.8+ recommended).

1. Clone this repository.
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

*Note: The Re-ID and YOLO modules will automatically leverage any available NVIDIA GPUs (CUDA) or Apple Silicon GPUs (MPS) for hardware acceleration.*

## Usage

To run the full end-to-end pipeline, use the `Run_PipeLine.py` script. You must provide videos from your dual camera setup and a MATLAB `.mat` calibration file.

```bash
python Run_PipeLine.py --vid1 path/to/cam1.mp4 \
                       --vid2 path/to/cam2.mp4 \
                       --calib path/to/stereoParams.mat \
                       --model yolov8n.pt \
                       --output ./results \
                       --correct_refraction \
                       --d_air 0.0 \
                       --d_glass 5.0
```

### Arguments

| Argument | Description | Default |
| --- | --- | --- |
| `--vid1` | **(Required)** Path to the Left Camera Video. | |
| `--vid2` | **(Required)** Path to the Right Camera Video. | |
| `--calib` | **(Required)** Path to the Stereo Parameters `.mat` file. | |
| `--model` | Path to the YOLO weights to use for detection. | `yolov8n.pt` |
| `--output` | Root directory to store all outputs. | `./results` |
| `--correct_refraction` | Enable refraction correction for 3D coordinate generation. Use this if your system was calibrated in air but records in water. | `False` |
| `--d_air` | Distance from camera origin to the glass flat port (mm). Only used if `--correct_refraction` is set. | `0.0` |
| `--d_glass` | Thickness of the glass flat port (mm). Only used if `--correct_refraction` is set. | `0.0` |

## Outputs Structure

Running the pipeline will automatically generate the following sub-directories inside your chosen `--output` folder:

- `/mots/`: Contains intermediate and final coordinate mappings (`cam1_raw.csv`, `cam2_mapped.csv`, `3d_trajectory.csv`).
- `/videos/`: Contains the final rendered `.mp4` video files with IDs and bounding boxes overlaid on the footage.
