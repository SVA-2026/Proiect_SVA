# Multi-View 3D Object Reconstruction

The application reconstructs a sparse 3D model of an object from multiple calibrated images using computer vision techniques. It detects the same feature points across images, reconstructs their 3D positions and visualizes the final point cloud.

## Features

- Uses segmentation masks to limit feature detection to the object of interest
- Detects feature points using **SIFT**
- Matches features between images with **FLANN**
- Removes incorrect matches using **Lowe Ratio Test** and **RANSAC**
- Reconstructs 3D points using **DLT triangulation**
- Improves the reconstruction with **Bundle Adjustment**
- Visualizes the reconstructed point cloud and camera positions

## Technologies

- Python
- OpenCV
- NumPy
- SciPy
- Matplotlib

## How to Run

Run the complete reconstruction:

```bash
python run_full_dataset.py
```

To test the pipeline on a pair of images:

```bash
python test__.py
```


## Pipeline

1. Load the images, segmentation masks and camera parameters.
2. Detect SIFT keypoints and descriptors.
3. Match features using FLANN.
4. Filter incorrect matches with Lowe Ratio Test and RANSAC.
5. Reconstruct the 3D points using triangulation.
6. Refine the reconstruction with Bundle Adjustment.
7. Visualize the reconstructed point cloud.


## Results
The application reconstructs a sparse 3D model of the object from multiple calibrated images. It also projects the reconstructed points back onto the original images to verify the reconstruction quality.

<p align="center">
  <img width="566" height="533" alt="image" src="https://github.com/user-attachments/assets/3816fc23-12ea-4ecb-9309-49f686295da1" />

<img width="497" height="427" alt="image" src="https://github.com/user-attachments/assets/803309ad-0080-4394-bfe5-a7df7776c39f" />
</p>

The reconstruction generates:

- a colored 3D point cloud;
- estimated camera positions;
- feature correspondences between images;
- reprojection statistics for evaluating the reconstruction quality. 



