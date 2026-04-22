"""
main_v2.py - Pipeline extins cu:
  1. Triangulare MULTI-VIEW prin tracks + DLT (SVD)  ->  puncte care apar in >=3 vederi
  2. Point cloud colorat cu RGB real din imagini
  3. Analiza calitativa (histograme erori + statistici per pereche)

Varianta originala (main.py) a fost pastrata pentru backup.
"""

import cv2
import numpy as np
import os
from src.loaders.image_loader import ImageLoader
from src.loaders.camera_loader import CameraLoader
from src.features.sift_detector_v2 import SIFTDetectorV2
from src.features.feature_matcher import FlannMatcher
from src.geometry.multiview_triangulator import MultiViewTriangulator
from src.geometry.bundle_adjuster import BundleAdjuster
from src.visualization.plotter import Plotter
from src.selection.image_selector import ImageSelector
from src.analysis.reconstruction_analyzer import ReconstructionAnalyzer

IMAGE_DIR   = "data/input/images"
MASK_DIR    = "data/input/masks"
CAMERA_JSON = "data/input/cameras.json"
N_VIEWS     = 20

MIN_TRACK_LENGTH   = 2
MAX_REPROJ_ERROR   = 20.0
RANSAC_THRESHOLD   = 4.0
LOWE_RATIO         = 0.85

SIFT_CONTRAST_THR  = 0.02
SIFT_EDGE_THR      = 15

USE_BUNDLE_ADJUSTMENT   = True
BA_MAX_ITERATIONS       = 50
POST_BA_REPROJ_FILTER   = 3.0   #dupa BA putem fi mult mai stricti

os.makedirs("data/output", exist_ok=True)

#incarcare camere
camera_data = CameraLoader.load(CAMERA_JSON)
print(f"Camere incarcate: {len(camera_data)}")
print(f"Focal length: {camera_data[0]['f']:.1f} px")

#incarcare imagini + masti
loader = ImageLoader()
all_images = loader.load_from_directory(IMAGE_DIR, extension=".jpg")
if not all_images:
    all_images = loader.load_from_directory(IMAGE_DIR, extension=".png")
print(f"Total imagini: {len(all_images)}")


def load_and_fix_mask(raw_mask):
    gray_m = cv2.cvtColor(raw_mask, cv2.COLOR_BGR2GRAY) \
             if len(raw_mask.shape) == 3 else raw_mask
    max_val = gray_m.max()
    if max_val == 0:
        return gray_m
    if max_val <= 1:
        gray_m = (gray_m * 255).astype(np.uint8)
    else:
        _, gray_m = cv2.threshold(gray_m, max_val // 2, 255, cv2.THRESH_BINARY)
    filled = cv2.morphologyEx(
        gray_m, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    )
    filled = cv2.morphologyEx(
        filled, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    return filled


all_masks = [None] * len(all_images)
if os.path.exists(MASK_DIR):
    raw_masks = loader.load_from_directory(MASK_DIR, extension=".png")
    if len(raw_masks) == len(all_images):
        all_masks = [load_and_fix_mask(m) for m in raw_masks]
        print(f"Masti corectate: {len(all_masks)}")

#detectie features
detector = SIFTDetectorV2(
    contrast_threshold=SIFT_CONTRAST_THR,
    edge_threshold=SIFT_EDGE_THR,
)
all_keypoints   = []
all_descriptors = []
for i, (img, mask) in enumerate(zip(all_images, all_masks)):
    kp, desc = detector.detect_and_compute(img, mask)
    all_keypoints.append(kp)
    all_descriptors.append(desc)
    print(f"Imagine {i:02d}: {len(kp)} keypoints")

#selectie folosim toate imaginile
USE_ALL_IMAGES = True   #True = toate imaginile; False = selectie greedy cu ImageSelector

if USE_ALL_IMAGES:
    selected_indices = list(range(len(all_images)))
    print(f"\nFolosim toate cele {len(selected_indices)} imagini din dataset")
else:
    selector = ImageSelector(n_select=N_VIEWS, min_matches=20, min_features=150)
    selected_indices = selector.select(all_images, all_keypoints, all_descriptors)
    print(f"\nImagini selectate: {selected_indices}")

images      = [all_images[i]      for i in selected_indices]
keypoints   = [all_keypoints[i]   for i in selected_indices]
descriptors = [all_descriptors[i] for i in selected_indices]
cameras_raw = [camera_data[i]     for i in selected_indices]
cameras     = CameraLoader.normalize_to_first_camera(cameras_raw)

#triangularizare multi-view
print("\n" + "=" * 60)
print("PIPELINE MULTI-VIEW (tracks + DLT)")
print("=" * 60)

matcher = FlannMatcher(ratio=LOWE_RATIO)
triangulator = MultiViewTriangulator(
    min_track_length=MIN_TRACK_LENGTH,
    max_reproj_error=MAX_REPROJ_ERROR,
    ransac_threshold=RANSAC_THRESHOLD,
)

#construim tracks peste toate vederile
print("\nConstruire tracks (union-find peste match-uri RANSAC)")
tracks, pair_match_stats = triangulator.build_tracks(
    keypoints, descriptors, matcher
)
print(f"   Tracks construite (>= {MIN_TRACK_LENGTH} vederi): {len(tracks)}")

#triangularizare DLT per track + filtrare consistenta
print(f"\nTriangulare DLT multi-view + filtru eroare max < {MAX_REPROJ_ERROR}px")
points_3d, track_info = triangulator.triangulate_tracks(
    tracks, keypoints, cameras
)
print(f"Puncte 3D supravietuitoare dupa filtru: {len(points_3d)}")

#bundle adjustment
ba_stats = None
if USE_BUNDLE_ADJUSTMENT and len(points_3d) > 10:
    ba = BundleAdjuster(max_iterations=BA_MAX_ITERATIONS, verbose=True,
                        refine_cameras=True)
    cameras, points_3d, track_info, ba_stats = ba.run(cameras, points_3d, track_info)

    #dupa BA aplicam un filtru mai strict pe erorile noi
    mean_errs_new = np.array([t['mean_err'] for t in track_info])
    keep_ba = mean_errs_new < POST_BA_REPROJ_FILTER
    n_removed_ba = (~keep_ba).sum()
    points_3d  = points_3d[keep_ba]
    track_info = [t for t, k in zip(track_info, keep_ba) if k]
    print(f"\nDupa BA: filtru mean_err < {POST_BA_REPROJ_FILTER}px - "
          f"eliminate {n_removed_ba} puncte, ramase {len(points_3d)}")

#psot-filtrare IQR (outliere extreme)
if len(points_3d) > 10:
    keep_iqr = np.ones(len(points_3d), dtype=bool)
    for ax in range(3):
        q25, q75 = np.percentile(points_3d[:, ax], [25, 75])
        iqr = q75 - q25
        lo = q25 - 2.5 * iqr
        hi = q75 + 2.5 * iqr
        keep_iqr &= (points_3d[:, ax] >= lo) & (points_3d[:, ax] <= hi)
    n_removed = (~keep_iqr).sum()
    points_3d = points_3d[keep_iqr]
    track_info = [t for t, k in zip(track_info, keep_iqr) if k]
    print(f"\nFiltrare IQR pe coordonatele 3D: eliminate {n_removed} outlieri extremi")
    print(f"Puncte 3D finale: {len(points_3d)}")

#extragere culori RGB
print("\nExtragere culori RGB din imagini la pozitiile keypoints...")
colors_rgb = MultiViewTriangulator.extract_colors(track_info, images)

#bounding box - verificare scala
if len(points_3d) > 0:
    bb_min = points_3d.min(axis=0)
    bb_max = points_3d.max(axis=0)
    bb_size = bb_max - bb_min
    print(f"\nBounding box obiect reconstruit:")
    print(f"  X: [{bb_min[0]:8.1f}, {bb_max[0]:8.1f}]  dim={bb_size[0]:8.1f}")
    print(f"  Y: [{bb_min[1]:8.1f}, {bb_max[1]:8.1f}]  dim={bb_size[1]:8.1f}")
    print(f"  Z: [{bb_min[2]:8.1f}, {bb_max[2]:8.1f}]  dim={bb_size[2]:8.1f}")

#analiza calitativa
ReconstructionAnalyzer.print_global_summary(track_info, points_3d)
ReconstructionAnalyzer.print_pair_statistics(pair_match_stats, track_info)

if len(points_3d) > 0:
    ReconstructionAnalyzer.plot_error_histogram(
        track_info,
        save_path="data/output/reproj_hist.png"
    )

#vizualizare
if len(points_3d) > 0:
    camera_poses = [(c['R'], c['t']) for c in cameras]

    #point cloud colorat cu RGB real
    Plotter.plot_colored_point_cloud(
        points_3d, colors_rgb,
        title=f"Point Cloud Multi-View (RGB real) - {len(points_3d)} puncte"
    )

    #puncte cu lungime track > 3 (observate in multe vederi = cele mai fiabile)
    lengths = np.array([t['track_length'] for t in track_info])
    mask_long = lengths >= 4
    if mask_long.sum() > 0:
        Plotter.plot_colored_point_cloud(
            points_3d[mask_long], colors_rgb[mask_long],
            title=f"Puncte din >=4 vederi - {mask_long.sum()} puncte (cele mai fiabile)"
        )

    Plotter.plot_camera_poses(
        camera_poses,
        title="Pozitii camere estimate din JSON"
    )

    #proiectie pe prima vedere pentru verificare vizuala
    view_idx = 0
    global_img_idx = selected_indices[view_idx]

    Plotter.overlay_points_on_image(
        all_images[global_img_idx],
        points_3d,
        cameras[view_idx],
        title=f"Toate punctele 3D ({len(points_3d)}) proiectate pe vederea {view_idx + 1} "
              f"(imaginea {global_img_idx:02d})"
    )

    mask_this_view = np.array([
        view_idx in t['views'] for t in track_info
    ])
    n_in_view = mask_this_view.sum()
    if n_in_view > 0:
        Plotter.overlay_points_on_image(
            all_images[global_img_idx],
            points_3d[mask_this_view],
            cameras[view_idx],
            title=f"Doar punctele triangulate cu vederea {view_idx + 1} "
                  f"({n_in_view} puncte, imaginea {global_img_idx:02d})"
        )
    print(f"\nDin {len(points_3d)} puncte 3D, {n_in_view} au fost observate si in vederea {view_idx + 1}")
else:
    print("Nu s-au reconstruit puncte 3D valide. Verifica pragurile.")