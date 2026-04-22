import cv2
import numpy as np
import os
from src.loaders.image_loader import ImageLoader
from src.loaders.camera_loader import CameraLoader
from src.features.sift_detector import SIFTDetector
from src.features.feature_matcher import FlannMatcher
from src.geometry.estimators import GeometryEstimator
from src.visualization.plotter import Plotter
from src.selection.image_selector import ImageSelector

IMAGE_DIR   = "data/input/images"
MASK_DIR    = "data/input/masks"
CAMERA_JSON = "data/input/cameras.json"
N_VIEWS     = 5
os.makedirs("data/output", exist_ok=True)


#incarcare date camera
camera_data = CameraLoader.load(CAMERA_JSON)

#verificare R, det sa fie 1
for i in range(3):
    det = np.linalg.det(camera_data[i]['R'])
    print(f"Camera {i:02d}: det(R) = {det:.6f}  "
          f"{'OK' if abs(det - 1.0) < 1e-4 else 'GRESIT'}")

#proiectare origine (0,0,0) in fiecare camera
for i in range(len(camera_data)):
    P  = CameraLoader.get_projection_matrix(camera_data[i]) #calcularea matricea de proiectie
    pt = P @ np.array([0, 0, 0, 1.0]) #trecerea din 2d in 3d
    px = pt[0] / pt[2]
    py = pt[1] / pt[2]
    h, w = camera_data[i]['image_size']
    inside = 'in imagine' if 0 < px < w and 0 < py < h else 'a iesit din imagine'
    print(f"Camera {i:02d}: pixel=({px:.0f}, {py:.0f})  "
          f"imagine=({w}x{h})  {inside}")

print(f"\nCamere incarcate: {len(camera_data)}")
print(f"Focal length: {camera_data[0]['f']:.1f} px")
print(f"cx={camera_data[0]['cx']:.1f}, cy={camera_data[0]['cy']:.1f}")
print(f"K:\n{camera_data[0]['K'].astype(int)}")

#incarcare imagini
loader = ImageLoader()
all_images = loader.load_from_directory(IMAGE_DIR, extension=".jpg")
if not all_images:
    all_images = loader.load_from_directory(IMAGE_DIR, extension=".png")
print(f"Total imagini: {len(all_images)}")

#incarcare si corectie masti
def load_and_fix_mask(raw_mask):
    gray_m  = cv2.cvtColor(raw_mask, cv2.COLOR_BGR2GRAY) \
              if len(raw_mask.shape) == 3 else raw_mask
    max_val = gray_m.max()
    if max_val == 0:
        return gray_m
    if max_val <= 1:
        gray_m = (gray_m * 255).astype(np.uint8)
    else:
        _, gray_m = cv2.threshold(gray_m, max_val // 2, 255, cv2.THRESH_BINARY)

    #estompare gauri mici si eliminare zgomot
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

#decatie features
detector        = SIFTDetector()
all_keypoints   = []
all_descriptors = []

for i, (img, mask) in enumerate(zip(all_images, all_masks)):
    kp, desc = detector.detect_and_compute(img, mask)
    all_keypoints.append(kp)
    all_descriptors.append(desc)
    print(f"Imagine {i:02d}: {len(kp)} keypoints")

#selectie automata
selector = ImageSelector(n_select=N_VIEWS, min_matches=30, min_features=200)
selected_indices = selector.select(all_images, all_keypoints, all_descriptors)

images      = [all_images[i]      for i in selected_indices]
keypoints   = [all_keypoints[i]   for i in selected_indices]
descriptors = [all_descriptors[i] for i in selected_indices]
cameras_raw = [camera_data[i]     for i in selected_indices]

# normalizare - toate pose-urile relativ la prima camera
cameras = CameraLoader.normalize_to_first_camera(cameras_raw)

print(f"Imagini selectate: {selected_indices}")
print("\nCentru camera dupa normalizare:")
for i, (idx, cam) in enumerate(zip(selected_indices, cameras)):
    C = CameraLoader.get_camera_center(cam)
    print(f"Camera {i} imginea {idx:02d}: C=[{C[0]:.1f}, {C[1]:.1f}, {C[2]:.1f}]")

#reconstructie 3D
matcher       = FlannMatcher(ratio=0.75)
all_points_3d = []
reproj_errors = []

for i in range(len(images)):
    for j in range(i + 1, len(images)):
        #matches brute cu FLANN
        raw_matches = matcher.match(descriptors[i], descriptors[j])
        if len(raw_matches) < 15:
            print(f"[{i} cu {j}] prea puține matches ({len(raw_matches)})")
            continue

        #filtrare RANSAC cu matricea fundamentala
        good_matches, pts1, pts2 = GeometryEstimator.filter_matches_ransac(
            keypoints[i], keypoints[j], raw_matches, threshold=2.0
        )

        if len(good_matches) < 10:
            continue

        # triangulare cu K,R,t
        pts3d, valid_mask = GeometryEstimator.triangulate_with_known_poses(
            pts1, pts2, cameras[i], cameras[j]
        )

        print(f"{len(pts3d)} puncte 3D valide")

        if len(pts3d) < 5:
            continue

        #eroare de reproiectie
        pts1_v = pts1[valid_mask]
        pts2_v = pts2[valid_mask]

        err1_mean, err1_std, errs1 = GeometryEstimator.reprojection_error(
            pts3d, pts1_v, cameras[i]
        )
        err2_mean, err2_std, errs2 = GeometryEstimator.reprojection_error(
            pts3d, pts2_v, cameras[j]
        )
        mean_err = (err1_mean + err2_mean) / 2
        reproj_errors.append(mean_err)

        #filtrare pe eroarea de reproiecție per punct
        per_point_err = (errs1 + errs2) / 2
        keep_reproj = per_point_err < 5.0
        pts3d = pts3d[keep_reproj]
        print(f"{len(pts3d)} puncte dupa filtrare reproj < 5px")

        #filtrare pe Z - distanta fata de camera
        if len(pts3d) > 10:
            #filtrare cu IQR
            for ax in range(3):
                q25, q75 = np.percentile(pts3d[:, ax], [25, 75])
                iqr = q75 - q25
                lo = q25 - 2.5 * iqr
                hi = q75 + 2.5 * iqr
                pts3d = pts3d[(pts3d[:, ax] >= lo) & (pts3d[:, ax] <= hi)]

        if len(pts3d) > 0:
            all_points_3d.append(pts3d)

#analiza calitate reconstructie
total_pts = sum(len(p) for p in all_points_3d)
print(f"Total puncte 3D reconstruite : {total_pts}")
print(f"Perechi cu reconstructie valida: {len(all_points_3d)}")

if reproj_errors:
    mean_err = np.mean(reproj_errors)
    print(f"Eroare reproiectie medie: {mean_err:.2f} px")
    if mean_err < 2:
        print("Excelent (< 2px)")
    elif mean_err < 5:
        print("Bun (2-5px)")
    elif mean_err < 20:
        print("Acceptabil (5-20px)")
    else:
        print("Probleme — pose-urile sau matches-urile au erori mari")

if all_points_3d:
    all_pts  = np.vstack(all_points_3d)
    bb_min   = all_pts.min(axis=0)
    bb_max   = all_pts.max(axis=0)
    bb_size  = bb_max - bb_min
    print(f"\nBounding box obiect reconstruit:")
    print(f"  X: [{bb_min[0]:.1f}, {bb_max[0]:.1f}]  dim={bb_size[0]:.1f} mm")
    print(f"  Y: [{bb_min[1]:.1f}, {bb_max[1]:.1f}]  dim={bb_size[1]:.1f} mm")
    print(f"  Z: [{bb_min[2]:.1f}, {bb_max[2]:.1f}]  dim={bb_size[2]:.1f} mm")
    print(f"\nSurse de eroare comune:")
    print(f"- Eroare quaternion/conventie rotatie")
    print(f"- Distorsiune lentilt necalibratt (dataset fara k1,k2)")
    print(f"- Matches false pozitive care trec de RANSAC")
    print(f"- Baseline mic intre camere adiacente")

#vizualizare
camera_poses = [(c['R'], c['t']) for c in cameras]

if all_points_3d:
    Plotter.plot_point_cloud(
        all_points_3d,
        title="Sparse 3D Point Cloud - pose reale din JSON"
    )
    Plotter.plot_camera_poses(
        camera_poses,
        title="Pozitii camere estimate din JSON"
    )
    view_idx = 0

    #obtinere date originale
    global_img_idx = selected_indices[view_idx]
    img_to_plot = all_images[global_img_idx]
    cam_data_to_plot = cameras[view_idx]

    #reunim punctele intr-un singur array de tip (N, 3) pentru functie
    all_pts = np.vstack(all_points_3d)

    #rulam vizualizarea
    Plotter.overlay_points_on_image(
        img_to_plot,
        all_pts,
        cam_data_to_plot,
        title=f"Norul de puncte proiectat pe vederea {view_idx + 1} (imaginea {global_img_idx:02d})"
    )
else:
    print("Nu s-au reconstruit puncte 3D valide")

