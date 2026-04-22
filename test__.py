import cv2
import numpy as np
import json
from src.loaders.image_loader import ImageLoader
from src.loaders.camera_loader import CameraLoader
from src.features.sift_detector import SIFTDetector
from src.features.feature_matcher import FlannMatcher

#incarca doar doua imgagini, 05 si 08, perechea cu 862 matches
with open("data/input/cameras.json") as f:
    data = json.load(f)

loader   = ImageLoader()
all_imgs = loader.load_from_directory("data/input/images", extension=".jpg")

img1 = all_imgs[5]
img2 = all_imgs[8]
cam1 = CameraLoader.load("data/input/cameras.json")[5]
cam2 = CameraLoader.load("data/input/cameras.json")[8]

print(f"K:\n{cam1['K'].astype(int)}")
print(f"\nCam5 centru: {(-cam1['R'].T @ cam1['t']).ravel().astype(int)}")
print(f"Cam8 centru: {(-cam2['R'].T @ cam2['t']).ravel().astype(int)}")
print(f"Baseline: {np.linalg.norm((-cam1['R'].T@cam1['t']) - (-cam2['R'].T@cam2['t'])):.1f}mm")

#SIFT fara masca pt test
detector = SIFTDetector()
kp1, desc1 = detector.detect_and_compute(img1, None)
kp2, desc2 = detector.detect_and_compute(img2, None)
print(f"\nKeypoints: {len(kp1)} / {len(kp2)}")

# Matches
matcher = FlannMatcher(ratio=0.75)
matches = matcher.match(desc1, desc2)
print(f"Matches dupa Lowe: {len(matches)}")

#RANSAC cu F
pts1 = np.float64([kp1[m.queryIdx].pt for m in matches])
pts2 = np.float64([kp2[m.trainIdx].pt for m in matches])
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.999)
inliers = mask.ravel().astype(bool)
pts1_in = pts1[inliers]
pts2_in = pts2[inliers]
print(f"Inlieri RANSAC (F): {inliers.sum()}")

P1 = cam1['K'] @ np.hstack([cam1['R'], cam1['t']])
P2 = cam2['K'] @ np.hstack([cam2['R'], cam2['t']])

pts4d = cv2.triangulatePoints(P1, P2, pts1_in.T, pts2_in.T)
pts3d = (pts4d[:3] / pts4d[3]).T

#filtrare cheirality
z1 = (cam1['R'] @ pts3d.T + cam1['t']).T[:, 2]
z2 = (cam2['R'] @ pts3d.T + cam2['t']).T[:, 2]
valid = (z1 > 0) & (z2 > 0)
pts3d = pts3d[valid]
print(f"Puncte 3D valide: {len(pts3d)}")

def reproj_err(pts3d, pts2d, P):
    pts_h  = np.hstack([pts3d, np.ones((len(pts3d),1))])
    proj   = (P @ pts_h.T).T
    proj2d = proj[:,:2] / proj[:,2:3]
    return np.linalg.norm(proj2d - pts2d, axis=1)

errs1 = reproj_err(pts3d, pts1_in[valid], P1)
errs2 = reproj_err(pts3d, pts2_in[valid], P2)

print(f"\nEroare reproiectie:")
print(f"camera 5: mean={errs1.mean():.2f}px  median={np.median(errs1):.2f}px  max={errs1.max():.1f}px")
print(f"camera 8: mean={errs2.mean():.2f}px  median={np.median(errs2):.2f}px  max={errs2.max():.1f}px")

#distributia erorilor
for thresh in [1, 2, 5, 10, 20, 50]:
    pct1 = (errs1 < thresh).mean() * 100
    pct2 = (errs2 < thresh).mean() * 100
    print(f"{thresh:3d}px: cam5={pct1:.0f}%  cam8={pct2:.0f}%")

print(f"\nPuncte 3D statistici:")
print(f"X: [{pts3d[:,0].min():.1f}, {pts3d[:,0].max():.1f}]")
print(f"Y: [{pts3d[:,1].min():.1f}, {pts3d[:,1].max():.1f}]")
print(f"Z: [{pts3d[:,2].min():.1f}, {pts3d[:,2].max():.1f}]")
print(f"Dist de la origine: mean={np.linalg.norm(pts3d, axis=1).mean():.1f}  "
      f"median={np.median(np.linalg.norm(pts3d, axis=1)):.1f}")

#vizualizare matches pe imagini
def show_small(name, img, scale=0.2):
    h, w = img.shape[:2]
    cv2.imshow(name, cv2.resize(img, (int(w*scale), int(h*scale))))

img_matches = cv2.drawMatches(
    img1, kp1, img2, kp2,
    [m for m, ok in zip(matches, inliers) if ok][:100],
    None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
show_small("Inlieri RANSAC (primele 100)", img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()