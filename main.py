import cv2
import numpy as np
from src.loaders.image_loader import ImageLoader
from src.features.sift_detector import SIFTDetector
from src.features.feature_matcher import FlannMatcher
from src.geometry.estimators import GeometryEstimator
from src.visualization.plotter import Plotter


def main():
    loader = ImageLoader()
    images = loader.load_from_directory("data/input/")

    if len(images) < 2:
        print("Nu s-au gasit poze in folderul data/input/")
        return

    img1, img2 = images[0], images[1]

    #facem detectia SIFT
    detector = SIFTDetector()
    kp1, des1 = detector.detect_and_compute(img1)
    kp2, des2 = detector.detect_and_compute(img2)

    # 3.lowe s ratio set
    matcher = FlannMatcher(ratio=0.75)
    matches = matcher.match(des1, des2)

    # RANSAC ( pentru a elimina liniile ce sunt complet pe langa majoritatea )
    F, mask, pts1, pts2 = GeometryEstimator.estimate_fundamental_matrix(kp1, kp2, matches)

    #luam doar cele acceptate de RANSAC
    pts1_inliers = pts1[mask.ravel() == 1]
    pts2_inliers = pts2[mask.ravel() == 1]
    inlier_matches = [matches[i] for i in range(len(matches)) if mask[i]]

    print(f"S-au gasit {len(inlier_matches)} puncte comune intre aceste imagini")

    Plotter.draw_matches(img1, kp1, img2, kp2, inlier_matches, "Asociere imagini")

    ########
    lines1 = cv2.computeCorrespondEpilines(pts2_inliers.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)

    #desenam liniile epipolare
    Plotter.draw_epipolar_lines(img1, img2, lines1, pts1_inliers, pts2_inliers)

if __name__ == "__main__":
    main()