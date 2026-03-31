import cv2
import numpy as np


class GeometryEstimator:

    @staticmethod
    def get_camera_matrix(image_shape):
        h, w = image_shape[:2]
        f = max(w, h) * 1.2
        K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], dtype=np.float64)
        return K

    @staticmethod
    def estimate_fundamental_matrix(kp1, kp2, matches):
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)
        return F, mask, pts1, pts2

    @staticmethod
    def filter_matches_ransac(kp1, kp2, matches, threshold=3.0):
        if len(matches) < 8:
            return [], None, None

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        F, mask = cv2.findFundamentalMat(
            pts1, pts2,
            cv2.FM_RANSAC,
            ransacReprojThreshold=threshold,
            confidence=0.999
        )

        if F is None or mask is None:
            return [], None, None

        mask_flat    = mask.ravel().astype(bool)
        good_matches = [m for m, ok in zip(matches, mask_flat) if ok]
        pts1_inliers = pts1[mask_flat]
        pts2_inliers = pts2[mask_flat]

        return good_matches, pts1_inliers, pts2_inliers

    @staticmethod
    def triangulate_with_known_poses(pts1, pts2, cam1, cam2):
        P1 = cam1['K'] @ np.hstack([cam1['R'], cam1['t']])
        P2 = cam2['K'] @ np.hstack([cam2['R'], cam2['t']])

        pts4d = cv2.triangulatePoints(
            P1, P2,
            pts1.T.astype(np.float64),
            pts2.T.astype(np.float64)
        )

        #omogenice -> euclidiene
        w_coord = pts4d[3]
        #eliminam punctele cu w aprox 0
        valid_w = np.abs(w_coord) > 1e-6
        pts3d   = np.zeros((pts1.shape[0], 3))
        pts3d[valid_w] = (pts4d[:3, valid_w] / w_coord[valid_w]).T

        #cheirality: punctul trebuie sa fie in fata ambelor camere
        def in_front(cam, pts):
            pts_cam = (cam['R'] @ pts.T + cam['t']).T
            return pts_cam[:, 2] > 0.01  # minim 1cm in fata

        valid = valid_w & in_front(cam1, pts3d) & in_front(cam2, pts3d)
        return pts3d[valid], valid

    @staticmethod
    def reprojection_error(pts3d, pts2d, cam):
        if len(pts3d) == 0:
            return 0.0, 0.0, np.array([])

        P = cam['K'] @ np.hstack([cam['R'], cam['t']])
        pts_h  = np.hstack([pts3d, np.ones((len(pts3d), 1))])
        proj   = (P @ pts_h.T).T
        proj2d = proj[:, :2] / proj[:, 2:3]
        errors = np.linalg.norm(proj2d - pts2d, axis=1)
        return errors.mean(), errors.std(), errors