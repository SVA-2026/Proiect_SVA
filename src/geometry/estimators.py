#aici calculam matricea F (fundamentala) ce se ocupa de relatia dintre 2 vederi
import cv2
import numpy as np

class GeometryEstimator:
    @staticmethod
    def estimate_fundamental_matrix ( kp1, kp2, matches):
        #extragem coordonatele (x, y) din obieectivele KeyPoints
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        #RANSAC gaseste ccel mai bun model matematic
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
        return F, mask, pts1, pts2