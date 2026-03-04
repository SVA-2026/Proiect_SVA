import cv2
import numpy as np


class Plotter:
    @staticmethod
    def draw_matches(img1, kp1, img2, kp2, matches, window_name="Feature Matches"):
        #desenare linii intre 2 imagini cu potrivire
        img_matches = cv2.drawMatches(
            img1, kp1, img2, kp2, matches[:50], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.imshow(window_name, img_matches)
        cv2.waitKey(0)

    @staticmethod
    def draw_epipolar_lines(img1, img2, lines, pts1, pts2):
        #desenare linii epipolare
        r, c, _ = img1.shape
        img1_out = img1.copy()
        img2_out = img2.copy()

        for r, pt1, pt2 in zip(lines, pts1, pts2):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
            img1_out = cv2.line(img1_out, (x0, y0), (x1, y1), color, 1)
            img1_out = cv2.circle(img1_out, tuple(map(int, pt1)), 5, color, -1)
            img2_out = cv2.circle(img2_out, tuple(map(int, pt2)), 5, color, -1)

        cv2.imshow("Epipolar Lines", img1_out)
        cv2.waitKey(0)