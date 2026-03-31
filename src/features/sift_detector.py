import cv2
from src.interfaces.feature_interfaces import IFeatureDetector


class SIFTDetector(IFeatureDetector):
    def __init__(self):
        self.detector = cv2.SIFT_create()

    def detect_and_compute(self, image, mask=None):
        #convertim in grayscale pt SIFT
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #daca avem masca, innegrim complet fundalul imaginii
        if mask is not None:
            #verificam ca masca are strict aceeasi dimensiune cu imaginea
            mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]))
            #tot ce e in afara mastii devine complet negru
            gray = cv2.bitwise_and(gray, gray, mask=mask)

        return self.detector.detectAndCompute(gray, None)