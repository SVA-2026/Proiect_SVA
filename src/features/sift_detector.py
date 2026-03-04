import cv2
import numpy as np

from src.interfaces.feature_interfaces import IFeatureDetector, IFeatureMatcher

class SIFTDetector(IFeatureDetector):
    def __init__(self):
        self.detector = cv2.SIFT_create()

    def detect_and_compute(self, image):
        #convertim in grayscale pt SIFT
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.detector.detectAndCompute(gray, None)




