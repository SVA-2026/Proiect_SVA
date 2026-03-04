import cv2
from src.interfaces.feature_interfaces import IFeatureMatcher

class FlannMatcher(IFeatureMatcher):
    def __init__(self, ratio=0.75):
        self.ratio = ratio
        #parametri pt alg FLANN specifici pt SIFT
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def match(self, desc1, desc2):
        if desc1 is None or desc2 is None:
            return []

        #k=2 pt aplicare loew s ratio test
        raw_matches = self.matcher.knnMatch(desc1, desc2, k=2)

        #aplicam lowe s ratio test
        #pastram doar potrvirile unde cea mai buna distanta e semnificativ mai mica decat cea de a doua
        good_matches = []
        for m, n in raw_matches:
            if m.distance < self.ratio * n.distance:
                good_matches.append(m)

        return good_matches


