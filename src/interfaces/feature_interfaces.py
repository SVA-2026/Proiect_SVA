from abc import ABC, abstractmethod
import numpy as np

class IFeatureDetector(ABC):
    @abstractmethod
    def detect_and_compute(self, image: np.ndarray):
        #returneaza keypoints, descriptors
        pass

class IFeatureMatcher(ABC):
    @abstractmethod
    def match(self, desc1: np.ndarray, desc2: np.ndarray):
        #returneaza o lista de potriviri filtrate
        pass