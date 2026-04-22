import cv2
from src.interfaces.feature_interfaces import IFeatureDetector


class SIFTDetectorV2(IFeatureDetector):
    """
    Detector SIFT parametrizabil pentru a controla densitatea keypoints.
    Versiune imbunatatita fata de SIFTDetector original - permite control fin
    asupra pragurilor de contrast si muchii pentru a extrage mai multe features.

    Parametri importanti:
      contrast_threshold: prag mai mic = MAI MULTE keypoints in zone intunecate/slabe
                         (default OpenCV = 0.04; 0.01-0.02 = mult mai agresiv)
      edge_threshold   : prag mai mare = mai putine keypoints pe muchii (default = 10)
      n_features       : limita de keypoints per imagine (0 = fara limita)
      sigma            : nivel de smoothing initial (default = 1.6)
    """
    def __init__(self,
                 contrast_threshold=0.02,   #mai mic decat default pt. mai multe features
                 edge_threshold=15,         #ceva mai permisiv la muchii
                 n_features=0,              #fara limita
                 sigma=1.6):
        self.detector = cv2.SIFT_create(
            nfeatures=n_features,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
            sigma=sigma,
        )

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
