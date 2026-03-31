import cv2
import os


class ImageLoader:
    @staticmethod
    def load_from_directory(directory_path, extension=".jpg"):
        images = []

        filenames = sorted(os.listdir(directory_path))

        for filename in filenames:
            path = os.path.join(directory_path, filename)
            img = cv2.imread(path)

            if img is not None:
                images.append(img)

        return images