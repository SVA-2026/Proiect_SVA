import cv2
import os


class ImageLoader:
    @staticmethod
    def load_from_directory(directory_path, extension=".jpg", target_width=800):
        images = []
        # sortam fisierele pentru a pastra ordinea corecta
        filenames = sorted([f for f in os.listdir(directory_path) if f.lower().endswith(extension)])

        for filename in filenames:
            path = os.path.join(directory_path, filename)
            img = cv2.imread(path)

            if img is not None:
                h, w = img.shape[:2]
                aspect_ratio = h / w
                target_height = int(target_width * aspect_ratio)
                img_resized = cv2.resize(img, (target_width, target_height))

                images.append(img_resized)
                print(f"S-a reusit dimensionarea imaginilor")

        return images