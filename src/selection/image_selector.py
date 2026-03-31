import numpy as np
import cv2
from itertools import combinations

#selecteaza automat cele mai bune N imagini dintr-un set mare, maximizand coverage-ul unghiular si mentinand overlap suficient
class ImageSelector:

    def __init__(self, n_select=5, min_matches=30, min_features=200):
        self.n_select = n_select
        self.min_matches = min_matches   # overlap minim intre 2 imagini alese
        self.min_features = min_features  # imagini cu prea putine features sunt excluse

    #returneaza indicii celor mai bune n_select imagini.
    def select(self, images, keypoints_list, descriptors_list, masks=None):

        n = len(images)
        print(f"\nSelector - Total imagini: {n}")

        #eliminam imaginile cu prea putine features
        valid_indices = []
        for i, kp in enumerate(keypoints_list):
            if kp is not None and len(kp) >= self.min_features:
                valid_indices.append(i)
            else:
                count = len(kp) if kp is not None else 0

        print(f"Imagini valide dupa filtrare: {len(valid_indices)}")

        if len(valid_indices) <= self.n_select:
            print("Suficient de putine imagini, returnam toate cele valide")
            return valid_indices[:self.n_select]

       #construim matricea de overlap
        overlap = self._build_overlap_matrix(
            valid_indices, descriptors_list
        )

        #selectie greedy bazata pe coverage
        selected = self._greedy_select(valid_indices, overlap, keypoints_list)

        print(f"\nImagini selectate: {selected}")
        for i, idx in enumerate(selected):
            print(f"  View {i+1}: imaginea {idx:02d} "
                  f"({len(keypoints_list[idx])} features)")
        return selected


    def _build_overlap_matrix(self, valid_indices, descriptors_list):
        #calculeaz numarul de matches bune intre fiecare pereche de imagini valide
        #returneaza o matrice simetrica n_valid x n_valid
        n = len(valid_indices)
        overlap = np.zeros((n, n), dtype=np.int32)

        #matcher FLANN rapid
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=30)  # mai putin checks = mai rapid
        matcher = cv2.FlannBasedMatcher(index_params, search_params)

        total_pairs = n * (n - 1) // 2
        done = 0

        for i, j in combinations(range(n), 2):
            idx_i = valid_indices[i]
            idx_j = valid_indices[j]

            desc_i = descriptors_list[idx_i]
            desc_j = descriptors_list[idx_j]

            if desc_i is None or desc_j is None:
                continue

            try:
                raw = matcher.knnMatch(desc_i, desc_j, k=2)
                good = [m for m, n_m in raw if m.distance < 0.8 * n_m.distance]
                overlap[i, j] = len(good)
                overlap[j, i] = len(good)
            except Exception:
                overlap[i, j] = 0
                overlap[j, i] = 0

            done += 1
            if done % 20 == 0:
                print(f"Overlap: {done}/{total_pairs} perechi procesate...")

        return overlap

    def _greedy_select(self, valid_indices, overlap, keypoints_list):
        #start cu imaginea cu cele mai multe features
        #la fiecare pas adaugam imaginea care: are overlap suficient cu cel putin una dintre cele deja alese si maximizeaza 'new coverage' = features care nu se suprapun mult cu setul curent

        n = len(valid_indices)
        feature_counts = np.array([
            len(keypoints_list[valid_indices[i]]) for i in range(n)
        ])

        selected_local = []  #indici in matricea de overlap

        start = int(np.argmax(feature_counts))
        selected_local.append(start)

        while len(selected_local) < self.n_select and len(selected_local) < n:
            best_score = -1
            best_idx = -1

            for i in range(n):
                if i in selected_local:
                    continue

                max_overlap_with_selected = max(
                    overlap[i, j] for j in selected_local
                )

                if max_overlap_with_selected < self.min_matches:
                    continue  # prea puțin overlap, sare

                mean_overlap = np.mean([
                    overlap[i, j] for j in selected_local
                ])
                diversity_score = feature_counts[i] - 0.5 * mean_overlap

                if diversity_score > best_score:
                    best_score = diversity_score
                    best_idx = i

            if best_idx == -1:
                remaining = [i for i in range(n) if i not in selected_local]
                if remaining:
                    fallback = max(remaining, key=lambda i: feature_counts[i])
                    selected_local.append(fallback)
                else:
                    break
            else:
                selected_local.append(best_idx)

        #convertim inapoi la indicii originali
        return [valid_indices[i] for i in selected_local]