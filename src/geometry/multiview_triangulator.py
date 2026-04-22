import numpy as np
import cv2


class MultiViewTriangulator:
    """
    Triangulare multi-view bazata pe tracks.
    Un 'track' = acelasi punct 3D observat in >= 3 vederi.
    Pentru fiecare track rezolvam DLT (Direct Linear Transform) prin SVD si
    filtram punctele inconsistente dupa eroarea de reproiectie in toate vederile.
    """

    def __init__(self, min_track_length=3, max_reproj_error=3.0, ransac_threshold=2.0):
        self.min_track_length = min_track_length
        self.max_reproj_error = max_reproj_error
        self.ransac_threshold = ransac_threshold

    #construire tracks
    def build_tracks(self, keypoints_list, descriptors_list, matcher):
        """
        Construieste tracks folosind union-find peste toate match-urile pair-wise.
        Returneaza:
          tracks = lista de dict-uri {view_idx: keypoint_idx}
        """
        n_views = len(keypoints_list)

        #fiecare (view_idx, kp_idx) este un nod; il mapam la un id global
        node_id = {}
        def get_id(v, k):
            key = (v, k)
            if key not in node_id:
                node_id[key] = len(node_id)
            return node_id[key]

        #Union-Find (DSU)
        parent = []

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        #pentru fiecare pereche de imagini: match + filtrare RANSAC + union
        pair_match_stats = {}
        total_pairs = n_views * (n_views - 1) // 2
        pair_idx = 0
        print(f"Procesare {total_pairs} perechi de imagini...")

        for i in range(n_views):
            for j in range(i + 1, n_views):
                pair_idx += 1
                if pair_idx % 20 == 0 or pair_idx == total_pairs:
                    print(f"Pereche {pair_idx}/{total_pairs} ({100*pair_idx/total_pairs:.0f}%)")

                desc_i = descriptors_list[i]
                desc_j = descriptors_list[j]
                if desc_i is None or desc_j is None:
                    continue

                raw = matcher.match(desc_i, desc_j)
                if len(raw) < 15:
                    pair_match_stats[(i, j)] = (len(raw), 0)
                    continue

                #filtrare RANSAC cu matricea fundamentala
                kp_i = keypoints_list[i]
                kp_j = keypoints_list[j]
                pts1 = np.float32([kp_i[m.queryIdx].pt for m in raw])
                pts2 = np.float32([kp_j[m.trainIdx].pt for m in raw])

                F, mask = cv2.findFundamentalMat(
                    pts1, pts2, cv2.FM_RANSAC,
                    self.ransac_threshold, 0.999
                )
                if F is None or mask is None:
                    pair_match_stats[(i, j)] = (len(raw), 0)
                    continue

                mask = mask.ravel().astype(bool)
                good = [m for m, ok in zip(raw, mask) if ok]
                pair_match_stats[(i, j)] = (len(raw), len(good))

                #pentru fiecare match bun, unim cei doi keypoints intr-un track
                for m in good:
                    a = get_id(i, m.queryIdx)
                    #asiguram capacitatea parent
                    while len(parent) < len(node_id):
                        parent.append(len(parent))
                    b = get_id(j, m.trainIdx)
                    while len(parent) < len(node_id):
                        parent.append(len(parent))
                    union(a, b)

        #asiguram toate nodurile in parent
        while len(parent) < len(node_id):
            parent.append(len(parent))

        #grupam nodurile pe componente conexe -> tracks
        tracks_dict = {}
        for (v, k), nid in node_id.items():
            root = find(nid)
            if root not in tracks_dict:
                tracks_dict[root] = {}
            #daca un track are deja un kp din aceasta vedere, pastram unul singur
            #(evitam conflicte: acelasi track nu poate avea 2 kp din aceeasi imagine)
            if v not in tracks_dict[root]:
                tracks_dict[root][v] = k

        #pastram doar tracks cu lungime >= min_track_length
        tracks = [
            t for t in tracks_dict.values()
            if len(t) >= self.min_track_length
        ]

        return tracks, pair_match_stats

    #DLT multi-view
    @staticmethod
    def triangulate_dlt(observations, cameras):
        """
        observations = lista de (view_idx, (x, y)) pixeli observati
        cameras      = lista tuturor camerelor (dict cu K, R, t)
        Rezolva AX = 0 prin SVD -> punctul 3D in coordonate omogene.
        """
        A = []
        for v_idx, (x, y) in observations:
            cam = cameras[v_idx]
            P = cam['K'] @ np.hstack([cam['R'], cam['t']])  # 3x4
            #doua ecuatii per observatie
            A.append(x * P[2] - P[0])
            A.append(y * P[2] - P[1])
        A = np.asarray(A)

        #SVD: solutia = ultimul vector singular
        _, _, Vt = np.linalg.svd(A)
        X_h = Vt[-1]
        if abs(X_h[3]) < 1e-9:
            return None  #punct la infinit
        X = X_h[:3] / X_h[3]
        return X

    @staticmethod
    def reprojection_errors_all_views(X, observations, cameras):
        """
        Calculeaza eroarea de reproiectie in fiecare vedere unde apare X.
        Returneaza vectorul de erori (pixeli) + flag daca e in fata camerelor.
        """
        errors = []
        all_in_front = True
        for v_idx, (x, y) in observations:
            cam = cameras[v_idx]
            X_cam = cam['R'] @ X.reshape(3, 1) + cam['t']  # 3x1
            if X_cam[2, 0] <= 0.01: #in spatele sau prea aproape de camera
                all_in_front = False
                errors.append(np.inf)
                continue
            x_proj = cam['K'] @ X_cam
            u = x_proj[0, 0] / x_proj[2, 0]
            v = x_proj[1, 0] / x_proj[2, 0]
            errors.append(np.hypot(u - x, v - y))
        return np.asarray(errors), all_in_front

    def triangulate_tracks(self, tracks, keypoints_list, cameras):
        """
        Pentru fiecare track:
          1. Colecteaza observatiile 2D
          2. DLT multi-view
          3. Verifica cheirality + eroarea de reproiectie in TOATE vederile
        Returneaza:
          points_3d   (N, 3)
          track_info  lista de dict-uri cu: views, mean_err, max_err, track_length
        """
        points_3d = []
        track_info = []

        all_mean_errs = []
        all_max_errs = []
        n_dlt_fail = 0
        n_cheirality_fail = 0

        for track in tracks:
            #colectare observatii (view_idx, (x,y))
            observations = []
            for v_idx, kp_idx in track.items():
                kp = keypoints_list[v_idx][kp_idx]
                observations.append((v_idx, kp.pt))

            #DLT
            X = self.triangulate_dlt(observations, cameras)
            if X is None:
                n_dlt_fail += 1
                continue

            #reproiectie + cheirality
            errs, in_front = self.reprojection_errors_all_views(X, observations, cameras)
            if not in_front:
                n_cheirality_fail += 1
                continue

            mean_err = float(errs.mean())
            max_err  = float(errs.max())

            all_mean_errs.append(mean_err)
            all_max_errs.append(max_err)

            #filtru consistenta: punctul trebuie sa aiba eroare medie sub prag
            #(mean este mai tolerant decat max - o singura vedere proasta nu omoara punctul)
            if mean_err > self.max_reproj_error:
                continue

            points_3d.append(X)
            track_info.append({
                'views': list(track.keys()),
                'observations': observations,
                'mean_err': mean_err,
                'max_err': max_err,
                'track_length': len(track),
            })

        if all_mean_errs:
            all_mean_errs = np.array(all_mean_errs)
            all_max_errs = np.array(all_max_errs)
            print(f"Tracks cu DLT esuat        : {n_dlt_fail}")
            print(f"Tracks cu cheirality esuat : {n_cheirality_fail}")
            print(f"Tracks triangulate OK      : {len(all_mean_errs)}")
            print(f"Eroare MEDIE (mean) - min/mediana/max: "
                  f"{all_mean_errs.min():.2f} / {np.median(all_mean_errs):.2f} / {all_mean_errs.max():.2f} px")
            print(f"Eroare MAXIMA (max) - min/mediana/max: "
                  f"{all_max_errs.min():.2f} / {np.median(all_max_errs):.2f} / {all_max_errs.max():.2f} px")
            print(f"Puncte supravietuitoare la diferite praguri (mean_err <):")
            for thr in [2, 3, 5, 10, 20, 50, 100]:
                n_kept = int((all_mean_errs < thr).sum())
                print(f"      < {thr:>3} px : {n_kept} puncte")

        points_3d = np.asarray(points_3d) if points_3d else np.zeros((0, 3))
        return points_3d, track_info

    @staticmethod
    def extract_colors(track_info, images):
        """
        Pentru fiecare track, media culorii RGB la pozitiile keypoint-ilor.
        Imaginile vin BGR din cv2.imread -> convertim la RGB in [0,1].
        """
        colors = []
        for info in track_info:
            pix_colors = []
            for v_idx, (x, y) in info['observations']:
                img = images[v_idx]
                h, w = img.shape[:2]
                xi = int(round(x))
                yi = int(round(y))
                if 0 <= xi < w and 0 <= yi < h:
                    bgr = img[yi, xi]  # BGR
                    pix_colors.append([bgr[2], bgr[1], bgr[0]])  # -> RGB
            if pix_colors:
                mean_rgb = np.mean(pix_colors, axis=0) / 255.0
                colors.append(mean_rgb)
            else:
                colors.append([0.5, 0.5, 0.5])
        return np.asarray(colors)
