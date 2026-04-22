import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import cv2


class BundleAdjuster:
    """
    Bundle Adjustment: optimizare neliniara globala care minimizeaza eroarea
    de reproiectie peste TOATE punctele 3D in TOATE vederile unde apar.

    Parametri optimizati:
      - pozitiile 3D ale punctelor (3 per punct)
      - pose-urile camerelor: rotatia (3 per camera, Rodrigues) + translatia (3 per camera)
        - camera 0 este INGHETATA (fixata la R=I, t=0) pt. a fixa gauge-ul

    Foloseste sparse Jacobian (obligatoriu pt. viteza cu mii de puncte).
    """

    def __init__(self, max_iterations=50, ftol=1e-6, verbose=True, refine_cameras=True):
        self.max_iterations = max_iterations
        self.ftol = ftol
        self.verbose = verbose
        self.refine_cameras = refine_cameras  # daca False, optimizam doar punctele

    #conversii
    @staticmethod
    def rotation_matrix_to_rvec(R):
        rvec, _ = cv2.Rodrigues(R)
        return rvec.flatten()

    @staticmethod
    def rvec_to_rotation_matrix(rvec):
        R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
        return R

    def _pack_params(self, cameras, points_3d):
        """
        Impacheteaza parametrii intr-un vector plat:
          [cam1_rvec, cam1_t, cam2_rvec, cam2_t, ..., X1, X2, ...]
        Camera 0 este exclusa (ancora).
        """
        n_cams_opt = len(cameras) - 1  # excludem cam 0
        n_points = len(points_3d)

        params = np.zeros(6 * n_cams_opt + 3 * n_points)

        #pose-uri camere (fara camera 0)
        for i, cam in enumerate(cameras[1:]):
            rvec = self.rotation_matrix_to_rvec(cam['R'])
            params[6 * i:6 * i + 3] = rvec
            params[6 * i + 3:6 * i + 6] = cam['t'].flatten()

        #puncte 3D
        params[6 * n_cams_opt:] = points_3d.flatten()
        return params

    def _unpack_params(self, params, cameras_fixed, n_points):
        """
        Desface vectorul plat inapoi in camere + puncte 3D.
        cameras_fixed contine K-urile originale (camera 0 ramane neschimbata).
        """
        n_cams = len(cameras_fixed)
        n_cams_opt = n_cams - 1

        cameras_new = [cameras_fixed[0]]  # camera 0 neschimbata

        for i in range(n_cams_opt):
            rvec = params[6 * i:6 * i + 3]
            t = params[6 * i + 3:6 * i + 6].reshape(3, 1)
            R = self.rvec_to_rotation_matrix(rvec)
            cam_new = cameras_fixed[i + 1].copy()
            cam_new['R'] = R
            cam_new['t'] = t
            cameras_new.append(cam_new)

        points_3d = params[6 * n_cams_opt:].reshape(n_points, 3)
        return cameras_new, points_3d

    def _residuals(self, params, cameras_fixed, observations, n_points):
        """
        Calculeaza vectorul de reziduuri (2 * n_observations).
        observations = list de (point_idx, view_idx, x_obs, y_obs)
        """
        cameras_new, points_3d = self._unpack_params(params, cameras_fixed, n_points)

        residuals = np.zeros(2 * len(observations))

        for k, (pt_idx, v_idx, x_obs, y_obs) in enumerate(observations):
            cam = cameras_new[v_idx]
            X = points_3d[pt_idx]
            Xc = cam['R'] @ X + cam['t'].flatten()
            if Xc[2] < 1e-6:
                #evitam diviziunea la 0 - penalizam
                residuals[2 * k] = 1e6
                residuals[2 * k + 1] = 1e6
                continue
            x_proj = cam['K'] @ Xc
            u = x_proj[0] / x_proj[2]
            v = x_proj[1] / x_proj[2]
            residuals[2 * k] = u - x_obs
            residuals[2 * k + 1] = v - y_obs

        return residuals

    def _jacobian_sparsity(self, observations, n_cameras, n_points):
        """
        Construim masca de sparsitate: care reziduu depinde de care parametru.
        Asta permite least_squares sa nu calculeze numeric Jacobiana completa.
        """
        n_cams_opt = n_cameras - 1
        m = 2 * len(observations)
        n = 6 * n_cams_opt + 3 * n_points

        A = lil_matrix((m, n), dtype=int)

        for k, (pt_idx, v_idx, _, _) in enumerate(observations):
            #reziduul k depinde de punctul 3D pt_idx (3 coloane)
            col_pt = 6 * n_cams_opt + 3 * pt_idx
            A[2 * k, col_pt:col_pt + 3] = 1
            A[2 * k + 1, col_pt:col_pt + 3] = 1

            #reziduul k depinde de camera v_idx (daca nu e camera 0)
            if v_idx > 0:
                col_cam = 6 * (v_idx - 1)
                A[2 * k, col_cam:col_cam + 6] = 1
                A[2 * k + 1, col_cam:col_cam + 6] = 1

        return A

    def run(self, cameras, points_3d, track_info):
        """
        Ruleaza BA. Returneaza (cameras_optimized, points_3d_optimized, stats).
        """
        #construim lista plata de observatii
        observations = []
        for pt_idx, info in enumerate(track_info):
            for v_idx, (x, y) in info['observations']:
                observations.append((pt_idx, v_idx, x, y))

        n_points = len(points_3d)
        n_cameras = len(cameras)
        n_obs = len(observations)

        if self.verbose:
            print(f"\n" + "=" * 60)
            print("BUNDLE ADJUSTMENT")
            print("=" * 60)
            print(f"Camere                  : {n_cameras} (cam 0 inghetata ca referinta)")
            print(f"Puncte 3D               : {n_points}")
            print(f"Observatii 2D           : {n_obs}")
            print(f"Parametri optimizati    : {6 * (n_cameras - 1) + 3 * n_points}")
            print(f"Reziduuri               : {2 * n_obs}")

        #pack initial
        params_init = self._pack_params(cameras, points_3d)

        #eroare initiala
        res_init = self._residuals(params_init, cameras, observations, n_points)
        err_init = np.sqrt((res_init ** 2).reshape(-1, 2).sum(axis=1))
        if self.verbose:
            print(f"\nEroare reproiectie initiala (RMS): {np.sqrt((res_init ** 2).mean()):.3f} px")
            print(f"Eroare reproiectie initiala (medie per obs): {err_init.mean():.3f} px")

        #sparse Jacobian
        A_sparse = self._jacobian_sparsity(observations, n_cameras, n_points)

        #optimizare
        if self.verbose:
            print(f"\nRulare Levenberg-Marquardt (max {self.max_iterations} iteratii)")

        result = least_squares(
            self._residuals,
            params_init,
            jac_sparsity=A_sparse,
            verbose=2 if self.verbose else 0,
            x_scale='jac',
            ftol=self.ftol,
            method='trf',  # Trust Region Reflective - robust
            max_nfev=self.max_iterations,
            args=(cameras, observations, n_points)
        )

        #unpack rezultat
        cameras_opt, points_3d_opt = self._unpack_params(result.x, cameras, n_points)

        #eroare finala
        res_final = result.fun
        err_final = np.sqrt((res_final ** 2).reshape(-1, 2).sum(axis=1))

        stats = {
            'initial_rms': float(np.sqrt((res_init ** 2).mean())),
            'final_rms': float(np.sqrt((res_final ** 2).mean())),
            'initial_mean_err': float(err_init.mean()),
            'final_mean_err': float(err_final.mean()),
            'n_iterations': result.njev,
            'success': result.success,
        }

        if self.verbose:
            print(f"\n" + "-" * 60)
            print(f"Eroare reproiectie finala (RMS):  {stats['final_rms']:.3f} px")
            print(f"Eroare reproiectie finala (medie): {stats['final_mean_err']:.3f} px")
            improvement = (1 - stats['final_mean_err'] / stats['initial_mean_err']) * 100
            print(f"Imbunatatire: {improvement:.1f}%")
            print("=" * 60)

        #actualizeaza si track_info cu erorile noi per punct
        track_info_updated = []
        per_point_errs = {}
        for k, (pt_idx, v_idx, _, _) in enumerate(observations):
            if pt_idx not in per_point_errs:
                per_point_errs[pt_idx] = []
            per_point_errs[pt_idx].append(err_final[k])

        for pt_idx, info in enumerate(track_info):
            errs = np.array(per_point_errs[pt_idx])
            new_info = dict(info)
            new_info['mean_err'] = float(errs.mean())
            new_info['max_err']  = float(errs.max())
            track_info_updated.append(new_info)

        return cameras_opt, points_3d_opt, track_info_updated, stats
