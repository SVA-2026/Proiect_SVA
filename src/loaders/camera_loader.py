import json
import numpy as np


class CameraLoader:

    @staticmethod
    def load(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)

        cameras = []
        for entry in data:
            cam = entry['camera']
            h, w = entry['image_size']

            #param intrinseci
            f = cam['focal_length']
            cx = w / 2.0
            cy = h / 2.0
            K = np.array([ #transforma coordonatele din 3d in 2d
                [f, 0, cx],
                [0, f, cy],
                [0, 0, 1]
            ], dtype=np.float64)

            #param extrinseci
            q = cam['q']
            qw, qx, qy, qz = q[0], q[1], q[2], q[3]
            #se returneaza matricea de rotatia
            R = CameraLoader._quat_to_R(qw, qx, qy, qz)
            #translatia
            t = np.array(cam['t'], dtype=np.float64).reshape(3, 1)

            #impachetare
            cameras.append({
                'filename': entry['filename'],
                'K': K,
                'R': R,
                't': t,
                'f': f,
                'cx': cx,
                'cy': cy,
                'image_size': entry['image_size'],
            })

        return cameras

    @staticmethod
    def _quat_to_R(qw, qx, qy, qz):
        n = np.sqrt(qw ** 2 + qx ** 2 + qy ** 2 + qz ** 2) #normala cuaternionului
        #ne asiguram ca cuaternionul are lungimea 1
        qw /= n
        qx /= n
        qy /= n
        qz /= n
        return np.array([
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)]
        ], dtype=np.float64)

    @staticmethod
    def get_projection_matrix(cam_data):
        return cam_data['K'] @ np.hstack([cam_data['R'], cam_data['t']])

    @staticmethod
    def get_camera_center(cam_data):
        return (-cam_data['R'].T @ cam_data['t']).ravel()

    @staticmethod
    def normalize_to_first_camera(cameras):
        R0 = cameras[0]['R'].copy()
        t0 = cameras[0]['t'].copy()

        normalized = []
        for cam in cameras:
            R_new = cam['R'] @ R0.T
            t_new = cam['t'] - R_new @ t0

            new_cam = cam.copy()
            new_cam['R'] = R_new
            new_cam['t'] = t_new
            normalized.append(new_cam)

        #verificare- prima camera trebuie să fie R=I, t=0
        assert np.allclose(normalized[0]['R'], np.eye(3), atol=1e-6), \
            "Prima camera nu este identitate dupa normalizare!"
        assert np.allclose(normalized[0]['t'], 0, atol=1e-6), \
            "Prima camera nu e la origine dupa normalizare!"

        return normalized