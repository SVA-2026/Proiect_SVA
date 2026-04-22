import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class Plotter:

    @staticmethod
    def draw_matches(img1, kp1, img2, kp2, matches,
                     window_name="Feature Matches"):
        img_m = cv2.drawMatches(
            img1, kp1, img2, kp2, matches[:50], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1400, 600)
        cv2.imshow(window_name, img_m)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def plot_point_cloud(points_3d_list, title="Sparse 3D Point Cloud"):
        colors = ['steelblue', 'tomato', 'seagreen', 'darkorange',
                  'mediumpurple', 'gold', 'deeppink', 'cyan', 'lime', 'salmon']

        all_pts = np.vstack([p for p in points_3d_list if p is not None and len(p) > 0])

        center = np.median(all_pts, axis=0)
        std = np.std(all_pts, axis=0)

        keep = np.all(np.abs(all_pts - center) < 3 * std, axis=1)
        all_pts_clean = all_pts[keep]

        print(f"Puncte dupa curatare globala: "
              f"{len(all_pts_clean)} / {len(all_pts)}")

        fig = plt.figure(figsize=(14, 10))

        #toate perechile colorate diferit
        ax1 = fig.add_subplot(121, projection='3d')
        total = 0
        for idx, pts in enumerate(points_3d_list):
            if pts is None or len(pts) == 0:
                continue
            keep_i = np.all(np.abs(pts - center) < 3 * std, axis=1)
            pts_f = pts[keep_i]
            if len(pts_f) == 0:
                continue
            ax1.scatter(pts_f[:, 0], pts_f[:, 1], pts_f[:, 2],
                        c=colors[idx % len(colors)],
                        s=2, alpha=0.7,
                        label=f'P{idx + 1} ({len(pts_f)}pts)')
            total += len(pts_f)

        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_zlabel('Z (mm)')
        ax1.set_title(f'Pe perechi\n{total} puncte totale')
        ax1.legend(markerscale=4, fontsize=7, loc='upper left')

        ax2 = fig.add_subplot(122, projection='3d')
        z_vals = all_pts_clean[:, 2]
        z_norm = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min() + 1e-9)
        scatter = ax2.scatter(
            all_pts_clean[:, 0],
            all_pts_clean[:, 1],
            all_pts_clean[:, 2],
            c=z_norm, cmap='plasma',
            s=1.5, alpha=0.6
        )
        plt.colorbar(scatter, ax=ax2, label='Z normalizat', shrink=0.6)
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        ax2.set_zlabel('Z (mm)')
        ax2.set_title(f'Point cloud unit\nculoare = adâncime Z')

        fig.suptitle(title, fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig("data/output/point_cloud.png", dpi=150, bbox_inches='tight')
        print("[Plotter] Salvat: data/output/point_cloud.png")
        plt.show()

    @staticmethod
    def plot_camera_poses(camera_poses, title="Camera Poses"):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        positions = []
        for i, (R, t) in enumerate(camera_poses):
            C = (-R.T @ t).ravel()
            positions.append(C)
            ax.scatter(C[0], C[1], C[2], c='red', s=120, zorder=5)
            ax.text(C[0] + 0.5, C[1] + 0.5, C[2] + 0.5, f'Cam {i}',
                    fontsize=10, fontweight='bold')

            look = R.T @ np.array([0, 0, 5.0])
            ax.quiver(C[0], C[1], C[2],
                      look[0], look[1], look[2],
                      color='royalblue', linewidth=2)

        if len(positions) > 1:
            pos = np.array(positions)
            ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                    'r--', alpha=0.5, linewidth=1.5)

        # Marcăm originea (obiectul)
        ax.scatter(0, 0, 0, c='gold', s=200, marker='*',
                   zorder=10, label='Obiect (origine)')

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(title)
        ax.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig("data/output/camera_poses.png", dpi=150, bbox_inches='tight')
        print("[Plotter] Salvat: data/output/camera_poses.png")
        plt.show()
    #proiecteaza punctele 3D inapoi in imaginea 2D a camerei si le deseneaza peste ea
    @staticmethod
    def overlay_points_on_image(image, points_3d, cam_data, title="Puncte proiectate pe imagine"):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        #extragere date camera
        K = cam_data['K']
        R = cam_data['R']
        t = cam_data['t']

        #proiectare puncte 3D -> 2D
        #transformare puncte in coordonate omogene (4xN)
        pts3d_h = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))]).T

        #P = K * [R | t]
        Rt = np.hstack([R, t])
        P = K @ Rt

        #x = P * X
        pts2d_h = P @ pts3d_h

        #trecere la pixeli reali
        w = pts2d_h[2, :]

        #ignoram punctele care ar fi la infinit sau in spatele camerei
        valid_idx = w > 1e-6
        pts2d_h = pts2d_h[:, valid_idx]
        w = w[valid_idx]
        valid_points_3d = points_3d[valid_idx]

        px = pts2d_h[0, :] / w
        py = pts2d_h[1, :] / w

        h, w_img = image.shape[:2]
        fig, ax = plt.subplots(figsize=(12, 9))

        # desenam imaginea originala pe fundal
        ax.imshow(image)

        #mapam adancimea la o culoare pentru a fi mai clar
        depths = valid_points_3d[:, 2]  #folosim axa Z ca adancime
        norm = plt.Normalize(vmin=depths.min(), vmax=depths.max())
        cmap = cm.get_cmap('jet')
        colors = cmap(norm(depths))

        #desenam punctele proiectate peste imagine
        sc = ax.scatter(px, py, c=colors, marker='.', s=15, alpha=0.7, edgecolors='none')

        #limitam la dimensiunea imaginii si ascundem axele
        ax.set_xlim(0, w_img)
        ax.set_ylim(h, 0)  #inveersam axa Y pentru coordonate de pixel
        ax.axis('off')
        ax.set_title(title)

        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_colored_point_cloud(points_3d, colors_rgb, title="Colored 3D Point Cloud"):
        """Deseneaza un nor de puncte folosind culorile reale RGB."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Limităm axele pentru a elimina extremele care strica zoom-ul
        center = np.median(points_3d, axis=0)
        std = np.std(points_3d, axis=0)
        keep = np.all(np.abs(points_3d - center) < 3 * std, axis=1)

        pts_clean = points_3d[keep]
        colors_clean = colors_rgb[keep]

        # Desenam punctele (parametrul 'c' primeste matricea Nx3 cu valorile RGB)
        ax.scatter(
            pts_clean[:, 0],
            pts_clean[:, 1],
            pts_clean[:, 2],
            c=colors_clean,
            s=2.0,
            alpha=0.8,
            edgecolors='none'
        )

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(title)

        # Oprim putin perspectiva sa fie mai realista
        ax.view_init(elev=20, azim=-45)

        plt.tight_layout()
        plt.show()