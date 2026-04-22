import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class ReconstructionAnalyzer:
    """
    Analiza calitativa a reconstructiei:
    - histograma erorilor de reproiectie per punct
    - statistici per pereche de camere (matches brute, inliers, puncte partajate, eroare medie)
    - distributia lungimii tracks-urilor
    """

    @staticmethod
    def per_point_errors(track_info):
        mean_errs = np.array([t['mean_err'] for t in track_info])
        max_errs  = np.array([t['max_err']  for t in track_info])
        lengths   = np.array([t['track_length'] for t in track_info])
        return mean_errs, max_errs, lengths

    @staticmethod
    def per_pair_stats(track_info):
        """
        Pentru fiecare pereche (i,j) de vederi calculeaza:
         - numarul de puncte 3D observate in AMBELE vederi
         - eroarea medie de reproiectie (aproximata prin media track-ului)
        """
        stats = {}
        for info in track_info:
            views = sorted(info['views'])
            for a in range(len(views)):
                for b in range(a + 1, len(views)):
                    key = (views[a], views[b])
                    if key not in stats:
                        stats[key] = {'n_points': 0, 'err_sum': 0.0}
                    stats[key]['n_points'] += 1
                    stats[key]['err_sum'] += info['mean_err']
        for k in stats:
            n = stats[k]['n_points']
            stats[k]['mean_err'] = stats[k]['err_sum'] / n if n else 0.0
        return stats

    # ---------- PLOTS ----------
    @staticmethod
    def plot_error_histogram(track_info, save_path="data/output/reproj_hist.png"):
        if len(track_info) == 0:
            print("[Analyzer] Nu exista tracks de analizat")
            return

        mean_errs, max_errs, lengths = ReconstructionAnalyzer.per_point_errors(track_info)

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

        #eroare medie per punct
        axes[0].hist(mean_errs, bins=40, color='steelblue',
                     edgecolor='black', alpha=0.8)
        axes[0].axvline(mean_errs.mean(), color='red', linestyle='--',
                        label=f'media={mean_errs.mean():.2f}px')
        axes[0].axvline(np.median(mean_errs), color='orange', linestyle='--',
                        label=f'mediana={np.median(mean_errs):.2f}px')
        axes[0].set_xlabel('Eroare medie reproiectie (pixeli)')
        axes[0].set_ylabel('Numar puncte 3D')
        axes[0].set_title('Distributie eroare medie per punct')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        #eroare maxima per punct - indica outlierii
        axes[1].hist(max_errs, bins=40, color='tomato',
                     edgecolor='black', alpha=0.8)
        axes[1].axvline(max_errs.mean(), color='blue', linestyle='--',
                        label=f'media={max_errs.mean():.2f}px')
        axes[1].set_xlabel('Eroare maxima reproiectie (pixeli)')
        axes[1].set_ylabel('Numar puncte 3D')
        axes[1].set_title('Distributie eroare maxima per punct')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        #lungime track
        unique, counts = np.unique(lengths, return_counts=True)
        axes[2].bar(unique, counts, color='seagreen',
                    edgecolor='black', alpha=0.8)
        axes[2].set_xlabel('Lungime track (numar de vederi)')
        axes[2].set_ylabel('Numar puncte 3D')
        axes[2].set_title('Distributie lungime tracks')
        axes[2].set_xticks(unique)
        axes[2].grid(alpha=0.3, axis='y')
        for x, c in zip(unique, counts):
            axes[2].text(x, c, str(c), ha='center', va='bottom', fontsize=9)

        plt.suptitle('Analiza calitativa a reconstructiei',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Analyzer] Salvat: {save_path}")
        plt.show()

    # ---------- TEXT REPORT ----------
    @staticmethod
    def print_pair_statistics(pair_match_stats, track_info):
        print("\n" + "=" * 78)
        print("STATISTICI PER PERECHE DE CAMERE")
        print("=" * 78)
        print(f"{'Pereche':<10}{'raw matches':>14}{'inliers RANSAC':>18}"
              f"{'pts partajate 3D':>20}{'err medie (px)':>16}")
        print("-" * 78)

        pair_stats = ReconstructionAnalyzer.per_pair_stats(track_info)
        keys = sorted(set(list(pair_match_stats.keys()) + list(pair_stats.keys())))

        for (i, j) in keys:
            raw, inl = pair_match_stats.get((i, j), (0, 0))
            s = pair_stats.get((i, j), {'n_points': 0, 'mean_err': 0.0})
            print(f"({i},{j}){'':<6}{raw:>14}{inl:>18}"
                  f"{s['n_points']:>20}{s['mean_err']:>16.2f}")
        print("=" * 78)

    @staticmethod
    def print_global_summary(track_info, points_3d):
        if len(track_info) == 0:
            print("Nu s-au reconstruit puncte 3D")
            return

        mean_errs, max_errs, lengths = ReconstructionAnalyzer.per_point_errors(track_info)

        print("\n" + "=" * 60)
        print("SUMAR GLOBAL RECONSTRUCTIE MULTI-VIEW")
        print("=" * 60)
        print(f"Numar puncte 3D finale        : {len(points_3d)}")
        print(f"Lungime track medie           : {lengths.mean():.2f} vederi")
        print(f"Lungime track maxima          : {lengths.max()} vederi")
        print(f"Puncte cu >=4 vederi          : {(lengths >= 4).sum()}")
        print(f"Eroare reproiectie medie      : {mean_errs.mean():.2f} px")
        print(f"Eroare reproiectie mediana    : {np.median(mean_errs):.2f} px")
        print(f"Eroare reproiectie maxima     : {max_errs.max():.2f} px")

        #evaluare calitativa
        m = mean_errs.mean()
        if m < 1.0:
            verdict = "EXCELENT (< 1px) - reconstructie foarte precisa"
        elif m < 2.0:
            verdict = "FOARTE BUN (1-2px)"
        elif m < 3.0:
            verdict = "BUN (2-3px)"
        elif m < 5.0:
            verdict = "ACCEPTABIL (3-5px)"
        else:
            verdict = "SLAB (>5px) - verifica pose-urile sau distorsiunea"
        print(f"Calitate                      : {verdict}")
        print("=" * 60)
