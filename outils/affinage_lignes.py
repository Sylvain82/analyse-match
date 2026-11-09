"""
Affinage de la projection a partir des lignes blanches du terrain.

Probleme : sur le bord proche de la camera, il n'existe presque aucun point
reconnaissable (la ligne de touche ne croise que la ligne mediane, les coins
sont souvent hors champ). La projection n'est alors tenue que par les points
du centre et du cote eloigne, et la ligne du bas penche.

Solution : on utilise la ligne blanche elle-meme. On detecte les pixels
blancs fins sur la pelouse, puis on ajuste legerement la projection pour que
les lignes du terrain reconstituees tombent sur ces pixels, tout en restant
coherent avec les points detectes. Aucun entrainement n'est necessaire.

Prudence : l'ajustement n'est accepte que s'il ameliore nettement
l'alignement et reste petit. Sinon, la projection d'origine est conservee.
"""
import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

L, W = 12000.0, 7000.0
PEN_W, PEN_L, GOAL_W, GOAL_L, CIRCLE_R = 4100.0, 2015.0, 1832.0, 550.0, 915.0


def pitch_polylines():
    """Lignes du terrain (cm) : contour, mediane, rond central, surfaces."""
    def rect(x0, y0, x1, y1):
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], np.float64)
    ang = np.linspace(0, 2 * np.pi, 97)
    circle = np.stack([L / 2 + CIRCLE_R * np.cos(ang), W / 2 + CIRCLE_R * np.sin(ang)], 1)
    py0, py1 = (W - PEN_W) / 2, (W + PEN_W) / 2
    gy0, gy1 = (W - GOAL_W) / 2, (W + GOAL_W) / 2
    return [rect(0, 0, L, W), np.array([[L / 2, 0], [L / 2, W]]), circle,
            rect(0, py0, PEN_L, py1), rect(L - PEN_L, py0, L, py1),
            rect(0, gy0, GOAL_L, gy1), rect(L - GOAL_L, gy0, L, gy1)]


def line_samples(step_cm: float = 60.0) -> np.ndarray:
    """Points regulierement espaces le long de toutes les lignes (cm)."""
    pts = []
    for poly in pitch_polylines():
        for a, b in zip(poly[:-1], poly[1:]):
            n = max(int(np.hypot(*(b - a)) / step_cm), 1)
            pts.append(a + (b - a) * (np.arange(n)[:, None] / n))
    return np.concatenate(pts).astype(np.float32)


def _odd(n: float, minimum: int = 3) -> int:
    n = max(int(round(n)), minimum)
    return n if n % 2 == 1 else n + 1


class LineRefiner:
    def __init__(self, work_width: int = 960, max_shift_px: float = 25.0,
                 min_gain: float = 0.05, kp_weight: float = 0.25, step_cm: float = 60.0,
                 min_grass: float = 0.65, tau_frac: float = 0.012, band_frac: float = 0.022):
        """
        work_width    largeur de travail (l'image est reduite pour aller plus vite)
        max_shift_px  deplacement maximal autorise des 4 points d'ancrage (a la taille de travail)
        min_gain      hausse minimale de la part d'echantillons qui tombent sur une ligne detectee
        kp_weight     poids des points detectes (garde la projection coherente avec eux)
        """
        self.work_width = work_width
        self.max_shift_px = max_shift_px
        self.min_gain = min_gain
        self.kp_weight = kp_weight
        self.min_grass = min_grass
        self.tau_frac = tau_frac
        self.band_frac = band_frac
        self.samples = line_samples(step_cm)

    # ---- carte des pixels de ligne blanche ----
    def line_mask(self, small: np.ndarray, band=None) -> np.ndarray:
        """Pixels de vraie ligne de terrain : fins, clairs, et entoures d'herbe.

        - canal minimum (R, V, B) : le blanc ressort, pas le vert de la pelouse ;
        - chapeau haut-de-forme etroit : ne garde que les structures fines (lignes), pas les joueurs ;
        - herbe de part et d'autre : ecarte le bord et le texte des panneaux publicitaires ;
        - 'band' (facultatif) : masque des zones proches de la projection actuelle.
        """
        hs, ws = small.shape[:2]
        low = small.min(axis=2)
        k = _odd(0.008 * ws, 5)
        tophat = cv2.morphologyEx(low, cv2.MORPH_TOPHAT,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        grass = cv2.inRange(hsv, (25, 30, 25), (95, 255, 255)) > 0
        r = max(int(round(0.008 * ws)), 3)
        around = cv2.boxFilter(grass.astype(np.float32), -1, (2 * r + 1, 2 * r + 1), normalize=True)
        zone = cv2.morphologyEx(grass.astype(np.uint8), cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(0.03 * ws, 7),) * 2)) > 0
        if zone.sum() < 0.05 * hs * ws:
            return np.zeros((hs, ws), np.uint8)
        thr = max(24.0, 3.0 * float(np.percentile(tophat[zone], 90)))
        mask = (tophat > thr) & (around >= self.min_grass)
        if band is not None:
            mask &= band
        return mask.astype(np.uint8)

    def distance_map(self, small: np.ndarray, tau: float, band=None) -> np.ndarray:
        lines = self.line_mask(small, band)
        if lines.sum() == 0:
            return np.full(small.shape[:2], tau, np.float32)
        dist = cv2.distanceTransform(1 - lines, cv2.DIST_L2, 3)
        return np.minimum(dist, tau).astype(np.float32)

    def refine(self, frame: np.ndarray, H: np.ndarray, kp_xy=None, kp_pitch=None):
        """H : homographie image -> terrain (cm), pleine resolution.

        kp_xy / kp_pitch : points du terrain detectes retenus (pixels, cm), facultatifs.
        Retourne (H_affinee, info). Si l'ajustement n'est pas fiable, H est rendue telle quelle.
        """
        h, w = frame.shape[:2]
        s = min(1.0, self.work_width / w)
        ws, hs = int(round(w * s)), int(round(h * s))
        small = cv2.resize(frame, (ws, hs), interpolation=cv2.INTER_AREA) if s < 1 else frame
        S = np.diag([s, s, 1.0])
        info = {'ok': False, 'raison': '', 'gain': 0.0, 'deplacement_px': 0.0}

        try:
            G0 = np.linalg.inv(H @ np.linalg.inv(S))             # terrain -> image reduite
        except np.linalg.LinAlgError:
            info['raison'] = 'homographie non inversible'
            return H, info

        img0 = cv2.perspectiveTransform(self.samples.reshape(-1, 1, 2).astype(np.float64), G0).reshape(-1, 2)
        margin = 0.01 * ws
        inside0 = ((img0[:, 0] > margin) & (img0[:, 0] < ws - 1 - margin)
                   & (img0[:, 1] > margin) & (img0[:, 1] < hs - 1 - margin))
        if inside0.sum() < 60:
            info['raison'] = 'trop peu de lignes visibles'
            return H, info
        samples = self.samples[inside0].astype(np.float32)

        tau = self.tau_frac * ws                                       # portee de recherche des lignes (~ 11 px a 960)
        band_px = int(round(self.band_frac * ws))                        # seules les zones proches de la projection comptent
        canvas = np.zeros((hs, ws), np.uint8)
        for poly in pitch_polylines():
            q = cv2.perspectiveTransform(poly.reshape(-1, 1, 2), G0).reshape(-1, 2)
            q = np.clip(q, -1e4, 1e4).astype(np.int32)
            cv2.polylines(canvas, [q.reshape(-1, 1, 2)], False, 255, 1)
        band = cv2.dilate(canvas, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_px + 1,) * 2)) > 0
        dist = self.distance_map(small, tau, band)

        a_img = np.array([[0.15 * ws, 0.30 * hs], [0.85 * ws, 0.30 * hs],
                          [0.85 * ws, 0.85 * hs], [0.15 * ws, 0.85 * hs]], np.float32)
        a_pitch = cv2.perspectiveTransform(a_img.reshape(-1, 1, 2).astype(np.float64),
                                           H @ np.linalg.inv(S)).reshape(-1, 2).astype(np.float32)

        use_kp = kp_xy is not None and kp_pitch is not None and len(kp_xy) >= 1
        if use_kp:
            kp_small = (np.asarray(kp_xy, np.float64) * s)
            kp_cm = np.asarray(kp_pitch, np.float64)

        def make_G(p):
            return cv2.getPerspectiveTransform(a_pitch, (a_img + p.reshape(4, 2)).astype(np.float32))

        def line_values(G):
            pts = cv2.perspectiveTransform(samples.reshape(-1, 1, 2).astype(np.float64), G).reshape(-1, 2)
            d = map_coordinates(dist, [pts[:, 1], pts[:, 0]], order=1, mode='nearest')
            ok = (pts[:, 0] >= 0) & (pts[:, 0] <= ws - 1) & (pts[:, 1] >= 0) & (pts[:, 1] <= hs - 1)
            return np.where(ok, d, tau)

        def residual(p):
            G = make_G(p)
            r = [line_values(G)]
            if use_kp:
                proj = cv2.perspectiveTransform(kp_cm.reshape(-1, 1, 2), G).reshape(-1, 2)
                r.append(self.kp_weight * (proj - kp_small).ravel())
            return np.concatenate(r)

        p0 = np.zeros(8)
        v0 = line_values(make_G(p0))
        c0, m0 = float(v0.mean()), int((v0 < 3.0).sum())       # distance moyenne ; echantillons presque sur une ligne
        if c0 < 1e-6:
            info['raison'] = 'deja aligne'
            return H, info
        def jacobian(p):                       # differences finies calculees ici (pas de pas relatif de scipy)
            base = residual(p)
            cols = []
            for i in range(8):
                q = p.copy()
                q[i] += 0.5
                cols.append((residual(q) - base) / 0.5)
            return np.stack(cols, 1)

        try:
            sol = least_squares(residual, p0, jac=jacobian, loss='soft_l1', f_scale=3.0,
                                bounds=(-self.max_shift_px * np.ones(8), self.max_shift_px * np.ones(8)),
                                max_nfev=40)
        except Exception as err:                                  # echec numerique : on garde l'original
            info['raison'] = f'echec : {err}'
            return H, info
        G1 = make_G(sol.x)
        v1 = line_values(G1)
        c1, m1 = float(v1.mean()), int((v1 < 3.0).sum())
        gain = (m1 - m0) / max(m0, 10)                         # part d'echantillons en plus qui tombent sur une ligne
        info.update(gain=gain, deplacement_px=float(np.abs(sol.x).max()) / s)
        if gain < self.min_gain or c1 > c0:
            info['raison'] = 'amelioration insuffisante'
            return H, info
        try:
            H_small = np.linalg.inv(G1)
        except np.linalg.LinAlgError:
            info['raison'] = 'homographie non inversible'
            return H, info
        H_new = H_small @ S
        H_new = H_new / H_new[2, 2]
        info.update(ok=True, raison='ok')
        return H_new, info
