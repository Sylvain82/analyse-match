"""
Projection image -> terrain fiabilisee.

Le programme d'origine recalcule, a chaque image, une homographie a partir de
tous les points du terrain detectes, y compris les mauvais. Une seule erreur de
detection suffit a deplacer tous les joueurs de plusieurs metres sur le radar.

Cette classe corrige cela de trois facons :
  1. seuls les points detectes avec assez de confiance sont utilises ;
  2. RANSAC ecarte les points aberrants qui restent ;
  3. une homographie qui « saute » par rapport a l'image precedente est rejetee
     (l'image est alors ignoree plutot que fausse), et les homographies valides
     sont legerement lissees dans le temps.

Aucune dependance a roboflow/sports : seulement OpenCV et NumPy.
"""
import cv2
import numpy as np


class PitchProjector:
    def __init__(
        self,
        vertices,
        kp_conf: float = 0.5,
        ransac_cm: float = 200.0,
        max_drift_cm: float = 300.0,
        max_reject: int = 10,
        smoothing: float = 0.6,
    ):
        """
        vertices      coordonnees (cm) des points de reference du terrain
        kp_conf       confiance minimale d'un point detecte (0 = pas de filtre)
        ransac_cm     ecart tolere (cm, sur le terrain) pour garder un point
        max_drift_cm  deplacement maximal plausible de la projection entre
                      deux images consecutives
        max_reject    nombre d'images rejetees d'affilee au-dela duquel on
                      considere que la camera a vraiment change de plan
        smoothing     poids de la nouvelle homographie (1 = pas de lissage)
        """
        self.vertices = np.asarray(vertices, dtype=np.float32)
        self.kp_conf = kp_conf
        self.ransac_cm = ransac_cm
        self.max_drift_cm = max_drift_cm
        self.max_reject = max_reject
        self.smoothing = smoothing
        self.H = None
        self.rejected = 0
        self.missing = 0

    def reset(self) -> None:
        self.H = None
        self.rejected = 0
        self.missing = 0

    @staticmethod
    def _normalize(H: np.ndarray) -> np.ndarray:
        return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H

    @staticmethod
    def transform(points_xy: np.ndarray, H: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
        if len(pts) == 0:
            return pts
        out = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H.astype(np.float64))
        return out.reshape(-1, 2).astype(np.float32)

    def _drift_cm(self, H_a: np.ndarray, H_b: np.ndarray, frame_wh) -> float:
        """Ecart median (cm) entre deux homographies, mesure sur 9 points de l'image."""
        w, h = frame_wh
        xs, ys = np.array([0.2, 0.5, 0.8]) * w, np.array([0.3, 0.5, 0.7]) * h
        grid = np.array([[x, y] for y in ys for x in xs], dtype=np.float32)
        a, b = self.transform(grid, H_a), self.transform(grid, H_b)
        return float(np.median(np.hypot(a[:, 0] - b[:, 0], a[:, 1] - b[:, 1])))

    def update(self, kp_xy, kp_conf, frame_wh):
        """Traite une image.

        Retourne (H, n_points, etat) :
          etat 'ok'       -> H est utilisable pour projeter les joueurs
          etat 'rejetee'  -> projection douteuse, H vaut None : image ignoree
          etat 'absente'  -> pas assez de points du terrain, H vaut None
        """
        H, n_points, state = self._update(kp_xy, kp_conf, frame_wh)
        if state == 'absente':
            self.missing += 1
            if self.missing > self.max_reject:  # trop longtemps sans repere : on repart de zero
                self.H = None
        else:
            self.missing = 0
        return H, n_points, state

    def _update(self, kp_xy, kp_conf, frame_wh):
        kp_xy = np.asarray(kp_xy, dtype=np.float32)
        if kp_xy.ndim != 2 or len(kp_xy) != len(self.vertices):
            return None, 0, 'absente'

        mask = (kp_xy[:, 0] > 1) & (kp_xy[:, 1] > 1)
        if kp_conf is not None and self.kp_conf > 0:
            mask &= np.asarray(kp_conf, dtype=np.float32) > self.kp_conf
        if int(mask.sum()) < 4:
            return None, int(mask.sum()), 'absente'

        src, dst = kp_xy[mask], self.vertices[mask]
        H_new, inliers = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_cm)
        if H_new is None or inliers is None or int(inliers.sum()) < 4:
            return None, int(mask.sum()), 'absente'
        keep = inliers.ravel().astype(bool)
        n_in = int(keep.sum())

        # points alignes ou regroupes dans un coin de l'image : homographie instable
        hull = cv2.convexHull(src[keep])
        if cv2.contourArea(hull) < 0.005 * frame_wh[0] * frame_wh[1]:
            return None, n_in, 'rejetee'

        H_new = self._normalize(H_new)
        if self.H is not None and self.rejected < self.max_reject:
            if self._drift_cm(self.H, H_new, frame_wh) > self.max_drift_cm:
                self.rejected += 1
                return None, n_in, 'rejetee'

        self.rejected = 0
        if self.H is None or self.smoothing >= 1.0:
            self.H = H_new
        else:
            self.H = self._normalize(
                self.smoothing * H_new + (1.0 - self.smoothing) * self.H)
        return self.H, n_in, 'ok'
