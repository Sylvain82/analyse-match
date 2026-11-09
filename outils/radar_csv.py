"""
Mode RADAR + export CSV des positions.

A placer dans le meme dossier que main.py (examples/soccer). Ce script
n'utilise que des elements deja presents dans main.py : main.py n'est
pas modifie.

Exemple :
    python radar_csv.py --source_video_path data\\08fd33_0.mp4 ^
        --target_video_path data\\radar.mp4 ^
        --csv_path data\\positions.csv --device cuda

Le script fiabilise la projection sur le terrain (voir pitch_projection.py) et
expose les reglages de detection : python radar_csv.py --help

Colonnes du CSV (separateur ";", compatible Excel en francais) :
    frame        numero de l'image dans la video
    time_ms      instant correspondant, en millisecondes
    tracker_id   identifiant du joueur suivi par le tracker
    role         joueur / gardien / arbitre
    team         0 ou 1 (vide pour les arbitres)
    x_cm, y_cm   position sur le terrain, dans l'unite de la configuration
                 du depot (terrain de 12000 x 7000 attendu, a verifier)
    n_keypoints  nombre de points du terrain utilises pour la projection
                 (plus il est eleve, plus la position est fiable)
"""
import argparse
import csv
from typing import Iterator

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer

from pitch_projection import PitchProjector

from main import (
    COLORS,
    CONFIG,
    ELLIPSE_ANNOTATOR,
    ELLIPSE_LABEL_ANNOTATOR,
    GOALKEEPER_CLASS_ID,
    PITCH_DETECTION_MODEL_PATH,
    PLAYER_CLASS_ID,
    PLAYER_DETECTION_MODEL_PATH,
    REFEREE_CLASS_ID,
    STRIDE,
    get_crops,
    resolve_goalkeepers_team_id,
)

# Reglages de la detection et de la projection (modifiables en ligne de commande)
DEFAULT_OPTIONS = {
    'conf': 0.25,          # confiance minimale d'une detection de joueur
    'imgsz': 1280,         # taille de l'image donnee au modele (plus grand = joueurs lointains mieux vus)
    'nms_iou': 0.5,        # supprime les doublons joueur/gardien au meme endroit (0 = desactive)
    'lost_buffer': 60,     # images pendant lesquelles un joueur perdu peut etre retrouve (defaut d'origine : 30)
    'kp_conf': 0.5,        # confiance minimale d'un point du terrain
    'ransac_cm': 200.0,    # ecart tolere (cm) pour garder un point du terrain
    'max_drift_cm': 300.0, # saut maximal plausible de la projection entre 2 images
    'max_reject': 10,      # images rejetees d'affilee avant d'accepter un changement de plan
    'smoothing': 0.6,      # lissage de la projection (1 = aucun)
    'legacy': False,       # True = projection d'origine (pour comparer)
    'affiner_lignes': False,  # affine la projection sur les lignes blanches (experimental)
    'lissage_calibration': 2.0,  # lissage temporel de la calibration, en images (0 = aucun)
    'ignorer_debut_s': 0.0,      # secondes initiales ignorees avec une calibration externe
    'calibration': None,   # CSV d'homographies par image (pnl_export.py) : remplace la projection par points
}

ROLE_NAMES = {
    PLAYER_CLASS_ID: 'joueur',
    GOALKEEPER_CLASS_ID: 'gardien',
    REFEREE_CLASS_ID: 'arbitre',
}


def project_on_pitch(detections: sv.Detections, keypoints: sv.KeyPoints):
    """Projette les pieds des personnes detectees sur le plan du terrain.

    Retourne (positions, nombre_de_points_utilises). Les positions valent
    None quand moins de 4 points du terrain sont visibles : la projection
    est alors impossible pour cette image.
    """
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    n_points = int(mask.sum())
    if n_points < 4:
        return None, n_points
    transformer = ViewTransformer(
        source=keypoints.xy[0][mask].astype(np.float32),
        target=np.array(CONFIG.vertices)[mask].astype(np.float32)
    )
    xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    return transformer.transform_points(points=xy), n_points


def first_keypoints(keypoints: sv.KeyPoints):
    """Points du terrain de la 1re detection : (xy, confiance) ou (None, None)."""
    if keypoints.xy is None or len(keypoints.xy) == 0:
        return None, None
    conf = None
    if getattr(keypoints, 'confidence', None) is not None and len(keypoints.confidence):
        conf = keypoints.confidence[0]
    return keypoints.xy[0], conf


def make_tracker(fps: float, opts: dict):
    try:
        return sv.ByteTrack(
            lost_track_buffer=int(opts['lost_buffer']),
            frame_rate=int(round(fps)),
            minimum_consecutive_frames=3)
    except TypeError:  # version de supervision aux noms de parametres differents
        return sv.ByteTrack(minimum_consecutive_frames=3)


PNL_PITCH_CM = (10500.0, 6800.0)      # terrain standard de PnLCalib (105 x 68 m)


def _runs(valid):
    runs, start = [], None
    for i, v in enumerate(valid):
        if v and start is None:
            start = i
        if not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(valid)))
    return runs


def load_calibration(path: str, frame_wh=None, fps: float = 25.0, smooth_sigma: float = 2.0,
                     skip_start_s: float = 0.0, max_gap: int = 10, max_jump_cm: float = 300.0,
                     min_segment: int = 15) -> dict:
    """Homographies image -> terrain (cm) par image, produites par pnl_export.py.

    Nettoyage applique avant usage :
      - images implausibles ecartees (la projection d'une grille de l'image doit tomber
        pres du terrain) ;
      - trous de max_gap images ou moins combles par interpolation ;
      - coupure aux sauts (plus de max_jump_cm entre deux images) ; les etats de camera
        de moins de min_segment images sont abandonnes ;
      - lissage temporel (mediane sur 7 images puis gaussienne de largeur smooth_sigma) :
        la calibration image par image est bruitee, la camera, elle, bouge de facon continue ;
      - skip_start_s : secondes initiales ignorees (la calibration y est parfois instable).
    """
    from scipy.ndimage import gaussian_filter1d, median_filter
    rows = {}
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter=';'):
            if row.get('ok') == '1':
                H = np.array([[float(row[f'h{i}{j}']) for j in range(3)] for i in range(3)])
                if abs(H[2, 2]) > 1e-12:
                    rows[int(row['frame'])] = H / H[2, 2]
    if not rows:
        raise ValueError(f"Aucune image calibree dans {path}")

    n = max(rows) + 1
    params = np.full((n, 8), np.nan)
    grid = None
    if frame_wh is not None:
        w, h = frame_wh
        grid = np.array([[x * w, y * h] for y in (0.5, 0.7) for x in (0.2, 0.5, 0.8)],
                        np.float32).reshape(-1, 1, 2)
    for k, H in rows.items():
        if k < int(round(skip_start_s * fps)):
            continue
        if grid is not None:                       # plausibilite : l'image doit montrer le terrain, ou presque
            pts = cv2.perspectiveTransform(grid, H).reshape(-1, 2)
            margin = 2000.0                        # 20 m autour du terrain
            inside = ((pts[:, 0] > -margin) & (pts[:, 0] < PNL_PITCH_CM[0] + margin)
                      & (pts[:, 1] > -margin) & (pts[:, 1] < PNL_PITCH_CM[1] + margin))
            if not inside.all():
                continue
        params[k] = H.ravel()[:8]

    # trous courts : interpolation des 8 parametres
    valid = ~np.isnan(params[:, 0])
    idx = np.arange(n)
    for a, b in _runs(~valid):
        if a > 0 and b < n and b - a <= max_gap:
            for c in range(8):
                params[a:b, c] = np.interp(idx[a:b], [a - 1, b], [params[a - 1, c], params[b, c]])
    valid = ~np.isnan(params[:, 0])

    # coupure aux sauts de calibration : on ne moyenne jamais deux etats differents de la camera
    w_, h_ = frame_wh if frame_wh is not None else (1920, 1080)
    probe = np.array([[x * w_, y * h_] for y in (0.3, 0.5, 0.7) for x in (0.2, 0.5, 0.8)],
                     np.float32).reshape(-1, 1, 2)

    def projected(p):
        H = np.array([[p[0], p[1], p[2]], [p[3], p[4], p[5]], [p[6], p[7], 1.0]])
        return cv2.perspectiveTransform(probe, H).reshape(-1, 2)

    cuts = []
    for k in range(1, n):
        if valid[k] and valid[k - 1]:
            if np.median(np.hypot(*(projected(params[k]) - projected(params[k - 1])).T)) > max_jump_cm:
                cuts.append(k)
    cut_set = set(cuts)
    for a, b in _runs(valid):
        bounds = [a] + [k for k in range(a + 1, b) if k in cut_set] + [b]
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            if hi - lo < min_segment:
                params[lo:hi] = np.nan                     # etat de camera trop bref pour etre fiable
    valid = ~np.isnan(params[:, 0])

    # lissage temporel sur chaque tronçon continu
    if smooth_sigma > 0:
        for a, b in _runs(valid):
            bounds = [a] + [k for k in range(a + 1, b) if k in cut_set] + [b]
            for lo, hi in zip(bounds[:-1], bounds[1:]):
                seg = params[lo:hi]
                size = min(7, len(seg) if len(seg) % 2 else len(seg) - 1)
                if size >= 3:
                    seg = median_filter(seg, size=(size, 1), mode='nearest')
                params[lo:hi] = gaussian_filter1d(seg, smooth_sigma, axis=0, mode='nearest')

    out = {}
    for k in np.flatnonzero(valid):
        p = params[k]
        out[int(k)] = np.array([[p[0], p[1], p[2]], [p[3], p[4], p[5]], [p[6], p[7], 1.0]])
    if not out:
        raise ValueError("Aucune calibration plausible apres nettoyage")
    return out


def write_pitch_meta(csv_path: str, length_cm: float, width_cm: float, source: str) -> None:
    """Dimensions du terrain utilisees pour ce CSV, relues par analyse_positions.py."""
    import json
    with open(csv_path + '.meta.json', 'w', encoding='utf-8') as f:
        json.dump({'pitch_length_cm': length_cm, 'pitch_width_cm': width_cm, 'projection': source}, f)


def draw_radar(transformed_xy, color_lookup: np.ndarray) -> np.ndarray:
    radar = draw_pitch(config=CONFIG)
    if transformed_xy is None:
        return radar
    for team_id in range(4):
        radar = draw_points_on_pitch(
            config=CONFIG,
            xy=transformed_xy[color_lookup == team_id],
            face_color=sv.Color.from_hex(COLORS[team_id]),
            radius=20,
            pitch=radar)
    return radar


def _run_radar_csv(
    source_video_path: str,
    device: str,
    csv_path: str,
    options: dict,
    counts: dict
) -> Iterator[np.ndarray]:
    opts = {**DEFAULT_OPTIONS, **(options or {})}
    fps = sv.VideoInfo.from_video_path(source_video_path).fps
    refiner = None
    if opts['affiner_lignes'] and not opts['legacy']:
        from affinage_lignes import LineRefiner
        refiner = LineRefiner()
    projector = PitchProjector(
        CONFIG.vertices, kp_conf=opts['kp_conf'], ransac_cm=opts['ransac_cm'],
        max_drift_cm=opts['max_drift_cm'], max_reject=opts['max_reject'],
        smoothing=opts['smoothing'])

    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    calibration = None
    if opts['calibration']:
        vinfo = sv.VideoInfo.from_video_path(source_video_path)
        calibration = load_calibration(
            opts['calibration'], (vinfo.width, vinfo.height), fps,
            opts['lissage_calibration'], opts['ignorer_debut_s'])
    prev_H_ext, rejected_ext, calibration_gate = None, 0, False
    pitch_l_cm, pitch_w_cm = PNL_PITCH_CM if calibration is not None else (12000.0, 7000.0)
    write_pitch_meta(csv_path, pitch_l_cm, pitch_w_cm,
                     'calibration externe' if calibration is not None else 'points du terrain')
    pitch_detection_model = None if calibration is not None else YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)

    # 1re passe : on collecte des vignettes de joueurs pour apprendre les 2 equipes
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)
    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(
            frame, imgsz=opts['imgsz'], conf=opts['conf'], verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    # 2e passe : analyse complete, image par image
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = make_tracker(fps, opts)

    with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        writer.writerow([
            'frame', 'time_ms', 'tracker_id', 'role', 'team',
            'x_cm', 'y_cm', 'n_keypoints', 'pitch_l_cm', 'pitch_w_cm'])

        for frame_index, frame in enumerate(frame_generator):
            keypoints = None
            if pitch_detection_model is not None:
                result = pitch_detection_model(frame, verbose=False)[0]
                keypoints = sv.KeyPoints.from_ultralytics(result)
            result = player_detection_model(
                frame, imgsz=opts['imgsz'], conf=opts['conf'], verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
            if opts['nms_iou'] > 0:
                detections = detections.with_nms(
                    threshold=opts['nms_iou'], class_agnostic=True)
            detections = tracker.update_with_detections(detections)

            players = detections[detections.class_id == PLAYER_CLASS_ID]
            crops = get_crops(frame, players)
            players_team_id = team_classifier.predict(crops)

            goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
            goalkeepers_team_id = resolve_goalkeepers_team_id(
                players, players_team_id, goalkeepers)

            referees = detections[detections.class_id == REFEREE_CLASS_ID]

            detections = sv.Detections.merge([players, goalkeepers, referees])
            color_lookup = np.array(
                players_team_id.tolist() +
                goalkeepers_team_id.tolist() +
                [REFEREE_CLASS_ID] * len(referees)
            )
            labels = [str(tracker_id) for tracker_id in detections.tracker_id]

            # --- Positions sur le terrain + ecriture dans le CSV ---
            h, w, _ = frame.shape
            if calibration is not None:
                H_ext = calibration.get(frame_index)
                transformed_xy, n_points = None, 0
                if H_ext is not None and prev_H_ext is not None and rejected_ext < opts['max_reject']:
                    # une calibration qui saute d'une image a l'autre est rejetee, comme pour les points du terrain
                    if projector._drift_cm(prev_H_ext, H_ext, (w, h)) > opts['max_drift_cm']:
                        rejected_ext += 1
                        counts['rejetee'] += 1
                        H_ext = None
                        calibration_gate = True
                if H_ext is None:
                    if not calibration_gate:
                        counts['absente'] += 1
                    calibration_gate = False
                else:
                    rejected_ext = 0
                    prev_H_ext = H_ext
                    counts['ok'] += 1
                    anchors = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
                    transformed_xy = PitchProjector.transform(anchors, H_ext)
            elif opts['legacy']:
                transformed_xy, n_points = project_on_pitch(detections, keypoints)
                counts['ok' if transformed_xy is not None else 'absente'] += 1
            else:
                kp_xy, kp_conf = first_keypoints(keypoints)
                if kp_xy is None:
                    H, n_points, state = None, 0, 'absente'
                else:
                    H, n_points, state = projector.update(kp_xy, kp_conf, (w, h))
                counts[state] += 1
                transformed_xy = None
                if H is not None and refiner is not None:
                    conf_ok = np.ones(len(kp_xy), dtype=bool) if kp_conf is None else (kp_conf > opts['kp_conf'])
                    idx = np.flatnonzero((kp_xy[:, 0] > 1) & (kp_xy[:, 1] > 1) & conf_ok)
                    if len(idx):
                        proj_kp = projector.transform(kp_xy[idx], H)
                        vert = np.asarray(CONFIG.vertices, dtype=np.float32)[idx]
                        keep = np.hypot(proj_kp[:, 0] - vert[:, 0], proj_kp[:, 1] - vert[:, 1]) <= opts['ransac_cm']
                        H, ref_info = refiner.refine(frame, H, kp_xy[idx][keep], vert[keep])
                        counts['affine'] += int(ref_info['ok'])
                if H is not None:
                    anchors = detections.get_anchors_coordinates(
                        anchor=sv.Position.BOTTOM_CENTER)
                    transformed_xy = projector.transform(anchors, H)
            if transformed_xy is not None:
                time_ms = int(round(frame_index * 1000 / fps))
                for i in range(len(detections)):
                    class_id = int(detections.class_id[i])
                    is_referee = class_id == REFEREE_CLASS_ID
                    writer.writerow([
                        frame_index,
                        time_ms,
                        int(detections.tracker_id[i]),
                        ROLE_NAMES.get(class_id, str(class_id)),
                        '' if is_referee else int(color_lookup[i]),
                        int(round(transformed_xy[i][0])),
                        int(round(transformed_xy[i][1])),
                        n_points,
                        int(pitch_l_cm),
                        int(pitch_w_cm),
                    ])
                csv_file.flush()

            # --- Meme rendu video que le mode RADAR d'origine ---
            annotated_frame = frame.copy()
            annotated_frame = ELLIPSE_ANNOTATOR.annotate(
                annotated_frame, detections, custom_color_lookup=color_lookup)
            annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
                annotated_frame, detections, labels,
                custom_color_lookup=color_lookup)

            display_xy = transformed_xy
            if calibration is not None and transformed_xy is not None:      # le radar est dessine sur le gabarit 120 x 70 m
                display_xy = transformed_xy * np.array([12000.0 / PNL_PITCH_CM[0], 7000.0 / PNL_PITCH_CM[1]])
            radar = draw_radar(display_xy, color_lookup)
            radar = sv.resize_image(radar, (w // 2, h // 2))
            radar_h, radar_w, _ = radar.shape
            rect = sv.Rect(
                x=w // 2 - radar_w // 2,
                y=h - radar_h,
                width=radar_w,
                height=radar_h
            )
            annotated_frame = sv.draw_image(
                annotated_frame, radar, opacity=0.5, rect=rect)
            yield annotated_frame


def run_radar_csv(
    source_video_path: str,
    device: str,
    csv_path: str,
    options: dict = None
) -> Iterator[np.ndarray]:
    """Analyse une video : produit des images annotees et ecrit le CSV de positions."""
    counts = {'ok': 0, 'rejetee': 0, 'absente': 0, 'affine': 0}
    try:
        yield from _run_radar_csv(source_video_path, device, csv_path, options, counts)
    finally:
        total = counts['ok'] + counts['rejetee'] + counts['absente']
        if total:
            print(f"Projection sur le terrain : {counts['ok']} images utilisables, "
                  f"{counts['rejetee']} rejetees (projection douteuse), "
                  f"{counts['absente']} sans repere sur {total}.")
            if counts['affine']:
                print(f"Affinage sur les lignes blanches : {counts['affine']} images ajustees.")


def main(
    source_video_path: str,
    target_video_path: str,
    csv_path: str,
    device: str,
    preview: bool,
    options: dict = None
) -> None:
    frame_generator = run_radar_csv(
        source_video_path=source_video_path,
        device=device,
        csv_path=csv_path,
        options=options)

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in frame_generator:
            sink.write_frame(frame)
            if preview:
                cv2.imshow("frame", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        if preview:
            cv2.destroyAllWindows()

    frame_generator.close()  # referme proprement le fichier CSV
    print(f'Video enregistree : {target_video_path}')
    print(f'Positions enregistrees : {csv_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mode RADAR avec export CSV')
    parser.add_argument('--source_video_path', type=str, required=True)
    parser.add_argument('--target_video_path', type=str, required=True)
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument(
        '--no_preview', action='store_true',
        help="ne pas afficher la fenetre pendant le traitement (plus rapide)")
    d = DEFAULT_OPTIONS
    parser.add_argument('--conf', type=float, default=d['conf'],
                        help="confiance minimale des joueurs (defaut %(default)s)")
    parser.add_argument('--imgsz', type=int, default=d['imgsz'],
                        help="taille d'image du modele, ex. 1920 (defaut %(default)s)")
    parser.add_argument('--nms_iou', type=float, default=d['nms_iou'],
                        help="recouvrement au-dela duquel deux detections sont fusionnees, 0 = desactive")
    parser.add_argument('--lost_buffer', type=int, default=d['lost_buffer'],
                        help="images pendant lesquelles un joueur perdu est recherche (defaut %(default)s)")
    parser.add_argument('--kp_conf', type=float, default=d['kp_conf'],
                        help="confiance minimale des points du terrain (defaut %(default)s)")
    parser.add_argument('--ransac_cm', type=float, default=d['ransac_cm'],
                        help="ecart tolere des points du terrain, en cm (defaut %(default)s)")
    parser.add_argument('--max_drift_cm', type=float, default=d['max_drift_cm'],
                        help="saut maximal plausible de la projection entre 2 images, en cm")
    parser.add_argument('--smoothing', type=float, default=d['smoothing'],
                        help="lissage de la projection, 1 = aucun (defaut %(default)s)")
    parser.add_argument('--affiner_lignes', action='store_true',
                        help="affine la projection sur les lignes blanches (experimental, voir verifier_terrain.py --lignes)")
    parser.add_argument('--calibration', default=None,
                        help="CSV d'homographies par image produit par pnl_export.py (remplace la projection par points)")
    parser.add_argument('--lissage_calibration', type=float, default=d['lissage_calibration'],
                        help="lissage temporel de la calibration, en images (0 = aucun, defaut %(default)s)")
    parser.add_argument('--ignorer_debut_s', type=float, default=d['ignorer_debut_s'],
                        help="ignore les N premieres secondes avec une calibration externe (instable au debut)")
    parser.add_argument('--legacy', action='store_true',
                        help="projection d'origine, sans filtrage (pour comparer)")
    args = parser.parse_args()
    main(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        csv_path=args.csv_path,
        device=args.device,
        preview=not args.no_preview,
        options={
            'conf': args.conf, 'imgsz': args.imgsz, 'nms_iou': args.nms_iou,
            'lost_buffer': args.lost_buffer, 'kp_conf': args.kp_conf,
            'ransac_cm': args.ransac_cm, 'max_drift_cm': args.max_drift_cm,
            'smoothing': args.smoothing, 'legacy': args.legacy,
            'affiner_lignes': args.affiner_lignes, 'calibration': args.calibration,
            'lissage_calibration': args.lissage_calibration, 'ignorer_debut_s': args.ignorer_debut_s,
        }
    )
