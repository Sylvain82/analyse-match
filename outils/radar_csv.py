"""
Mode RADAR + export CSV des positions.

A placer dans le meme dossier que main.py (examples/soccer). Ce script
n'utilise que des elements deja presents dans main.py : main.py n'est
pas modifie.

Exemple :
    python radar_csv.py --source_video_path data\\08fd33_0.mp4 ^
        --target_video_path data\\radar.mp4 ^
        --csv_path data\\positions.csv --device cuda

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


def run_radar_csv(
    source_video_path: str,
    device: str,
    csv_path: str
) -> Iterator[np.ndarray]:
    fps = sv.VideoInfo.from_video_path(source_video_path).fps

    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)

    # 1re passe : on collecte des vignettes de joueurs pour apprendre les 2 equipes
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)
    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    # 2e passe : analyse complete, image par image
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)

    with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        writer.writerow([
            'frame', 'time_ms', 'tracker_id', 'role', 'team',
            'x_cm', 'y_cm', 'n_keypoints'])

        for frame_index, frame in enumerate(frame_generator):
            result = pitch_detection_model(frame, verbose=False)[0]
            keypoints = sv.KeyPoints.from_ultralytics(result)
            result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
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
            transformed_xy, n_points = project_on_pitch(detections, keypoints)
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
                    ])
                csv_file.flush()

            # --- Meme rendu video que le mode RADAR d'origine ---
            annotated_frame = frame.copy()
            annotated_frame = ELLIPSE_ANNOTATOR.annotate(
                annotated_frame, detections, custom_color_lookup=color_lookup)
            annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
                annotated_frame, detections, labels,
                custom_color_lookup=color_lookup)

            h, w, _ = frame.shape
            radar = draw_radar(transformed_xy, color_lookup)
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


def main(
    source_video_path: str,
    target_video_path: str,
    csv_path: str,
    device: str,
    preview: bool
) -> None:
    frame_generator = run_radar_csv(
        source_video_path=source_video_path,
        device=device,
        csv_path=csv_path)

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
    args = parser.parse_args()
    main(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        csv_path=args.csv_path,
        device=args.device,
        preview=not args.no_preview
    )
