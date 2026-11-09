"""
Export de la projection PnLCalib, image par image.

A placer dans le dossier PnLCalib (a cote de inference.py) et a lancer avec
l'environnement virtuel de PnLCalib. Il reprend exactement la logique de
inference.py, mais au lieu de dessiner les lignes sur la video, il ecrit pour
chaque image une homographie IMAGE -> TERRAIN, en centimetres, sur un terrain
standard de 105 x 68 m (10500 x 6800 cm), dans la convention de nos autres
scripts : x croit vers la droite de l'image, y vers le bas (le bord proche de la
camera est y = 6800).

    python pnl_export.py --weights_kp weights\\SV_kp --weights_line weights\\SV_lines ^
        --input_path C:\\...\\08fd33_0.mp4 --output_csv C:\\...\\calibration.csv --pnl_refine

Le CSV (separateur ";") contient : frame; ok; h00 ... h22; focale
Les images sans calibration valable ont ok = 0.
"""
import argparse
import csv

import cv2
import numpy as np

PITCH_L_M, PITCH_W_M = 105.0, 68.0       # terrain standard utilise par PnLCalib


def projection_matrix(cam_params: dict) -> np.ndarray:
    """Matrice de projection 3x4 (coordonnees du terrain centrees, en metres) -> pixels."""
    fx, fy = cam_params['x_focal_length'], cam_params['y_focal_length']
    pp = np.array(cam_params['principal_point'], dtype=float)
    position = np.array(cam_params['position_meters'], dtype=float)
    rotation = np.array(cam_params['rotation_matrix'], dtype=float)
    translation = np.eye(4)[:-1]
    translation[:, -1] = -position
    intrinsics = np.array([[fx, 0, pp[0]], [0, fy, pp[1]], [0, 0, 1]], dtype=float)
    return intrinsics @ (rotation @ translation)


def homography_image_to_pitch_cm(P: np.ndarray, length: float = PITCH_L_M, width: float = PITCH_W_M):
    """Homographie image -> terrain (cm), origine au coin, x vers la droite, y vers le bas.

    Le terrain etant symetrique, on choisit pour chaque image les symetries qui
    ramenent la convention de PnLCalib a la notre.
    """
    M = P[:, [0, 1, 3]]                      # plan du terrain (Z = 0)
    if abs(np.linalg.det(M)) < 1e-12:
        return None

    def to_image(x, y):
        v = M @ np.array([x - length / 2, y - width / 2, 1.0])
        return v[:2] / v[2]

    flip_x = to_image(0.0, width / 2)[0] > to_image(length, width / 2)[0]
    flip_y = to_image(length / 2, 0.0)[1] > to_image(length / 2, width)[1]

    shift = np.array([[1, 0, length / 2], [0, 1, width / 2], [0, 0, 1.0]])
    flips = np.array([[-1.0 if flip_x else 1.0, 0, length if flip_x else 0.0],
                      [0, -1.0 if flip_y else 1.0, width if flip_y else 0.0],
                      [0, 0, 1.0]])
    to_cm = np.diag([100.0, 100.0, 1.0])
    H = to_cm @ flips @ shift @ np.linalg.inv(M)
    if not np.all(np.isfinite(H)) or abs(H[2, 2]) < 1e-12:
        return None
    return H / H[2, 2]


def main():
    parser = argparse.ArgumentParser(description="Exporte la projection PnLCalib image par image")
    parser.add_argument('--weights_kp', required=True)
    parser.add_argument('--weights_line', required=True)
    parser.add_argument('--input_path', required=True)
    parser.add_argument('--output_csv', required=True)
    parser.add_argument('--kp_threshold', type=float, default=0.3434)
    parser.add_argument('--line_threshold', type=float, default=0.7867)
    parser.add_argument('--pnl_refine', action='store_true')
    parser.add_argument('--stride', type=int, default=1,
                        help="ne calibre qu'une image sur N (les autres sont marquees ok = 0)")
    args = parser.parse_args()

    # imports lourds ici : le reste du fichier reste utilisable sans PnLCalib
    import torch
    import torchvision.transforms as T
    import torchvision.transforms.functional as F
    import yaml
    from PIL import Image
    from tqdm import tqdm

    from model.cls_hrnet import get_cls_net
    from model.cls_hrnet_l import get_cls_net as get_cls_net_l
    from utils.utils_calib import FramebyFrameCalib
    from utils.utils_heatmap import (
        complete_keypoints, coords_to_dict,
        get_keypoints_from_heatmap_batch_maxpool, get_keypoints_from_heatmap_batch_maxpool_l)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = yaml.safe_load(open('config/hrnetv2_w48.yaml', 'r'))
    cfg_l = yaml.safe_load(open('config/hrnetv2_w48_l.yaml', 'r'))

    model = get_cls_net(cfg)
    model.load_state_dict(torch.load(args.weights_kp, map_location=device, weights_only=True))
    model.to(device).eval()
    model_l = get_cls_net_l(cfg_l)
    model_l.load_state_dict(torch.load(args.weights_line, map_location=device, weights_only=True))
    model_l.to(device).eval()
    resize = T.Resize((540, 960))

    cap = cv2.VideoCapture(args.input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cam = FramebyFrameCalib(iwidth=width, iheight=height, denormalize=True)

    def calibrate(frame_bgr):
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        tensor = F.to_tensor(image).float().unsqueeze(0)
        tensor = tensor if tensor.size()[-1] == 960 else resize(tensor)
        tensor = tensor.to(device)
        _, _, h, w = tensor.size()
        with torch.no_grad():
            heatmaps = model(tensor)
            heatmaps_l = model_l(tensor)
        kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:, :-1, :, :])
        line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:, :-1, :, :])
        kp_dict = coords_to_dict(kp_coords, threshold=args.kp_threshold)
        lines_dict = coords_to_dict(line_coords, threshold=args.line_threshold)
        kp_dict, lines_dict = complete_keypoints(kp_dict[0], lines_dict[0], w=w, h=h, normalize=True)
        cam.update(kp_dict, lines_dict)
        return cam.heuristic_voting(refine_lines=args.pnl_refine)

    header = ['frame', 'ok'] + [f'h{i}{j}' for i in range(3) for j in range(3)] + ['focale']
    n_ok = 0
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(header)
        for index in tqdm(range(total)):
            ok, frame = cap.read()
            if not ok:
                break
            H, focal = None, ''
            if index % max(args.stride, 1) == 0:
                try:
                    params = calibrate(frame)
                    if params is not None:
                        H = homography_image_to_pitch_cm(projection_matrix(params['cam_params']))
                        focal = f"{params['cam_params']['x_focal_length']:.1f}"
                except Exception as err:            # une image qui echoue ne doit pas arreter l'export
                    print(f'image {index} : {err}')
            if H is None:
                writer.writerow([index, 0] + [''] * 9 + [''])
            else:
                n_ok += 1
                writer.writerow([index, 1] + [f'{v:.9g}' for v in H.ravel()] + [focal])
    cap.release()
    print(f'\n{n_ok} images calibrees sur {total}. Fichier : {args.output_csv}')


if __name__ == '__main__':
    main()
