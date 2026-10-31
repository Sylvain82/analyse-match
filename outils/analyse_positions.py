"""
Analyse du fichier positions.csv produit par radar_csv.py.

Ce que fait le script :
  1. corrige l'equipe et le role de chaque joueur (vote a la majorite sur
     l'ensemble de son suivi) ;
  2. nettoie les trajectoires (positions ramenees dans le terrain, petits
     trous combles, filtre median puis lissage) ;
  3. calcule par joueur : distance, vitesse moyenne et maximale ;
  4. calcule par equipe : position moyenne, largeur, profondeur, compacite ;
  5. produit des cartes de chaleur et un trace des trajectoires (PNG) ;
  6. ecrit un classeur Excel (analyse.xlsx) et un CSV de positions nettoyees.

Tout est calcule sur ton PC, aucune donnee n'est envoyee ailleurs.

Utilisation (depuis le dossier examples/soccer, environnement virtuel actif) :
    pip install openpyxl
    python analyse_positions.py --csv_path data\\positions.csv

Les resultats sont ecrits dans le dossier "analyse" a cote du CSV.
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')  # pas de fenetre : on enregistre directement des PNG
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Arc, Circle, Rectangle
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import savgol_filter

# Dimensions du terrain (cm), identiques a la configuration du depot roboflow
PITCH_L = 12000
PITCH_W = 7000
PENALTY_BOX_W, PENALTY_BOX_L = 4100, 2015
GOAL_BOX_W, GOAL_BOX_L = 1832, 550
CENTRE_CIRCLE_R = 915
PENALTY_SPOT = 1100

TEAM_COLORS = {0: '#00BFFF', 1: '#FF1493'}


# ----------------------------------------------------------------------------
# Lecture et correction des etiquettes
# ----------------------------------------------------------------------------
def load_positions(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=';')
    expected = {'frame', 'time_ms', 'tracker_id', 'role', 'team', 'x_cm', 'y_cm'}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV : {sorted(missing)}")
    return df


def estimate_fps(df: pd.DataFrame) -> float:
    f = df[['frame', 'time_ms']].drop_duplicates().sort_values('frame')
    d_frames = f.frame.iloc[-1] - f.frame.iloc[0]
    d_seconds = (f.time_ms.iloc[-1] - f.time_ms.iloc[0]) / 1000
    return float(d_frames / d_seconds) if d_seconds > 0 else 25.0


def fix_labels(df: pd.DataFrame):
    """Equipe et role retenus = valeur la plus frequente pour chaque tracker_id."""
    persons = df[df.role != 'arbitre']
    team_major = persons.groupby('tracker_id').team.agg(
        lambda s: s.dropna().mode().iloc[0] if s.notna().any() else np.nan)
    role_major = persons.groupby('tracker_id').role.agg(lambda s: s.mode().iloc[0])

    df = df.copy()
    df['team_corr'] = df.tracker_id.map(team_major)
    df['role_corr'] = np.where(
        df.role == 'arbitre', 'arbitre', df.tracker_id.map(role_major))
    n_team_fixed = int(((df.role != 'arbitre') & (df.team != df.team_corr)).sum())
    n_role_fixed = int(((df.role != 'arbitre') & (df.role != df.role_corr)).sum())
    return df, n_team_fixed, n_role_fixed


# ----------------------------------------------------------------------------
# Nettoyage des trajectoires
# ----------------------------------------------------------------------------
def odd(n: int) -> int:
    n = max(int(n), 3)
    return n if n % 2 == 1 else n + 1


def contiguous_runs(mask: np.ndarray):
    """Liste des (debut, fin_exclue) des series de True consecutifs."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    groups = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    return [(g[0], g[-1] + 1) for g in groups]


def outlier_mask(arr: np.ndarray, win: int, threshold_cm: float) -> np.ndarray:
    """Points trop eloignes de la mediane glissante de leur trajectoire.

    Une position qui s'ecarte de plusieurs metres de la mediane des images
    voisines est une erreur de detection ou de projection, pas un deplacement.
    """
    s = pd.DataFrame(arr, columns=['x', 'y'])
    med = s.rolling(win, center=True, min_periods=3).median()
    dev = np.hypot(s.x - med.x, s.y - med.y).to_numpy()
    return dev > threshold_cm


def smooth_segment(arr: np.ndarray, med_win: int, sg_win: int) -> np.ndarray:
    n = len(arr)
    out = arr.copy()
    if n >= 5:
        w = min(med_win, n if n % 2 == 1 else n - 1)
        out = median_filter(out, size=(w, 1), mode='nearest')
    w = min(sg_win, n if n % 2 == 1 else n - 1)
    if w >= 5:
        out = savgol_filter(out, window_length=w, polyorder=2, axis=0)
    return out


def clean_track(g: pd.DataFrame, max_gap: int, med_win: int, sg_win: int,
                hampel_win: int, hampel_cm: float, jump_cm: float, min_segment: int):
    """Nettoie une trajectoire.

    1. positions ramenees dans le terrain ;
    2. points aberrants ecartes (ecart a la mediane glissante) ;
    3. petits trous combles par interpolation ;
    4. trajectoire coupee a chaque saut impossible entre deux images
       (erreur de projection ou echange d'identifiant entre deux joueurs) ;
    5. chaque troncon continu d'au moins min_segment images est lisse ;
       les troncons plus courts sont abandonnes.
    """
    frames = np.arange(g.frame.min(), g.frame.max() + 1)
    xy = (g.drop_duplicates('frame').set_index('frame')[['x_cm', 'y_cm']]
          .astype(float).reindex(frames))
    xy['x_cm'] = xy.x_cm.clip(0, PITCH_L)
    xy['y_cm'] = xy.y_cm.clip(0, PITCH_W)

    raw = np.array(xy.to_numpy(), dtype=float)
    n_outliers = 0
    for _ in range(2):  # deux passes : une 1re erreur peut en masquer une autre
        bad = outlier_mask(raw, hampel_win, hampel_cm)
        n_outliers += int(bad.sum())
        raw[bad] = np.nan
    observed = ~np.isnan(raw[:, 0])

    xy = pd.DataFrame(raw, index=frames, columns=['x_cm', 'y_cm'])
    arr = np.array(xy.interpolate(method='linear', limit_area='inside').to_numpy(),
                   dtype=float)
    for a, b in contiguous_runs(~observed):  # trous trop longs : on ne comble pas
        if b - a > max_gap:
            arr[a:b] = np.nan
    valid = ~np.isnan(arr[:, 0])

    # coupure a chaque saut impossible entre deux images consecutives
    step = np.full(len(arr), 0.0)
    step[1:] = np.hypot(np.diff(arr[:, 0]), np.diff(arr[:, 1]))
    cut = np.zeros(len(arr), dtype=bool)
    cut[1:] = valid[1:] & valid[:-1] & (step[1:] > jump_cm)
    n_jumps = int(cut.sum())

    out = np.full_like(arr, np.nan)
    segment = np.full(len(arr), -1, dtype=int)
    seg_id = 0
    for a, b in contiguous_runs(valid):
        bounds = [a] + [i for i in range(a + 1, b) if cut[i]] + [b]
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            if hi - lo >= min_segment:
                out[lo:hi] = smooth_segment(arr[lo:hi], med_win, sg_win)
                segment[lo:hi] = seg_id
                seg_id += 1

    res = pd.DataFrame({
        'frame': frames,
        'x_cm': out[:, 0],
        'y_cm': out[:, 1],
        'interpole': (segment >= 0) & ~observed,
        'segment': segment,
    })
    return res, n_outliers, n_jumps


def track_metrics(res: pd.DataFrame, fps: float) -> dict:
    x, y = res.x_cm.to_numpy(), res.y_cm.to_numpy()
    seg = res.segment.to_numpy()
    same = (seg[1:] == seg[:-1]) & (seg[1:] >= 0)
    step_cm = np.where(same, np.hypot(np.diff(x), np.diff(y)), np.nan)
    speed_ms = np.r_[np.nan, step_cm * fps / 100]       # m/s, aligne sur les images
    res['vitesse_kmh'] = speed_ms * 3.6

    valid_frames = int((seg >= 0).sum())
    distance_m = float(np.nansum(step_cm) / 100)
    duration_s = valid_frames / fps
    win = max(int(round(1.0 * fps)), 3)
    v_roll = pd.Series(res.vitesse_kmh).rolling(win, center=True, min_periods=win // 2).mean()
    return {
        'images_validees': valid_frames,
        'couverture_pct': round(100 * valid_frames / max(len(res), 1), 1),
        'distance_validee_m': round(distance_m, 1),
        'distance_extrapolee_m': round(distance_m * len(res) / max(valid_frames, 1), 1),
        'vitesse_moy_kmh': round(distance_m / duration_s * 3.6, 1) if duration_s > 0 else np.nan,
        'vitesse_max_kmh': round(float(v_roll.max()), 1) if v_roll.notna().any() else np.nan,
        'pct_interpole': round(100 * float(res.interpole.sum()) / max(valid_frames, 1), 1),
    }


# ----------------------------------------------------------------------------
# Statistiques d'equipe
# ----------------------------------------------------------------------------
def team_frame_stats(pos: pd.DataFrame) -> pd.DataFrame:
    rows = []
    outfield = pos[(pos.role == 'joueur') & pos.x_cm.notna()]
    for (team, frame), g in outfield.groupby(['team', 'frame']):
        if len(g) < 3:
            continue
        x, y = g.x_cm.to_numpy(), g.y_cm.to_numpy()
        cx, cy = x.mean(), y.mean()
        rows.append({
            'equipe': int(team), 'frame': int(frame), 'joueurs': len(g),
            'centre_x_m': cx / 100, 'centre_y_m': cy / 100,
            'largeur_m': (y.max() - y.min()) / 100,
            'profondeur_m': (x.max() - x.min()) / 100,
            'compacite_m': float(np.hypot(x - cx, y - cy).mean() / 100),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Graphiques
# ----------------------------------------------------------------------------
def draw_pitch(ax):
    ax.add_patch(Rectangle((-300, -300), PITCH_L + 600, PITCH_W + 600,
                           facecolor='#2e7d32', edgecolor='none', zorder=0))
    kw = dict(fill=False, edgecolor='white', linewidth=1.5, zorder=1)
    ax.add_patch(Rectangle((0, 0), PITCH_L, PITCH_W, **kw))
    ax.plot([PITCH_L / 2] * 2, [0, PITCH_W], color='white', linewidth=1.5, zorder=1)
    ax.add_patch(Circle((PITCH_L / 2, PITCH_W / 2), CENTRE_CIRCLE_R, **kw))
    for side in (0, 1):
        x0 = 0 if side == 0 else PITCH_L - PENALTY_BOX_L
        ax.add_patch(Rectangle((x0, (PITCH_W - PENALTY_BOX_W) / 2),
                               PENALTY_BOX_L, PENALTY_BOX_W, **kw))
        x1 = 0 if side == 0 else PITCH_L - GOAL_BOX_L
        ax.add_patch(Rectangle((x1, (PITCH_W - GOAL_BOX_W) / 2),
                               GOAL_BOX_L, GOAL_BOX_W, **kw))
        spot_x = PENALTY_SPOT if side == 0 else PITCH_L - PENALTY_SPOT
        ax.add_patch(Circle((spot_x, PITCH_W / 2), 25, color='white', zorder=1))
    ax.set_xlim(-300, PITCH_L + 300)
    ax.set_ylim(PITCH_W + 300, -300)  # y vers le bas, comme dans la video
    ax.set_aspect('equal')
    ax.axis('off')


def save_heatmap(pos: pd.DataFrame, team: int, path: str):
    sel = pos[(pos.team == team) & (pos.role == 'joueur') & (pos.segment >= 0)]
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=100)
    draw_pitch(ax)
    if len(sel):
        bins_x = np.linspace(0, PITCH_L, 61)
        bins_y = np.linspace(0, PITCH_W, 36)
        h, _, _ = np.histogram2d(sel.x_cm, sel.y_cm, bins=[bins_x, bins_y])
        h = gaussian_filter(h.T, sigma=1.5)
        h = h / h.max() if h.max() > 0 else h
        ax.imshow(np.ma.masked_less(h, 0.04), extent=(0, PITCH_L, PITCH_W, 0),
                  cmap='YlOrRd', alpha=0.8, vmin=0, vmax=1, zorder=2,
                  interpolation='bilinear')
    ax.set_title(f"Carte de chaleur - equipe {team}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_trajectories(pos: pd.DataFrame, path: str):
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=100)
    draw_pitch(ax)
    valid = pos[pos.segment >= 0]
    for (tid, _), g in valid.groupby(['tracker_id', 'segment']):
        team = g.team.iloc[0]
        color = TEAM_COLORS.get(int(team), 'grey') if pd.notna(team) else 'grey'
        ax.plot(g.x_cm, g.y_cm, color=color, linewidth=1.2, zorder=2)
        ax.plot(g.x_cm.iloc[0], g.y_cm.iloc[0], 'o', color=color,
                markeredgecolor='white', markersize=3, zorder=3)
    ax.set_title("Trajectoires fiables (un point = debut d'un troncon continu)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Export
# ----------------------------------------------------------------------------
def write_excel(path, sheets: dict, notes: list, images: list) -> bool:
    try:
        import openpyxl  # noqa: F401
        from openpyxl.drawing.image import Image as XLImage
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError:
        return False

    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
        pd.DataFrame({'Notes': notes}).to_excel(writer, sheet_name='Notes', index=False)
        wb = writer.book

        header_fill = PatternFill('solid', fgColor='D9E1F2')
        for ws in wb.worksheets:
            for row in ws.iter_rows():
                for cell in row:
                    cell.font = Font(name='Arial', size=10, bold=(cell.row == 1))
            for cell in ws[1]:
                cell.fill = header_fill
                cell.alignment = Alignment(wrap_text=True, vertical='center')
            for i, col in enumerate(ws.columns, start=1):
                longest = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                limit = 110 if ws.title == 'Notes' else 24
                ws.column_dimensions[get_column_letter(i)].width = min(max(longest + 2, 10), limit)
            ws.freeze_panes = 'A2'

        ws = wb.create_sheet('Cartes')
        row = 1
        for img_path in images:
            img = XLImage(img_path)
            img.width, img.height = 640, 384
            ws.add_image(img, f'A{row}')
            row += 21
    return True


def run_analysis(csv_path: str, out_dir: str = None, min_frames: int = 50,
                 max_gap: int = 10, outlier_cm: float = 150.0, jump_cm: float = 150.0,
                 min_segment: int = 25, log=print) -> dict:
    """Analyse complete d'un CSV de positions. Retourne un dictionnaire de resultats.

    Utilisee par la ligne de commande (main) et par l'application (app.py).
    Leve ValueError si le fichier est inutilisable.
    """
    out_dir = out_dir or os.path.join(
        os.path.dirname(os.path.abspath(csv_path)), 'analyse')
    os.makedirs(out_dir, exist_ok=True)

    df = load_positions(csv_path)
    fps = estimate_fps(df)
    log(f'{len(df)} lignes, {df.frame.nunique()} images, {fps:.1f} images/s')

    df, n_team_fixed, n_role_fixed = fix_labels(df)
    log(f'Corrections : {n_team_fixed} equipes, {n_role_fixed} roles')

    med_win = odd(round(0.4 * fps))   # filtre median : ~0,4 s
    sg_win = odd(round(1.5 * fps))    # lissage : ~1,5 s
    hampel_win = odd(round(1.0 * fps))  # detection des points aberrants : ~1 s

    persons = df[df.role_corr != 'arbitre']
    tracks, metrics, ignored = [], [], []
    for tid, g in persons.groupby('tracker_id'):
        if g.frame.nunique() < min_frames:
            ignored.append((int(tid), int(g.frame.nunique())))
            continue
        res, n_out, n_jumps = clean_track(
            g, max_gap, med_win, sg_win, hampel_win, outlier_cm,
            jump_cm, min_segment)
        m = track_metrics(res, fps)
        m['positions_ecartees'] = n_out
        m['sauts_detectes'] = n_jumps
        team = g.team_corr.iloc[0]
        role = g.role_corr.iloc[0]
        res['tracker_id'] = int(tid)
        res['team'] = team
        res['role'] = role
        tracks.append(res)
        metrics.append({
            'tracker_id': int(tid), 'role': role,
            'equipe': int(team) if pd.notna(team) else None, **m,
        })

    if not tracks:
        raise ValueError('Aucun suivi assez long pour etre analyse.')

    pos = pd.concat(tracks, ignore_index=True)
    players = pd.DataFrame(metrics).sort_values(['equipe', 'role', 'tracker_id'])

    frames_stats = team_frame_stats(pos)
    team_rows = []
    for team in sorted(frames_stats.equipe.unique()):
        fs = frames_stats[frames_stats.equipe == team]
        pl = players[(players.equipe == team) & (players.role == 'joueur')]
        team_rows.append({
            'equipe': team,
            'joueurs_suivis': len(pl),
            'effectif_moyen_par_image': round(fs.joueurs.mean(), 1),
            'centre_x_moyen_m': round(fs.centre_x_m.mean(), 1),
            'centre_y_moyen_m': round(fs.centre_y_m.mean(), 1),
            'largeur_moyenne_m': round(fs.largeur_m.mean(), 1),
            'profondeur_moyenne_m': round(fs.profondeur_m.mean(), 1),
            'compacite_moyenne_m': round(fs.compacite_m.mean(), 1),
            'distance_validee_moy_m': round(pl.distance_validee_m.mean(), 1),
            'distance_extrapolee_moy_m': round(pl.distance_extrapolee_m.mean(), 1),
            'vitesse_moy_kmh': round(pl.vitesse_moy_kmh.mean(), 1),
            'couverture_moy_pct': round(pl.couverture_pct.mean(), 1),
        })
    teams = pd.DataFrame(team_rows)

    per_second = frames_stats[frames_stats.frame % int(round(fps)) == 0].copy()
    per_second.insert(1, 'temps_s', (per_second.frame / fps).round(1))
    per_second = per_second.drop(columns='frame').round(1)

    # Graphiques
    images = []
    for team in sorted(frames_stats.equipe.unique()):
        p = os.path.join(out_dir, f'carte_chaleur_equipe_{team}.png')
        save_heatmap(pos, team, p)
        images.append(p)
    p = os.path.join(out_dir, 'trajectoires.png')
    save_trajectories(pos, p)
    images.append(p)

    # Positions nettoyees (reutilisables pour d'autres analyses)
    clean = pos.loc[pos.segment >= 0, [
        'frame', 'tracker_id', 'role', 'team', 'segment', 'x_cm', 'y_cm',
        'vitesse_kmh', 'interpole']].copy()
    clean['time_s'] = (clean.frame / fps).round(2)
    clean[['x_cm', 'y_cm', 'vitesse_kmh']] = clean[['x_cm', 'y_cm', 'vitesse_kmh']].round(1)
    clean.to_csv(os.path.join(out_dir, 'positions_nettoyees.csv'),
                 sep=';', index=False)

    notes = [
        f'Source : {os.path.basename(csv_path)} - {df.frame.nunique()} images, '
        f'{fps:.1f} images/s, {df.frame.nunique() / fps:.1f} s de video.',
        f'Equipe et role : valeur majoritaire par tracker_id ({n_team_fixed} equipes et '
        f'{n_role_fixed} roles corriges). Les arbitres sont exclus de l\'analyse.',
        f'Suivis ignores car trop courts (< {min_frames} images) : '
        + (', '.join(f'n{t} ({n} images)' for t, n in ignored) or 'aucun') + '.',
        f'Positions ramenees dans le terrain (0-{PITCH_L} x 0-{PITCH_W} cm). Trous de '
        f'{max_gap} images ou moins combles par interpolation (colonne pct_interpole).',
        f'Positions aberrantes : un point a plus de {outlier_cm:.0f} cm de la mediane '
        'de sa trajectoire (sur ~1 s) est ecarte puis remplace par interpolation '
        '(colonne positions_ecartees).',
        f'Lissage : filtre median ({med_win} images) puis Savitzky-Golay ({sg_win} images). '
        'Sans lissage, le bruit de detection gonfle fortement les distances.',
        'Vitesse maximale : calculee sur une moyenne glissante de 1 s. Elle reste la mesure la '
        'moins fiable : une erreur de projection isolee peut la gonfler.',
        'Distance validee : somme des deplacements dans les troncons continus et fiables '
        '(couverture_pct = part de la video concernee). Distance extrapolee : cette distance '
        'ramenee a toute la duree du suivi, a titre indicatif seulement.',
        f'Troncons : la trajectoire est coupee a chaque deplacement de plus de {jump_cm:.0f} cm '
        f'en une image (erreur de projection ou echange d\'identifiant) ; les troncons de moins de '
        f'{min_segment} images sont abandonnes.',
        'Largeur et profondeur d\'equipe : ecart entre les joueurs de champ extremes '
        '(gardiens exclus). Compacite : distance moyenne des joueurs au centre de l\'equipe.',
        'Les orientations (x, y) sont celles du terrain projete depuis la video : le sens '
        'de jeu n\'est pas connu, donc "avant" et "arriere" ne sont pas distingues.',
        'Limites : une camera de diffusion ne montre pas toujours tout le terrain, et la '
        'projection peut dériver quand la camera zoome ou change de plan. Les distances '
        'sont des estimations, a prendre avec prudence.',
    ]

    sheets = {
        'Joueurs': players.rename(columns={
            'tracker_id': 'joueur_id', 'equipe': 'equipe'}),
        'Equipes': teams,
        'Equipes_par_seconde': per_second,
    }
    xlsx_path = os.path.join(out_dir, 'analyse.xlsx')
    excel_ok = write_excel(xlsx_path, sheets, notes, images)
    if excel_ok:
        log(f'Excel : {xlsx_path}')
    else:
        log("openpyxl n'est pas installe (pip install openpyxl) : ecriture en CSV.")
        for name, frame in sheets.items():
            frame.to_csv(os.path.join(out_dir, f'{name}.csv'), sep=';', index=False)

    return {
        'out_dir': out_dir,
        'players': players,
        'teams': teams,
        'per_second': per_second,
        'notes': notes,
        'images': images,
        'xlsx_path': xlsx_path if excel_ok else None,
        'clean_csv': os.path.join(out_dir, 'positions_nettoyees.csv'),
        'fps': fps,
        'n_frames': int(df.frame.nunique()),
    }


def main():
    parser = argparse.ArgumentParser(description='Analyse des positions de joueurs')
    parser.add_argument('--csv_path', required=True, help='CSV produit par radar_csv.py')
    parser.add_argument('--output_dir', default=None,
                        help='dossier de sortie (defaut : "analyse" a cote du CSV)')
    parser.add_argument('--min_frames', type=int, default=50,
                        help='suivis plus courts que ca : ignores (defaut 50 images)')
    parser.add_argument('--max_gap', type=int, default=10,
                        help='trou maximal comble par interpolation, en images (defaut 10)')
    parser.add_argument('--outlier_cm', type=float, default=150,
                        help='ecart a la mediane au-dela duquel une position est '
                             'consideree aberrante (defaut 150 cm)')
    parser.add_argument('--jump_cm', type=float, default=150,
                        help='deplacement maximal plausible entre 2 images (defaut 150 cm)')
    parser.add_argument('--min_segment', type=int, default=25,
                        help='longueur minimale d\'un troncon continu, en images (defaut 25)')
    args = parser.parse_args()
    try:
        result = run_analysis(
            args.csv_path, args.output_dir, args.min_frames, args.max_gap,
            args.outlier_cm, args.jump_cm, args.min_segment)
    except ValueError as err:
        raise SystemExit(str(err))

    print('\nRESUME PAR EQUIPE')
    print(result['teams'].to_string(index=False))
    print(f"\nSorties dans : {result['out_dir']}")


if __name__ == '__main__':
    main()
