"""
Application d'analyse de match de football (interface Streamlit, 100 % locale).

Tu deposes une video, l'application detecte les joueurs, les projette sur le
terrain, nettoie les trajectoires et te rend les statistiques (Excel, CSV,
cartes de chaleur, video radar).

A placer dans examples/soccer, a cote de main.py, radar_csv.py et
analyse_positions.py. Lancement : double-clic sur lancer_app.bat, ou

    pip install streamlit
    python -m streamlit run app.py --server.maxUploadSize 4000 --server.fileWatcherType none

Rien n'est envoye sur internet : tout est calcule sur ce PC.
"""
import os
import time
import traceback

import pandas as pd
import streamlit as st

from analyse_positions import TEAM_COLORS, draw_pitch, run_analysis, write_raw_excel

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'data')
RUNS_DIR = os.path.join(HERE, 'analyses')
REQUIRED_MODELS = ['football-player-detection.pt', 'football-pitch-detection.pt']
MAX_ROWS = 5000  # lignes affichees a l'ecran (les fichiers, eux, sont complets)

MODE_VIDEO = 'Une vidéo de match'
MODE_CSV = 'Un fichier positions.csv déjà produit'


def cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def missing_models() -> list:
    return [m for m in REQUIRED_MODELS if not os.path.exists(os.path.join(DATA_DIR, m))]


def trim_video(src: str, dst: str, seconds: float) -> None:
    """Copie les `seconds` premieres secondes de la video."""
    import supervision as sv
    info = sv.VideoInfo.from_video_path(src)
    end = max(int(seconds * info.fps), 1)
    frames = sv.get_video_frames_generator(src, end=end)
    with sv.VideoSink(dst, info) as sink:
        for frame in frames:
            sink.write_frame(frame)


def process_video(video_path: str, run_dir: str, device: str, on_progress, options=None):
    """Video -> positions.csv + video radar. Retourne (csv_path, radar_path)."""
    import supervision as sv
    from radar_csv import run_radar_csv  # import tardif : charge les modeles IA

    csv_path = os.path.join(run_dir, 'positions.csv')
    radar_path = os.path.join(run_dir, 'video_radar.mp4')
    info = sv.VideoInfo.from_video_path(video_path)
    total = info.total_frames or 0

    frames = run_radar_csv(video_path, device, csv_path, options)
    with sv.VideoSink(radar_path, info) as sink:
        for i, frame in enumerate(frames, start=1):
            sink.write_frame(frame)
            on_progress(i, total)
    frames.close()  # referme proprement le fichier CSV
    return csv_path, radar_path


@st.cache_data(show_spinner=False)
def load_csv(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path, sep=';')


def plot_frame(clean: pd.DataFrame, frame: int):
    """Terrain vu de dessus avec la position des joueurs a une image donnee."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=100)
    draw_pitch(ax)
    for r in clean[clean.frame == frame].itertuples():
        color = TEAM_COLORS.get(int(r.team), 'grey') if pd.notna(r.team) else 'grey'
        ax.scatter(r.x_cm, r.y_cm, s=90, c=color, marker='s' if r.role == 'gardien' else 'o',
                   edgecolors='white', zorder=3)
        ax.text(r.x_cm, r.y_cm - 170, str(int(r.tracker_id)), color='white',
                fontsize=7, ha='center', zorder=4)
    fig.tight_layout()
    return fig


def show_table(df: pd.DataFrame, key: str, csv_name: str):
    """Tableau filtrable + telechargement de la selection."""
    cols = st.columns(3)
    view = df
    if 'role' in df.columns:
        roles = sorted(df.role.dropna().unique())
        chosen = cols[0].multiselect('Rôle', roles, default=roles, key=key + '_role')
        view = view[view.role.isin(chosen)]
    if 'team' in df.columns:
        teams = sorted(df.team.dropna().unique())
        chosen = cols[1].multiselect('Équipe', teams, default=teams, key=key + '_team')
        view = view[view.team.isin(chosen) | view.team.isna()]
    ids = sorted(df.tracker_id.dropna().unique())
    chosen = cols[2].multiselect('Joueur (vide = tous)', ids, key=key + '_id')
    if chosen:
        view = view[view.tracker_id.isin(chosen)]
    st.caption(f'{len(view)} lignes sur {len(df)}'
               + (f' (les {MAX_ROWS} premières sont affichées)' if len(view) > MAX_ROWS else ''))
    st.dataframe(view.head(MAX_ROWS), hide_index=True)
    st.download_button('Télécharger cette sélection (CSV)',
                       view.to_csv(sep=';', index=False).encode('utf-8'),
                       file_name=csv_name, mime='text/csv', key=key + '_dl')


def read_bytes(path: str) -> bytes:
    with open(path, 'rb') as f:
        return f.read()


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
st.set_page_config(page_title='Analyse de match', page_icon='⚽', layout='wide')
st.title('⚽ Analyse de match')
st.caption('Tout est calculé sur ce PC : la vidéo et les données ne sont envoyées nulle part.')

with st.sidebar:
    st.header('Réglages')
    has_gpu = cuda_available()
    device = st.selectbox(
        'Calcul', ['cuda', 'cpu'] if has_gpu else ['cpu'],
        help='cuda = carte graphique Nvidia (beaucoup plus rapide).')
    if not has_gpu:
        st.warning('Aucune carte graphique Nvidia utilisable détectée : '
                   'le calcul sera lent.')
    seconds = st.number_input(
        'Analyser seulement les N premières secondes (0 = toute la vidéo)',
        min_value=0, value=0, step=10,
        help='Utile pour tester rapidement sur un extrait.')
    with st.expander('Réglages de la détection'):
        det_conf = st.slider('Confiance minimale des joueurs', 0.10, 0.60, 0.25, 0.05,
                             help='Plus bas : plus de joueurs détectés, mais plus de fausses détections.')
        det_imgsz = st.selectbox('Taille d\'image du modèle', [1280, 1920], index=0,
                                 help='1920 repère mieux les joueurs lointains, mais ralentit le calcul.')
        det_lines = st.checkbox('Affiner sur les lignes blanches (expérimental)', value=False,
                                help='Recale la projection sur les lignes visibles de l\'image, utile quand peu '
                                     'de points sont détectés sur le bord proche. À valider avec verifier_terrain.py --lignes.')
        calib_file = st.file_uploader(
            'Calibration PnLCalib (facultatif)', type=['csv'],
            help='Fichier produit par pnl_export.py (voir GUIDE_PNLCALIB.md). Il remplace la projection '
                 'par points du terrain : positions sur un terrain de 105 x 68 m, souvent plus justes sur le '
                 'bord proche de la caméra.')
        det_legacy = st.checkbox('Projection d\'origine (pour comparer)', value=False,
                                 help='Désactive le filtrage des points du terrain : utile pour mesurer '
                                      'l\'amélioration sur la même vidéo.')
    with st.expander('Réglages avancés du nettoyage'):
        min_frames = st.number_input('Suivi minimal (images)', 10, 500, 50)
        max_gap = st.number_input('Trou maximal comblé (images)', 0, 50, 10)
        outlier_cm = st.number_input('Écart aberrant (cm)', 30, 1000, 150)
        jump_cm = st.number_input('Saut impossible entre 2 images (cm)', 30, 1000, 150)
        min_segment = st.number_input('Tronçon minimal (images)', 5, 200, 25)

mode = st.radio('Que veux-tu analyser ?', [MODE_VIDEO, MODE_CSV], horizontal=True)
if mode == MODE_VIDEO:
    uploaded = st.file_uploader(
        'Dépose ta vidéo', type=['mp4', 'mov', 'avi', 'mkv'])
    st.caption('Glisse ta vidéo dans la zone ci-dessus (ou clique pour la choisir). Le mieux : '
               'une vidéo de diffusion TV, avec le terrain bien visible. Commence par un extrait court.')
    if uploaded is not None:
        with st.expander('Aperçu de la vidéo déposée'):
            st.video(uploaded)
else:
    uploaded = st.file_uploader('Dépose ton fichier positions.csv', type=['csv'])

st.caption("Un seul clic : détection des joueurs, Excel des données brutes, nettoyage, "
           "puis affichage des données nettoyées.")
go = st.button("Lancer l'analyse complète", type='primary', disabled=uploaded is None)

if go and uploaded is not None:
    st.session_state.pop('result', None)
    run_dir = os.path.join(RUNS_DIR, time.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(run_dir, exist_ok=True)
    radar_path = None
    try:
        if mode == MODE_VIDEO:
            missing = missing_models()
            if missing:
                raise FileNotFoundError(
                    'Modèles introuvables dans le dossier data : ' + ', '.join(missing))

            ext = os.path.splitext(uploaded.name)[1] or '.mp4'
            source = os.path.join(run_dir, 'source' + ext)
            with open(source, 'wb') as f:
                f.write(uploaded.getbuffer())
            if seconds > 0:
                trimmed = os.path.join(run_dir, 'extrait.mp4')
                trim_video(source, trimmed, seconds)
                source = trimmed

            bar = st.progress(0.0, text='Étape 1/4 : reconnaissance des équipes '
                                        '(quelques minutes, sans barre de progression)…')

            def on_progress(done, total):
                if total:
                    bar.progress(min(done / total, 1.0),
                                 text=f'Étape 2/4 : analyse des images ({done}/{total})')
                else:
                    bar.progress(0.5, text=f'Étape 2/4 : analyse des images ({done})')

            detection_options = {'conf': float(det_conf), 'imgsz': int(det_imgsz),
                                 'legacy': bool(det_legacy),
                                 'affiner_lignes': bool(det_lines)}
            if calib_file is not None:
                calib_path = os.path.join(run_dir, 'calibration.csv')
                with open(calib_path, 'wb') as f:
                    f.write(calib_file.getbuffer())
                detection_options['calibration'] = calib_path
            csv_path, radar_path = process_video(source, run_dir, device, on_progress, detection_options)
        else:
            csv_path = os.path.join(run_dir, 'positions.csv')
            with open(csv_path, 'wb') as f:
                f.write(uploaded.getbuffer())

        # Excel des données brutes, créé avant tout nettoyage
        raw_xlsx = os.path.join(run_dir, 'donnees_brutes.xlsx')
        if mode == MODE_VIDEO:
            bar.progress(1.0, text='Étape 3/4 : Excel des données brutes…')
        raw_ok = write_raw_excel(csv_path, raw_xlsx)

        if mode == MODE_VIDEO:
            bar.progress(1.0, text='Étape 4/4 : nettoyage et statistiques…')
        result = run_analysis(
            csv_path, out_dir=os.path.join(run_dir, 'analyse'),
            min_frames=int(min_frames), max_gap=int(max_gap),
            outlier_cm=float(outlier_cm), jump_cm=float(jump_cm),
            min_segment=int(min_segment), log=lambda msg: None)
        result.update(run_dir=run_dir, radar_path=radar_path, raw_csv=csv_path,
                      raw_xlsx=raw_xlsx if raw_ok else None)
        st.session_state['result'] = result
    except (ValueError, FileNotFoundError) as err:
        st.error(str(err))
    except Exception as err:  # erreur imprevue : on l'affiche pour pouvoir la corriger
        st.error(f"L'analyse a échoué : {err}")
        with st.expander("Détail de l'erreur (à me transmettre)"):
            st.code(traceback.format_exc())

result = st.session_state.get('result')
if result:
    st.success(f"Analyse terminée : {result['n_frames']} images "
               f"({result['n_frames'] / result['fps']:.0f} s). "
               f"Fichiers enregistrés dans : {result['run_dir']}")
    st.info("Les positions viennent d'une projection sur le terrain qui peut se tromper "
            "(changements de plan, zoom). Les statistiques d'équipe sont les plus fiables ; "
            "les distances et vitesses sont des estimations, la vitesse maximale surtout.")

    clean_df = load_csv(result['clean_csv'], os.path.getmtime(result['clean_csv']))
    raw_df = load_csv(result['raw_csv'], os.path.getmtime(result['raw_csv']))

    (tab_clean, tab_eq, tab_pl, tab_radar, tab_maps, tab_raw,
     tab_notes, tab_dl) = st.tabs(
        ['Données nettoyées', 'Équipes', 'Joueurs', 'Radar image par image', 'Cartes',
         'Données brutes', 'Méthode et limites', 'Téléchargements'])

    with tab_eq:
        st.dataframe(result['teams'], hide_index=True)
        per_second = result['per_second']
        if len(per_second):
            for col, label in [('largeur_m', 'Largeur de l\'équipe (m)'),
                               ('profondeur_m', 'Profondeur de l\'équipe (m)'),
                               ('compacite_m', 'Compacité (m)')]:
                st.subheader(label)
                chart = per_second.pivot(index='temps_s', columns='equipe', values=col)
                st.line_chart(chart.rename(columns=lambda c: f'Équipe {c}'))

    with tab_pl:
        st.dataframe(result['players'], hide_index=True)

    with tab_maps:
        cols = st.columns(len(result['images']))
        for col, img in zip(cols, result['images']):
            with col:
                st.image(img)

    with tab_radar:
        if clean_df.empty:
            st.info('Aucune position fiable à afficher.')
        else:
            f_min, f_max = int(clean_df.frame.min()), int(clean_df.frame.max())
            frame = st.slider('Image', f_min, f_max, f_min)
            st.caption(f'Instant : {frame / result["fps"]:.1f} s '
                       f'({int((clean_df.frame == frame).sum())} personnes avec une position fiable). '
                       'Carrés = gardiens, numéros = identifiants des joueurs.')
            fig = plot_frame(clean_df, frame)
            st.pyplot(fig)
            import matplotlib.pyplot as plt
            plt.close(fig)

    with tab_clean:
        st.caption('Positions fiables, lissées, en centimètres sur le terrain (x de 0 à 12000, '
                   'y de 0 à 7000), avec la vitesse et un indicateur de position reconstituée.')
        show_table(clean_df, 'clean', 'positions_nettoyees_selection.csv')

    with tab_raw:
        st.caption('Sortie directe de la détection, avant correction des équipes et nettoyage.')
        show_table(raw_df, 'raw', 'positions_selection.csv')

    with tab_notes:
        for note in result['notes']:
            st.markdown('- ' + note)

    with tab_dl:
        if result.get('raw_xlsx'):
            st.download_button(
                'Excel des données brutes (donnees_brutes.xlsx)', read_bytes(result['raw_xlsx']),
                file_name='donnees_brutes.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        if result['xlsx_path']:
            st.download_button(
                'Classeur Excel nettoyé et statistiques (analyse.xlsx)', read_bytes(result['xlsx_path']),
                file_name='analyse.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        st.download_button(
            'Positions nettoyées (CSV)', read_bytes(result['clean_csv']),
            file_name='positions_nettoyees.csv', mime='text/csv')
        st.download_button(
            'Positions brutes (CSV)', read_bytes(result['raw_csv']),
            file_name='positions.csv', mime='text/csv')
        if result.get('radar_path') and os.path.exists(result['radar_path']):
            st.download_button(
                'Vidéo avec radar (MP4)', read_bytes(result['radar_path']),
                file_name='video_radar.mp4', mime='video/mp4')
        st.subheader('Fichiers créés par cette analyse')
        files = []
        for root, _, names in os.walk(result['run_dir']):
            for name in sorted(names):
                path = os.path.join(root, name)
                files.append({'Fichier': os.path.relpath(path, result['run_dir']),
                              'Taille (Mo)': round(os.path.getsize(path) / 1e6, 2)})
        st.dataframe(pd.DataFrame(files), hide_index=True)
        st.caption(f"Dossier : {result['run_dir']}")
