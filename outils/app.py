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

import streamlit as st

from analyse_positions import run_analysis

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'data')
RUNS_DIR = os.path.join(HERE, 'analyses')
REQUIRED_MODELS = ['football-player-detection.pt', 'football-pitch-detection.pt']

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


def process_video(video_path: str, run_dir: str, device: str, on_progress):
    """Video -> positions.csv + video radar. Retourne (csv_path, radar_path)."""
    import supervision as sv
    from radar_csv import run_radar_csv  # import tardif : charge les modeles IA

    csv_path = os.path.join(run_dir, 'positions.csv')
    radar_path = os.path.join(run_dir, 'video_radar.mp4')
    info = sv.VideoInfo.from_video_path(video_path)
    total = info.total_frames or 0

    frames = run_radar_csv(video_path, device, csv_path)
    with sv.VideoSink(radar_path, info) as sink:
        for i, frame in enumerate(frames, start=1):
            sink.write_frame(frame)
            on_progress(i, total)
    frames.close()  # referme proprement le fichier CSV
    return csv_path, radar_path


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
    st.caption('Le mieux : une vidéo de diffusion TV, avec le terrain bien visible. '
               'Commence par un extrait court pour tester.')
else:
    uploaded = st.file_uploader('Dépose ton fichier positions.csv', type=['csv'])

go = st.button("Lancer l'analyse", type='primary', disabled=uploaded is None)

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

            bar = st.progress(0.0, text='Étape 1/3 : reconnaissance des équipes '
                                        '(quelques minutes, sans barre de progression)…')

            def on_progress(done, total):
                if total:
                    bar.progress(min(done / total, 1.0),
                                 text=f'Étape 2/3 : analyse des images ({done}/{total})')
                else:
                    bar.progress(0.5, text=f'Étape 2/3 : analyse des images ({done})')

            csv_path, radar_path = process_video(source, run_dir, device, on_progress)
            bar.progress(1.0, text='Étape 3/3 : calcul des statistiques…')
        else:
            csv_path = os.path.join(run_dir, 'positions.csv')
            with open(csv_path, 'wb') as f:
                f.write(uploaded.getbuffer())

        result = run_analysis(
            csv_path, out_dir=os.path.join(run_dir, 'analyse'),
            min_frames=int(min_frames), max_gap=int(max_gap),
            outlier_cm=float(outlier_cm), jump_cm=float(jump_cm),
            min_segment=int(min_segment), log=lambda msg: None)
        result.update(run_dir=run_dir, radar_path=radar_path, raw_csv=csv_path)
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

    tab_eq, tab_pl, tab_maps, tab_notes, tab_dl = st.tabs(
        ['Équipes', 'Joueurs', 'Cartes', 'Méthode et limites', 'Téléchargements'])

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

    with tab_notes:
        for note in result['notes']:
            st.markdown('- ' + note)

    with tab_dl:
        if result['xlsx_path']:
            st.download_button(
                'Classeur Excel (analyse.xlsx)', read_bytes(result['xlsx_path']),
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
