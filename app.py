"""
Detector de Defeitos em Superfície de Aço — GC10-DET
Pipeline:
  1) EfficientNetV2 (gc10det_cls_ood.pt, 11 classes) → classificação + guard OOD
  2) YOLOv8 (yolo_detector.pt, 10 classes GC10-DET)  → detecção com bbox
  3) Limiares lidos de detection_metadata.json (ajustáveis na UI)
"""

import json
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import os
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# ─────────────────────────────────────────────────────────────────────────────
# Página
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Defeitos em Aço — GC10-DET", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.5rem; padding-bottom: 2.5rem; max-width: 1340px; }

.cabecalho-app {
    background: #1c2b3a; border-radius: 16px; padding: 2rem 2.4rem;
    margin-bottom: 1.8rem; display: flex; align-items: center;
    gap: 1.5rem; border-left: 5px solid #3d85c8;
}
.cabecalho-texto h1 { margin:0; font-size:1.85rem; font-weight:800; color:#f0f4f8; }
.cabecalho-texto p  { margin:0.3rem 0 0 0; font-size:0.95rem; color:#8daabf; }

.card-info  { background:#f7f9fb; border:1.5px solid #d5dde6; border-radius:14px; padding:1.2rem 1.3rem; }
.card-soft  { background:#ffffff; border:1.5px solid #dce5ef; border-radius:14px; padding:1rem 1.15rem; transition:box-shadow 0.2s; }
.card-soft:hover { box-shadow:0 2px 12px rgba(60,100,150,0.08); }

.bar-bg { background:#dde5ee; border-radius:999px; height:8px; overflow:hidden; margin-top:8px; }

.section-title {
    font-size:0.78rem; font-weight:700; color:#6b8399;
    letter-spacing:0.08em; text-transform:uppercase; margin-bottom:0.7rem;
}
.section-divider { border:none; border-top:1.5px solid #e4ecf4; margin:1.4rem 0; }
.small-muted { color:#7a90a4; font-size:0.88rem; }
.result-caption { text-align:center; color:#7a90a4; font-size:0.82rem; margin-top:0.3rem; font-style:italic; }

.stButton > button {
    background:#1c5d99; color:white; border:none; border-radius:10px;
    padding:0.6rem 1.2rem; font-weight:600; font-size:0.9rem;
    width:100%; transition:background 0.2s;
}
.stButton > button:hover { background:#164f85 !important; color:white !important; }

.stTabs [data-baseweb="tab-list"] { gap:4px; border-bottom:2px solid #e4ecf4; }
.stTabs [data-baseweb="tab"] {
    font-weight:600; font-size:0.9rem; padding:0.5rem 1.2rem;
    border-radius:8px 8px 0 0; color:#6b8399;
}
.stTabs [aria-selected="true"] { color:#1c5d99 !important; border-bottom:2px solid #1c5d99 !important; }

code { background:#edf2f7 !important; color:#2a7a4f !important;
    border-radius:5px; padding:1px 6px !important; font-size:0.83em !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""<div class="cabecalho-app">
    <div class="cabecalho-texto">
        <h1>🔩 Detector de Defeitos em Aço</h1>
        <p>Classificação e localização de defeitos superficiais em chapas de aço — GC10-DET</p>
    </div>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

DEFECT_CLASSES = [
    'punching_hole', 'welding_line', 'crescent_gap', 'water_spot', 'oil_spot',
    'silk_spot', 'inclusion', 'rolled_pit', 'crease', 'waist_folding',
]
CLASS_NAMES   = DEFECT_CLASSES + ['negative']
N_DEFECTS     = len(DEFECT_CLASSES)   # 10
N_CLASSES     = len(CLASS_NAMES)       # 11
OOD_CLASS_IDX = 10

CLASS_PT = {
    'punching_hole': 'Puncionamento',
    'welding_line':  'Linha de Solda',
    'crescent_gap':  'Fresta Crescente',
    'water_spot':    "Mancha d'Água",
    'oil_spot':      'Mancha de Óleo',
    'silk_spot':     'Mancha Seda',
    'inclusion':     'Inclusão',
    'rolled_pit':    'Cavidade Laminada',
    'crease':        'Dobra',
    'waist_folding': 'Dobra de Cintura',
    'negative':      'Fora do Domínio',
}

CLASS_DESC = {
    'punching_hole': 'Furo indesejado causado por falha mecânica na linha de produção.',
    'welding_line':  'Marca da emenda entre dois rolos de aço durante troca de bobina.',
    'crescent_gap':  'Lacuna em forma de meia-lua na superfície da chapa.',
    'water_spot':    'Mancha resultante de contato com umidade durante o processo.',
    'oil_spot':      'Contaminação por óleo lubrificante na superfície do aço.',
    'silk_spot':     'Mancha difusa e suave; o tipo mais frequente no dataset.',
    'inclusion':     'Partícula estranha embutida na superfície da chapa.',
    'rolled_pit':    'Pequena cavidade formada durante o processo de laminação.',
    'crease':        'Dobra localizada por deformação excessiva.',
    'waist_folding': 'Dobra de cintura, deformação mais pronunciada e severa.',
    'negative':      'Imagem fora do domínio industrial esperado para o modelo.',
}

CLASS_COLORS = {
    'punching_hole': (231, 76,  60),
    'welding_line':  (52,  152, 219),
    'crescent_gap':  (241, 196, 15),
    'water_spot':    (26,  188, 156),
    'oil_spot':      (160, 100, 45),
    'silk_spot':     (46,  204, 113),
    'inclusion':     (155, 89,  182),
    'rolled_pit':    (50,  50,  50),
    'crease':        (230, 126, 34),
    'waist_folding': (192, 57,  43),
    'negative':      (210, 80,  80),
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

VIDEO_EXTS = ['mp4', 'avi', 'mov', 'mkv', 'mpeg', 'mpg']
IMAGE_EXTS = ['png', 'jpg', 'jpeg', 'bmp']

# ─────────────────────────────────────────────────────────────────────────────
# Localização dos arquivos
# ─────────────────────────────────────────────────────────────────────────────
def _find(name):
    p = BASE_DIR / name
    return p if p.exists() else None

YOLO_PATH = _find('yolo_detector.pt')
CLS_PATH  = _find('gc10det_cls_ood.pt')
META_PATH = _find('detection_metadata.json')

# ─────────────────────────────────────────────────────────────────────────────
# Carregamento dos modelos
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_yolo():
    if YOLO_PATH is None:
        return None, "yolo_detector.pt não encontrado na pasta do app."
    try:
        from ultralytics import YOLO
        return YOLO(str(YOLO_PATH)), None
    except Exception as e:
        return None, str(e)


@st.cache_resource
def load_cls():
    if CLS_PATH is None:
        return None, None, "gc10det_cls_ood.pt não encontrado na pasta do app."
    try:
        import torch.nn as nn
        from torchvision.models import efficientnet_v2_s

        class GC10MultiOutputNet(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                backbone = efficientnet_v2_s(weights=None)
                in_feat  = backbone.classifier[1].in_features
                backbone.classifier = nn.Identity()
                self.backbone = backbone
                self.cls_head = nn.Sequential(
                    nn.Linear(in_feat, 1024), nn.ReLU(), nn.Dropout(0.5),
                    nn.Linear(1024, 512),     nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(512, num_classes),
                )
                self.bbox_head = nn.Sequential(
                    nn.Linear(in_feat, 1024), nn.ReLU(), nn.Dropout(0.5),
                    nn.Linear(1024, 512),     nn.ReLU(),
                    nn.Linear(512, 4),        nn.Sigmoid(),
                )

            def forward(self, x):
                feat = self.backbone(x)
                return self.cls_head(feat), self.bbox_head(feat)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        try:
            sd = torch.load(str(CLS_PATH), map_location=device, weights_only=True)
        except TypeError:
            sd = torch.load(str(CLS_PATH), map_location=device)

        # Detecta n_classes pelo shape da última camada do cls_head
        last_w = [k for k in sd if 'cls_head' in k and k.endswith('.weight')][-1]
        n_cls  = sd[last_w].shape[0]

        model = GC10MultiOutputNet(num_classes=n_cls)
        model.load_state_dict(sd)
        model.to(device).eval()
        return model, n_cls, None

    except Exception as e:
        return None, None, str(e)


@st.cache_data
def load_meta():
    if META_PATH and META_PATH.exists():
        try:
            return json.loads(META_PATH.read_text(encoding='utf-8'))
        except Exception:
            pass
    return {}


yolo_model, yolo_err          = load_yolo()
cls_model,  n_cls_ck, cls_err = load_cls()
det_meta                      = load_meta()

YOLO_OK   = yolo_model is not None
CLS_OK    = cls_model  is not None
OOD_ATIVO = CLS_OK and (n_cls_ck == 11)

# ─────────────────────────────────────────────────────────────────────────────
# Funções de inferência
# ─────────────────────────────────────────────────────────────────────────────
def render_bar(score, color_hex):
    width = max(0, min(100, round(score * 100)))
    st.markdown(
        f'<div class="bar-bg">'
        f'<div style="width:{width}%;background:{color_hex};height:10px;border-radius:999px;"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def classify_image(pil_img, ood_threshold):
    from torchvision import transforms

    device = next(cls_model.parameters()).device
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    img_rgb = pil_img.convert('RGB')
    w, h   = img_rgb.size
    tensor = tf(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        cls_out, bbox_out = cls_model(tensor)

    probs  = torch.softmax(cls_out, dim=1)[0].cpu().numpy()
    bbox_n = bbox_out[0].cpu().numpy()
    x1, y1 = int(bbox_n[0]*w), int(bbox_n[1]*h)
    x2, y2 = int(bbox_n[2]*w), int(bbox_n[3]*h)

    if OOD_ATIVO:
        ood_score    = float(probs[OOD_CLASS_IDX])
        is_ood       = ood_score >= ood_threshold
        defect_probs = probs[:OOD_CLASS_IDX]
    else:
        ood_score    = 0.0
        is_ood       = False
        defect_probs = probs[:N_DEFECTS]

    top_idx   = int(defect_probs.argmax())
    top_label = DEFECT_CLASSES[top_idx]
    top_score = float(defect_probs[top_idx])

    all_probs = sorted(
        [{'label': CLASS_NAMES[i], 'score': float(p)} for i, p in enumerate(probs)],
        key=lambda x: -x['score'],
    )

    return {
        'top_label': top_label,
        'top_score': top_score,
        'all_probs': all_probs,
        'cls_bbox':  [x1, y1, x2, y2],
        'ood_score': ood_score,
        'is_ood':    is_ood,
    }


def detect_with_yolo(pil_img, thresh):
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        pil_img.convert('RGB').save(tmp.name, 'JPEG')
        tmp_path = tmp.name

    results = yolo_model.predict(tmp_path, conf=thresh, imgsz=640, verbose=False)
    Path(tmp_path).unlink(missing_ok=True)

    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            score  = float(box.conf[0].item())
            if cls_id >= N_DEFECTS:
                continue
            label = DEFECT_CLASSES[cls_id]
            detections.append({
                'label':    label,
                'label_pt': CLASS_PT.get(label, label),
                'score':    score,
                'bbox':     box.xyxy[0].tolist(),
            })
    return detections


def draw_detections(pil_img, detections, line_width=3):
    img  = pil_img.convert('RGB').copy()
    draw = ImageDraw.Draw(img)
    w, _ = img.size

    try:
        font = ImageFont.truetype(
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', max(11, w // 24))
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
        color = CLASS_COLORS.get(det['label'], (255, 200, 0))
        text  = f"{det['label_pt']} {det['score']*100:.1f}%"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        try:
            bb = draw.textbbox((x1, y1), text, font=font)
            tw, th = bb[2]-bb[0], bb[3]-bb[1]
        except AttributeError:
            tw, th = draw.textsize(text, font=font)
        ty = max(0, y1 - th - 4)
        draw.rectangle([x1, ty, x1+tw+8, ty+th+4], fill=color)
        draw.text((x1+4, ty+2), text, fill=(255, 255, 255), font=font)
    return img


def extrair_frames_1fps(video_bytes, nome_arquivo='video', max_frames=120):
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(nome_arquivo).suffix or '.mp4') as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        Path(tmp_path).unlink(missing_ok=True)
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(int(round(fps)), 1)
    frames, frame_idx, segundo = [], 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append({'segundo': segundo, 'frame_idx': frame_idx, 'imagem': pil})
            segundo += 1
            if len(frames) >= max_frames:
                break
        frame_idx += 1

    cap.release()
    Path(tmp_path).unlink(missing_ok=True)
    return frames


def analisar_pipeline_imagem(imagem, limiar_cls, limiar_det, ood_threshold):
    cls_result = classify_image(imagem, ood_threshold)
    confiante  = (cls_result['top_score'] >= limiar_cls) and (not cls_result.get('is_ood', False))
    yolo_dets  = detect_with_yolo(imagem, limiar_det) if (confiante and YOLO_OK) else []
    return {'cls': cls_result, 'confiante': confiante, 'yolo_dets': yolo_dets}


# ─────────────────────────────────────────────────────────────────────────────
# Componentes de UI
# ─────────────────────────────────────────────────────────────────────────────
def render_resultado_classificacao(cls_result, confiante, limiar_cls, ood_threshold):
    top_pt    = CLASS_PT.get(cls_result['top_label'], cls_result['top_label'])
    top_score = cls_result['top_score']

    if cls_result.get('is_ood'):
        st.markdown(
            f'<div class="card-info">'
            f'<div class="small-muted">Resultado</div>'
            f'<div style="font-size:1.9rem;font-weight:800;color:#c0392b;margin-top:0.2rem;">⛔ Fora do domínio</div>'
            f'<div style="margin-top:0.4rem;color:#7a90a4;font-size:0.9rem;">'
            f'Score OOD: <strong>{cls_result["ood_score"]*100:.1f}%</strong> '
            f'(limiar: {ood_threshold*100:.0f}%)</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif confiante:
        st.markdown(
            f'<div class="card-info">'
            f'<div class="small-muted">Defeito detectado</div>'
            f'<div style="font-size:1.9rem;font-weight:800;color:#1c2b3a;margin-top:0.2rem;">{top_pt}</div>'
            f'<div style="margin-top:0.5rem;">',
            unsafe_allow_html=True,
        )
        render_bar(top_score, '#1c5d99')
        st.markdown(
            f'<div style="margin-top:0.4rem;font-size:1.3rem;font-weight:700;">'
            f'{top_score*100:.1f}%</div></div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="card-info">'
            f'<div class="small-muted">Inconclusivo</div>'
            f'<div style="font-size:1.9rem;font-weight:800;color:#1c2b3a;margin-top:0.2rem;">{top_pt}</div>'
            f'<div style="margin-top:0.5rem;">',
            unsafe_allow_html=True,
        )
        render_bar(top_score, '#aab8c8')
        st.markdown(
            f'<div style="margin-top:0.4rem;font-size:1.3rem;font-weight:700;">{top_score*100:.1f}%</div>'
            f'<div style="margin-top:0.3rem;color:#7a90a4;font-size:0.88rem;">'
            f'Abaixo do limiar de {limiar_cls*100:.0f}%</div></div></div>',
            unsafe_allow_html=True,
        )


def render_resultado_deteccao(cls_result, confiante, yolo_dets, limiar_det):
    if cls_result.get('is_ood'):
        st.markdown(
            '<div class="card-info"><div class="small-muted">Detecção</div>'
            '<div style="font-size:1.9rem;font-weight:800;color:#1c2b3a;margin-top:0.2rem;">Bloqueada</div>'
            '<div style="margin-top:0.4rem;color:#7a90a4;font-size:0.9rem;">'
            'Imagem fora do domínio industrial.</div></div>',
            unsafe_allow_html=True,
        )
    elif not confiante:
        st.markdown(
            '<div class="card-info"><div class="small-muted">Detecção</div>'
            '<div style="font-size:1.9rem;font-weight:800;color:#1c2b3a;margin-top:0.2rem;">Não executada</div>'
            '<div style="margin-top:0.4rem;color:#7a90a4;font-size:0.9rem;">'
            'Classificação abaixo do limiar.</div></div>',
            unsafe_allow_html=True,
        )
    elif not YOLO_OK:
        st.markdown(
            '<div class="card-info"><div class="small-muted">Detecção</div>'
            '<div style="font-size:1.9rem;font-weight:800;color:#1c2b3a;margin-top:0.2rem;">'
            'Modelo não disponível</div>'
            '<div style="margin-top:0.4rem;color:#7a90a4;font-size:0.9rem;">'
            'yolo_detector.pt não encontrado.</div></div>',
            unsafe_allow_html=True,
        )
    elif not yolo_dets:
        st.markdown(
            f'<div class="card-info"><div class="small-muted">Detecção</div>'
            f'<div style="font-size:1.9rem;font-weight:800;color:#1c2b3a;margin-top:0.2rem;">Nenhuma região</div>'
            f'<div style="margin-top:0.4rem;color:#7a90a4;font-size:0.9rem;">'
            f'Nenhuma detecção acima de {limiar_det*100:.0f}%.</div></div>',
            unsafe_allow_html=True,
        )
    else:
        best = max(yolo_dets, key=lambda d: d['score'])
        st.markdown(
            f'<div class="card-info">'
            f'<div class="small-muted">{len(yolo_dets)} região(ões) encontrada(s)</div>'
            f'<div style="font-size:1.9rem;font-weight:800;color:#1c2b3a;margin-top:0.2rem;">'
            f'{CLASS_PT.get(best["label"], best["label"])}</div>'
            f'<div style="margin-top:0.5rem;">',
            unsafe_allow_html=True,
        )
        render_bar(best['score'], '#1c5d99')
        st.markdown(
            f'<div style="margin-top:0.4rem;font-size:1.3rem;font-weight:700;">'
            f'{best["score"]*100:.1f}%</div></div></div>',
            unsafe_allow_html=True,
        )


def render_resultado_visual(nome, imagem, cls_result, confiante, yolo_dets,
                             limiar_cls, limiar_det, ood_threshold,
                             subtitulo='Imagem enviada'):
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown(f'### {nome}')
    col_img, col_cls, col_det = st.columns(3, gap='large')

    with col_img:
        st.markdown('<div class="section-title">Imagem analisada</div>', unsafe_allow_html=True)
        if confiante and YOLO_OK and yolo_dets:
            st.image(draw_detections(imagem, yolo_dets), use_container_width=True)
            st.markdown('<div class="result-caption">Com marcações YOLOv8</div>', unsafe_allow_html=True)
        else:
            st.image(imagem, use_container_width=True)
            st.markdown(f'<div class="result-caption">{subtitulo}</div>', unsafe_allow_html=True)

    with col_cls:
        st.markdown('<div class="section-title">Classificação (EfficientNetV2)</div>', unsafe_allow_html=True)
        render_resultado_classificacao(cls_result, confiante, limiar_cls, ood_threshold)

    with col_det:
        st.markdown('<div class="section-title">Detecção (YOLOv8)</div>', unsafe_allow_html=True)
        render_resultado_deteccao(cls_result, confiante, yolo_dets, limiar_det)


# ─────────────────────────────────────────────────────────────────────────────
# Estado da sessão
# ─────────────────────────────────────────────────────────────────────────────
for key, default in [('resultado', None), ('arquivo_nome', None), ('uploader_key', 0)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────────────────────────────────────
# Abas
# ─────────────────────────────────────────────────────────────────────────────
aba_analise, aba_info, aba_ex_imagens = st.tabs(['Análise', 'Sobre o Dataset', 'Exemplos'])

# ── Aba Análise ───────────────────────────────────────────────────────────────
with aba_analise:

    if not CLS_OK:
        st.error(
            f"Modelo de classificação não carregado. "
            f"{'Erro: ' + cls_err if cls_err else 'Coloque gc10det_cls_ood.pt na pasta do app.'}"
        )
        st.stop()

    if not YOLO_OK:
        st.warning(
            f"Modelo YOLO não carregado — detecção com bbox desabilitada. "
            f"{'Erro: ' + yolo_err if yolo_err else 'Coloque yolo_detector.pt na pasta do app.'}"
        )

    if not OOD_ATIVO:
        st.warning(
            "⚠️ Modelo com 10 classes (sem OOD). "
            "Imagens fora do domínio não serão rejeitadas. "
            "Use gc10det_cls_ood.pt para habilitar."
        )

    # Parâmetros — inclui limiar OOD ajustável na UI
    st.markdown('<div class="section-title">Parâmetros de confiança</div>', unsafe_allow_html=True)
    cfg1, cfg2, cfg3, cfg4 = st.columns(4)
    with cfg1:
        limiar_cls_pct = st.number_input(
            "Confiança mínima — classificação (%)", 1, 99, 40, 1,
            help="Abaixo deste valor o resultado é inconclusivo e o YOLO não roda.",
        )
    with cfg2:
        limiar_det_pct = st.number_input(
            "Confiança mínima — detecção YOLOv8 (%)", 1, 99, 25, 1,
            help="Detecções do YOLO abaixo deste valor são descartadas.",
        )
    with cfg3:
        ood_default = int(det_meta.get('ood_threshold', 0.40) * 100)
        ood_pct = st.number_input(
            "Limiar OOD (%)", 1, 99, ood_default, 5,
            help="Score mínimo de 'negative' para rejeitar como fora do domínio. "
                 "Aumente se imagens de aço estiverem sendo rejeitadas erroneamente.",
        )
    with cfg4:
        max_frames_video = st.number_input("Máximo de frames por vídeo", 10, 600, 120, 10)

    limiar_cls    = limiar_cls_pct / 100.0
    limiar_det    = limiar_det_pct / 100.0
    ood_threshold = ood_pct / 100.0

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    arquivos = st.file_uploader(
        "Envie imagens ou vídeo para análise",
        type=IMAGE_EXTS + VIDEO_EXTS,
        accept_multiple_files=True,
        key=f'upload_{st.session_state.uploader_key}',
    )

    if not arquivos:
        st.session_state.resultado    = None
        st.session_state.arquivo_nome = None
    else:
        nomes = [a.name for a in arquivos]
        if nomes != st.session_state.arquivo_nome:
            st.session_state.resultado    = None
            st.session_state.arquivo_nome = nomes

    if arquivos:
        b1, b2, _ = st.columns([1, 1, 5])
        with b1:
            analisar = st.button("Analisar arquivos", key="btn_analisar")
        with b2:
            limpar = st.button("Limpar", key="btn_limpar_top")

        if limpar:
            st.session_state.resultado    = None
            st.session_state.arquivo_nome = None
            st.session_state.uploader_key += 1
            st.rerun()

        if analisar and st.session_state.resultado is None:
            resultados = []
            with st.spinner("Executando análise..."):
                for arquivo in arquivos:
                    ext = Path(arquivo.name).suffix.lower().lstrip('.')

                    if ext in IMAGE_EXTS:
                        imagem  = Image.open(arquivo).convert('RGB')
                        analise = analisar_pipeline_imagem(
                            imagem, limiar_cls, limiar_det, ood_threshold)
                        resultados.append({
                            'tipo': 'imagem', 'nome': arquivo.name,
                            'imagem': imagem.copy(), **analise,
                        })

                    elif ext in VIDEO_EXTS:
                        frames     = extrair_frames_1fps(
                            arquivo.read(), arquivo.name, int(max_frames_video))
                        frames_res = []
                        for item in frames:
                            analise = analisar_pipeline_imagem(
                                item['imagem'], limiar_cls, limiar_det, ood_threshold)
                            frames_res.append({**item, **analise})

                        resumo = Counter()
                        for fr in frames_res:
                            if fr['cls'].get('is_ood'):
                                resumo['Fora do domínio'] += 1
                            else:
                                resumo[CLASS_PT.get(
                                    fr['cls']['top_label'], fr['cls']['top_label'])] += 1

                        resultados.append({
                            'tipo': 'video', 'nome': arquivo.name,
                            'frames': frames_res, 'total_frames': len(frames_res),
                            'resumo_classes': dict(resumo),
                        })

            st.session_state.resultado = resultados
            st.rerun()

    if arquivos and st.session_state.resultado is not None:
        for item in st.session_state.resultado:
            if item['tipo'] == 'imagem':
                render_resultado_visual(
                    nome=item['nome'], imagem=item['imagem'],
                    cls_result=item['cls'], confiante=item['confiante'],
                    yolo_dets=item['yolo_dets'],
                    limiar_cls=limiar_cls, limiar_det=limiar_det,
                    ood_threshold=ood_threshold,
                )

            elif item['tipo'] == 'video':
                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                st.markdown(f"### {item['nome']}")
                st.markdown(f"**Frames analisados:** {item['total_frames']}")

                if item['resumo_classes']:
                    df_r = pd.DataFrame([
                        {'Classe': k, 'Frames': v}
                        for k, v in item['resumo_classes'].items()
                    ])
                    st.dataframe(df_r, use_container_width=True, hide_index=True)

                frames_com = [fr for fr in item['frames'] if fr['yolo_dets']]
                st.markdown('#### Frames com detecção')
                if not frames_com:
                    st.info("Nenhum frame com detecção YOLOv8 acima do limiar.")
                else:
                    for fr in frames_com:
                        render_resultado_visual(
                            nome=f"{item['nome']} — segundo {fr['segundo']}",
                            imagem=fr['imagem'], cls_result=fr['cls'],
                            confiante=fr['confiante'], yolo_dets=fr['yolo_dets'],
                            limiar_cls=limiar_cls, limiar_det=limiar_det,
                            ood_threshold=ood_threshold,
                            subtitulo=f'Frame {fr["segundo"]}s',
                        )

                with st.expander("Mostrar todos os frames"):
                    for fr in item['frames']:
                        render_resultado_visual(
                            nome=f"{item['nome']} — segundo {fr['segundo']}",
                            imagem=fr['imagem'], cls_result=fr['cls'],
                            confiante=fr['confiante'], yolo_dets=fr['yolo_dets'],
                            limiar_cls=limiar_cls, limiar_det=limiar_det,
                            ood_threshold=ood_threshold,
                            subtitulo=f'Frame {fr["segundo"]}s',
                        )

        st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
        if st.button("Limpar e enviar outros arquivos", key="btn_limpar_bottom"):
            st.session_state.resultado    = None
            st.session_state.arquivo_nome = None
            st.session_state.uploader_key += 1
            st.rerun()

# ── Aba Dataset ───────────────────────────────────────────────────────────────
with aba_info:
    st.markdown("## GC10-DET — Metallic Surface Defect Detection")

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("EfficientNetV2 (OOD)", "94.54%", help="Acurácia Fase 2 — 11 classes")
    with col_m2:
        st.metric("YOLOv8 mAP@0.5", "0.645", help="Fase 3 — 10 classes de defeito")
    with col_m3:
        ood_thr = det_meta.get('ood_threshold', 0.40)
        st.metric("Limiar OOD padrão", f"{ood_thr*100:.0f}%", help="Calibrado na Fase 2")

    st.write("""
O GC10-DET é um dataset industrial real coletado por câmeras CCD lineares ao longo de
linhas de produção de chapas de aço laminadas.

**Pipeline deste app:**
- **Fase 1** — EfficientNetV2-S treinado com 10 classes de defeito — acurácia 83%
- **Fase 2** — Fine-tuning com 11ª classe `negative` para rejeitar imagens fora do domínio — acurácia 94.54%
- **Fase 3** — YOLOv8l treinado no GC10-DET para localizar defeitos com bounding box — mAP@0.5 0.645
""")

    st.markdown('<div class="section-title">Classes de defeitos</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    for i, cls in enumerate(DEFECT_CLASSES):
        with (c1 if i % 2 == 0 else c2):
            st.markdown(
                f'<div class="card-soft" style="margin-bottom:0.85rem;">'
                f'<div style="font-weight:700;font-size:1.05rem;">'
                f'{CLASS_PT.get(cls, cls)} <code>{cls}</code></div>'
                f'<div style="margin-top:0.45rem;color:#5e6d7c;">{CLASS_DESC.get(cls, "")}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">Distribuição de imagens por classe</div>', unsafe_allow_html=True)
    dist_data = {
        'silk_spot': 650, 'water_spot': 289, 'welding_line': 273,
        'crescent_gap': 226, 'punching_hole': 219, 'inclusion': 216,
        'oil_spot': 204, 'waist_folding': 146, 'crease': 52, 'rolled_pit': 31,
    }
    df_dist = pd.DataFrame([
        {'Classe': k, 'Português': CLASS_PT.get(k, k), 'Imagens': v}
        for k, v in dist_data.items()
    ])
    st.dataframe(df_dist, use_container_width=True, hide_index=True)
    st.caption("Dataset desbalanceado: silk_spot tem 21× mais imagens que rolled_pit.")

# ── Aba Exemplos ──────────────────────────────────────────────────────────────
with aba_ex_imagens:
    st.markdown("### Imagens de exemplo")
    st.caption(
        "**ex1.jpg** — Linha de Solda (welding_line)  |  "
        "**ex2.jpg** — Fora do domínio (elefantes — deve ser rejeitado)  |  "
        "**ex3.jpg** — Puncionamento (punching_hole)"
    )

    exemplos = [
        ("Linha de Solda",        BASE_DIR / "ex1.jpg"),
        ("Fora do domínio", BASE_DIR / "ex2.jpg"),
        ("Puncionamento",         BASE_DIR / "ex3.jpg"),
    ]
    cols = st.columns(3)
    for i, (label, path) in enumerate(exemplos):
        with cols[i]:
            st.markdown(f"**{label}**")
            if path.exists():
                st.image(Image.open(path), use_container_width=True)
                with open(path, 'rb') as f:
                    st.download_button(
                        "⬇ Baixar", f, file_name=path.name,
                        mime="image/jpeg", key=f"dl_img_{i}")
            else:
                st.warning(f"{path.name} não encontrada.")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("### Vídeos de exemplo")
    st.caption(
        "Vídeos gerados a partir das imagens do GC10-DET. "
        "Faça download e envie na aba **Análise** para testar a detecção em vídeo."
    )

    exemplos_video = [
        ("Fora do domínio",  BASE_DIR / "ex_video1.mp4"),
        ("Linha de Solda",   BASE_DIR / "ex_video2.mp4"),
        ("Dobra",   BASE_DIR / "ex_video3.mp4")
    ]
    cols_v = st.columns(4)
    for i, (label, path) in enumerate(exemplos_video):
        with cols_v[i]:
            st.markdown(f"**{label}**")
            if path.exists():
                with open(path, 'rb') as f:
                    vb = f.read()
                st.video(vb)
                st.download_button(
                    "⬇ Baixar", vb, file_name=path.name,
                    mime="video/mp4", key=f"dl_vid_{i}")
            else:
                st.warning(f"{path.name} não encontrada.")