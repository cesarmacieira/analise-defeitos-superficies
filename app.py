"""
Detector de Defeitos em Superfície de Aço — GC10-DET
Fluxo:
1) Classificação com EfficientNetV2
2) Verificação OOD
3) Detecção com YOLOv8 somente quando permitido
"""

import json
import tempfile
from collections import Counter
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont


# ─────────────────────────────────────────────────────────────────────────────
# Configuração da página
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Detector GC10-DET — Defeitos em Aço",
    layout="wide",
)

st.markdown("""
<style>
.block-container {
    padding-top: 1.3rem;
    padding-bottom: 2rem;
    max-width: 1320px;
}

.cabecalho-app {
    background: linear-gradient(135deg, #0d1b2a 0%, #1b2838 50%, #1a3a5c 100%);
    border-radius: 20px;
    padding: 2.2rem 2.4rem;
    margin-bottom: 1.6rem;
    color: white;
}

.cabecalho-app h1 {
    margin: 0;
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.5px;
}

.cabecalho-app p {
    margin: 0.45rem 0 0 0;
    font-size: 1rem;
    opacity: 0.88;
}

.card-info {
    background: #eef4fb;
    border: 1.5px solid #cfd9e6;
    border-radius: 16px;
    padding: 1.2rem 1.25rem;
}

.card-warn {
    background: #fff8f0;
    border: 1.5px solid #e9b16a;
    border-radius: 16px;
    padding: 1.2rem 1.25rem;
}

.card-error {
    background: #fff3f3;
    border: 1.5px solid #e59a9a;
    border-radius: 16px;
    padding: 1.2rem 1.25rem;
}

.card-soft {
    background: #ffffff;
    border: 1.5px solid #d7e0ea;
    border-radius: 16px;
    padding: 1rem 1.1rem;
}

.bar-bg {
    background: #dfe6ee;
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
    margin-top: 6px;
}

.section-title {
    font-size: 1rem;
    font-weight: 700;
    color: #223248;
    margin-bottom: 0.55rem;
}

.small-muted {
    color: #66788a;
    font-size: 0.9rem;
}

.stButton > button {
    background: linear-gradient(135deg, #1a3a5c, #2d6a9f);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    font-weight: 700;
    width: 100%;
}

.stButton > button:hover {
    color: white;
    opacity: 0.92;
}

.result-caption {
    text-align: center;
    color: #6f7e8b;
    font-size: 0.88rem;
    margin-top: 0.25rem;
}

code {
    color: #1d8a43 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="cabecalho-app">
    <h1>Detector de Defeitos — GC10-DET</h1>
    <p>Classificação, verificação OOD e localização de defeitos superficiais em chapas de aço</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

CLASS_NAMES = [
    'punching_hole', 'welding_line', 'crescent_gap', 'water_spot', 'oil_spot',
    'silk_spot', 'inclusion', 'rolled_pit', 'crease', 'waist_folding',
    'negative',
]
OOD_CLASS_IDX = 10

CLASS_PT = {
    'punching_hole': 'Puncionamento',
    'welding_line': 'Linha de Solda',
    'crescent_gap': 'Fresta Crescente',
    'water_spot': "Mancha d'Água",
    'oil_spot': 'Mancha de Óleo',
    'silk_spot': 'Mancha Seda',
    'inclusion': 'Inclusão',
    'rolled_pit': 'Cavidade Laminada',
    'crease': 'Dobra',
    'waist_folding': 'Dobra de Cintura',
    'negative': 'Fora do Domínio',
}

CLASS_DESC = {
    'punching_hole': 'Furo indesejado causado por falha mecânica na linha de produção.',
    'welding_line': 'Marca da emenda entre dois rolos de aço durante troca de bobina.',
    'crescent_gap': 'Lacuna em forma de meia-lua na superfície da chapa.',
    'water_spot': 'Mancha resultante de contato com umidade durante o processo.',
    'oil_spot': 'Contaminação por óleo lubrificante na superfície do aço.',
    'silk_spot': 'Mancha difusa e suave; o tipo mais frequente no dataset.',
    'inclusion': 'Partícula estranha embutida na superfície da chapa.',
    'rolled_pit': 'Pequena cavidade formada durante o processo de laminação.',
    'crease': 'Dobra localizada por deformação excessiva.',
    'waist_folding': 'Dobra de cintura, deformação mais pronunciada e severa.',
    'negative': 'Imagem fora do domínio industrial esperado para o modelo.',
}

CLASS_COLORS = {
    'punching_hole': (231, 76, 60),
    'welding_line': (52, 152, 219),
    'crescent_gap': (241, 196, 15),
    'water_spot': (26, 188, 156),
    'oil_spot': (160, 100, 45),
    'silk_spot': (46, 204, 113),
    'inclusion': (155, 89, 182),
    'rolled_pit': (50, 50, 50),
    'crease': (230, 126, 34),
    'waist_folding': (192, 57, 43),
    'negative': (210, 80, 80),
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _find_model(*candidates):
    for c in candidates:
        p = BASE_DIR / c
        if p.exists():
            return p
    return None


YOLO_PATH = _find_model('yolo_detector.pt')
CLS_PATH = _find_model('gc10det_cls_ood.pt', 'gc10det_cls_best.pt')
META_PATH = _find_model('detection_metadata.json')


# ─────────────────────────────────────────────────────────────────────────────
# Carregamento dos modelos
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_yolo():
    st.write("BASE_DIR:", BASE_DIR)
    st.write("YOLO_PATH:", YOLO_PATH)
    st.write("Arquivos na pasta:", [p.name for p in BASE_DIR.iterdir()])

    if YOLO_PATH is None:
        return None, "yolo_detector.pt não encontrado"

    try:
        from ultralytics import YOLO
        model = YOLO(str(YOLO_PATH))
        st.success("YOLO carregado com sucesso")
        return model, None
    except Exception as e:
        st.exception(e)
        return None, str(e)


@st.cache_resource
def load_cls():
    if CLS_PATH is None:
        return None, 'gc10det_cls_ood.pt não encontrado'

    try:
        import torch.nn as nn
        from torchvision.models import efficientnet_v2_s

        class GC10MultiOutputNet(nn.Module):
            def __init__(self, num_classes=11):
                super().__init__()
                backbone = efficientnet_v2_s(weights=None)
                in_feat = backbone.classifier[1].in_features
                backbone.classifier = nn.Identity()
                self.backbone = backbone

                self.cls_head = nn.Sequential(
                    nn.Linear(in_feat, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, num_classes),
                )

                self.bbox_head = nn.Sequential(
                    nn.Linear(in_feat, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Linear(512, 4),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                feat = self.backbone(x)
                return self.cls_head(feat), self.bbox_head(feat)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GC10MultiOutputNet(num_classes=len(CLASS_NAMES))

        try:
            state_dict = torch.load(str(CLS_PATH), map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load(str(CLS_PATH), map_location=device)

        model.load_state_dict(state_dict)
        model.to(device).eval()
        return model, None

    except Exception as e:
        return None, str(e)


@st.cache_data
def load_meta():
    if META_PATH and META_PATH.exists():
        try:
            return json.loads(META_PATH.read_text(encoding='utf-8'))
        except Exception:
            return {}
    return {}


yolo_model, yolo_err = load_yolo()
cls_model, cls_err = load_cls()
det_meta = load_meta()

YOLO_OK = yolo_model is not None
CLS_OK = cls_model is not None


# ─────────────────────────────────────────────────────────────────────────────
# Funções auxiliares
# ─────────────────────────────────────────────────────────────────────────────
def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*rgb)


def render_bar(score, color_hex):
    width = max(0, min(100, round(score * 100)))
    st.markdown(
        f"""
        <div class="bar-bg">
            <div style="width:{width}%;background:{color_hex};height:10px;border-radius:999px;"></div>
        </div>
        """,
        unsafe_allow_html=True
    )


def classify_image(pil_img):
    from torchvision import transforms

    meta = load_meta()
    ood_threshold = meta.get("ood_threshold", 0.40)

    device = next(cls_model.parameters()).device
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    img_rgb = pil_img.convert('RGB')
    w, h = img_rgb.size
    tensor = transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        cls_out, bbox_out = cls_model(tensor)

    probs = torch.softmax(cls_out, dim=1)[0].cpu().numpy()
    bbox_n = bbox_out[0].cpu().numpy()

    x1 = int(bbox_n[0] * w)
    y1 = int(bbox_n[1] * h)
    x2 = int(bbox_n[2] * w)
    y2 = int(bbox_n[3] * h)

    ood_score = float(probs[OOD_CLASS_IDX])
    is_ood = ood_score >= ood_threshold

    defect_probs = probs[:OOD_CLASS_IDX]
    top_idx = int(defect_probs.argmax())
    top_label = CLASS_NAMES[top_idx]
    top_score = float(defect_probs[top_idx])

    all_probs = sorted(
        [{'label': CLASS_NAMES[i], 'score': float(p)} for i, p in enumerate(probs)],
        key=lambda x: -x['score']
    )

    return {
        'top_label': top_label,
        'top_score': top_score,
        'all_probs': all_probs,
        'cls_bbox': [x1, y1, x2, y2],
        'ood_score': ood_score,
        'is_ood': is_ood,
    }


def detect_with_yolo(pil_img, thresh=0.25):
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        pil_img.convert('RGB').save(tmp.name, 'JPEG')
        tmp_path = tmp.name

    results = yolo_model.predict(tmp_path, conf=thresh, imgsz=640, verbose=False)
    Path(tmp_path).unlink(missing_ok=True)

    detections = []
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls_id = int(box.cls[0].item())
            score = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()

            if cls_id >= OOD_CLASS_IDX:
                continue

            label = CLASS_NAMES[cls_id]
            detections.append({
                'label': label,
                'label_pt': CLASS_PT.get(label, label),
                'score': score,
                'bbox': xyxy,
                'source': 'YOLOv8',
            })

    return detections


def draw_detections(pil_img, detections, line_width=3):
    img = pil_img.convert('RGB').copy()
    draw = ImageDraw.Draw(img)
    w, _ = img.size

    try:
        font = ImageFont.truetype(
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
            max(11, w // 24)
        )
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
        label = det['label']
        color = CLASS_COLORS.get(label, (255, 200, 0))
        text = f"{det['label_pt']} {det['score'] * 100:.1f}%"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        try:
            bb = draw.textbbox((x1, y1), text, font=font)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
        except AttributeError:
            tw, th = draw.textsize(text, font=font)

        ty = max(0, y1 - th - 4)
        draw.rectangle([x1, ty, x1 + tw + 8, ty + th + 4], fill=color)
        draw.text((x1 + 4, ty + 2), text, fill=(255, 255, 255), font=font)

    return img


def zoom_defect(pil_img, bbox, pad_pct=0.3):
    w, h = pil_img.size
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw, bh = x2 - x1, y2 - y1
    pad_x = max(int(bw * pad_pct), 8)
    pad_y = max(int(bh * pad_pct), 8)

    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + pad_y)

    return pil_img.crop((cx1, cy1, cx2, cy2))


# ─────────────────────────────────────────────────────────────────────────────
# Estado
# ─────────────────────────────────────────────────────────────────────────────
for key, default in [
    ('resultado', None),
    ('arquivo_nome', None),
    ('uploader_key', 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────────────────────────────────────
# Abas
# ─────────────────────────────────────────────────────────────────────────────
aba_analise, aba_info = st.tabs(['Análise', 'Sobre o Dataset'])


# ─────────────────────────────────────────────────────────────────────────────
# Aba análise
# ─────────────────────────────────────────────────────────────────────────────
with aba_analise:
    if not CLS_OK:
        st.error(
            f"Modelo de classificação não encontrado. "
            f"{'Erro: ' + cls_err if cls_err else 'Coloque gc10det_cls_ood.pt na mesma pasta que app.py.'}"
        )
        st.stop()

    st.markdown("### Configurações")

    cfg1, cfg2 = st.columns(2)
    with cfg1:
        limiar_cls_pct = st.number_input(
            "Confiança mínima para classificação (%)",
            min_value=1,
            max_value=99,
            value=40,
            step=1,
        )
    with cfg2:
        limiar_det_pct = st.number_input(
            "Confiança mínima para detecção YOLOv8 (%)",
            min_value=1,
            max_value=99,
            value=25,
            step=1,
        )

    limiar_cls = limiar_cls_pct / 100.0
    limiar_det = limiar_det_pct / 100.0

    upload_col, preview_col = st.columns([1.5, 1])

    with upload_col:
        arquivo = st.file_uploader(
            "Envie uma imagem para análise",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            key=f'upload_{st.session_state.uploader_key}',
        )

    imagem = None
    if arquivo is not None:
        imagem = Image.open(arquivo)
        if imagem.mode != 'RGB':
            imagem = imagem.convert('RGB')

    if arquivo is None:
        st.session_state.resultado = None
        st.session_state.arquivo_nome = None
    elif arquivo.name != st.session_state.arquivo_nome:
        st.session_state.resultado = None
        st.session_state.arquivo_nome = arquivo.name

    with preview_col:
        if imagem is not None:
            st.image(imagem, use_container_width=True)

    if imagem is not None and st.session_state.resultado is None:
        b1, b2, b3 = st.columns([1, 1, 5])
        with b1:
            analisar = st.button("Analisar imagem", key="btn_analisar")
        with b2:
            limpar = st.button("Limpar imagem", key="btn_limpar_top")

        if limpar:
            st.session_state.resultado = None
            st.session_state.arquivo_nome = None
            st.session_state.uploader_key += 1
            st.rerun()

        if analisar:
            with st.spinner("Executando análise..."):
                cls_result = classify_image(imagem)
                confiante = (cls_result['top_score'] >= limiar_cls) and (not cls_result.get('is_ood', False))

                yolo_dets = []
                if confiante and YOLO_OK:
                    yolo_dets = detect_with_yolo(imagem, limiar_det)

                st.session_state.resultado = {
                    'cls': cls_result,
                    'confiante': confiante,
                    'yolo_dets': yolo_dets,
                }
            st.rerun()

    if imagem is not None and st.session_state.resultado is not None:
        res = st.session_state.resultado
        cls_result = res['cls']
        confiante = res['confiante']
        yolo_dets = res['yolo_dets']

        top_label = cls_result['top_label']
        top_score = cls_result['top_score']
        top_pt = CLASS_PT.get(top_label, top_label)
        top_hex = rgb_to_hex(CLASS_COLORS.get(top_label, (100, 100, 100)))

        st.markdown("---")
        col_img, col_cls, col_det = st.columns(3, gap="large")

        # ── Imagem anotada ─────────────────────────────────────────────────────
        with col_img:
            st.markdown('<div class="section-title">Imagem analisada</div>', unsafe_allow_html=True)
            if confiante and YOLO_OK and len(yolo_dets) > 0:
                img_annotated = draw_detections(imagem, yolo_dets)
                st.image(img_annotated, use_container_width=True)
                st.markdown('<div class="result-caption">Com marcações YOLOv8</div>', unsafe_allow_html=True)
            else:
                st.image(imagem, use_container_width=True)
                st.markdown('<div class="result-caption">Imagem enviada</div>', unsafe_allow_html=True)

        # ── Classificação ──────────────────────────────────────────────────────
        with col_cls:
            st.markdown('<div class="section-title">Classificação</div>', unsafe_allow_html=True)

            if cls_result.get('is_ood'):
                st.markdown(
                    f'''<div class="card-error">
                        <div style="font-size:1.05rem;font-weight:700;color:#b23b3b;">Fora do domínio industrial</div>
                        <div style="margin-top:0.5rem;font-size:2rem;font-weight:800;">—</div>
                        <div style="margin-top:0.4rem;color:#5f6972;font-size:0.9rem;">Score OOD: <strong>{cls_result["ood_score"]*100:.1f}%</strong></div>
                    </div>''',
                    unsafe_allow_html=True
                )
            elif confiante:
                st.markdown(
                    f'''<div class="card-info">
                        <div class="small-muted">Defeito detectado</div>
                        <div style="font-size:1.9rem;font-weight:800;color:{top_hex};margin-top:0.2rem;">{top_pt}</div>
                        <div style="margin-top:0.5rem;">''',
                    unsafe_allow_html=True
                )
                render_bar(top_score, top_hex)
                st.markdown(
                    f'<div style="margin-top:0.4rem;font-size:1.3rem;font-weight:700;">{top_score*100:.1f}%</div></div></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'''<div class="card-warn">
                        <div style="font-size:1.05rem;font-weight:700;color:#c37514;">Classificação inconclusiva</div>
                        <div style="font-size:1.9rem;font-weight:800;color:{top_hex};margin-top:0.2rem;">{top_pt}</div>
                        <div style="margin-top:0.5rem;">''',
                    unsafe_allow_html=True
                )
                render_bar(top_score, top_hex)
                st.markdown(
                    f'<div style="margin-top:0.4rem;font-size:1.3rem;font-weight:700;">{top_score*100:.1f}%</div>'
                    f'<div style="margin-top:0.3rem;color:#5f6972;font-size:0.88rem;">Abaixo do limiar de {limiar_cls*100:.0f}%</div></div></div>',
                    unsafe_allow_html=True
                )

        # ── Detecção YOLOv8 ────────────────────────────────────────────────────
        with col_det:
            st.markdown('<div class="section-title">Detecção YOLOv8</div>', unsafe_allow_html=True)

            if cls_result.get('is_ood'):
                st.markdown(
                    '<div class="card-error"><div style="color:#b23b3b;font-weight:700;">Bloqueada</div>'
                    '<div style="margin-top:0.4rem;color:#5f6972;font-size:0.9rem;">Imagem fora do domínio.</div></div>',
                    unsafe_allow_html=True
                )
            elif not confiante:
                st.markdown(
                    '<div class="card-warn"><div style="color:#c37514;font-weight:700;">Não executada</div>'
                    '<div style="margin-top:0.4rem;color:#5f6972;font-size:0.9rem;">Classificação abaixo do limiar.</div></div>',
                    unsafe_allow_html=True
                )
            elif not YOLO_OK:
                st.markdown(
                    '<div class="card-error"><div style="color:#b23b3b;font-weight:700;">Modelo não disponível</div></div>',
                    unsafe_allow_html=True
                )
            elif len(yolo_dets) == 0:
                st.markdown(
                    '<div class="card-soft"><div style="font-weight:700;">Nenhuma detecção</div>'
                    f'<div style="margin-top:0.4rem;color:#5f6972;font-size:0.9rem;">Nenhuma bounding box acima de {limiar_det*100:.0f}%.</div></div>',
                    unsafe_allow_html=True
                )
            else:
                best_det = max(yolo_dets, key=lambda d: d['score'])
                best_pt = CLASS_PT.get(best_det['label'], best_det['label'])
                best_hex = rgb_to_hex(CLASS_COLORS.get(best_det['label'], (100, 100, 100)))
                st.markdown(
                    f'''<div class="card-info">
                        <div class="small-muted">{len(yolo_dets)} bounding box(es) encontrada(s)</div>
                        <div style="font-size:1.9rem;font-weight:800;color:{best_hex};margin-top:0.2rem;">{best_pt}</div>
                        <div style="margin-top:0.5rem;">''',
                    unsafe_allow_html=True
                )
                render_bar(best_det['score'], best_hex)
                st.markdown(
                    f'<div style="margin-top:0.4rem;font-size:1.3rem;font-weight:700;">{best_det["score"]*100:.1f}%</div></div></div>',
                    unsafe_allow_html=True
                )

        st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
        if st.button("Limpar e enviar outra imagem", key="btn_limpar_bottom"):
            st.session_state.resultado = None
            st.session_state.arquivo_nome = None
            st.session_state.uploader_key += 1
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Aba dataset
# ─────────────────────────────────────────────────────────────────────────────
with aba_info:
    st.markdown("### GC10-DET — Metallic Surface Defect Detection")
    st.write("""
O GC10-DET é um dataset industrial real coletado por câmeras CCD lineares ao longo de linhas de produção de chapas de aço laminadas.

Esta versão do app usa:
- 10 classes industriais de defeitos
- 1 classe auxiliar `negative` para detectar imagens fora do domínio industrial
""")

    st.markdown("### Classes de defeitos")

    dataset_classes = CLASS_NAMES[:OOD_CLASS_IDX]
    c1, c2 = st.columns(2)

    for i, cls in enumerate(dataset_classes):
        target = c1 if i % 2 == 0 else c2
        with target:
            st.markdown(f"""
            <div class="card-soft" style="margin-bottom:0.85rem;">
                <div style="font-weight:700;font-size:1.05rem;">{CLASS_PT.get(cls, cls)} <code>{cls}</code></div>
                <div style="margin-top:0.45rem;color:#5e6d7c;">{CLASS_DESC.get(cls, '')}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### Distribuição aproximada de imagens")

    import pandas as pd

    dist_data = {
        'silk_spot': 650,
        'water_spot': 289,
        'welding_line': 273,
        'crescent_gap': 226,
        'punching_hole': 219,
        'inclusion': 216,
        'oil_spot': 204,
        'waist_folding': 146,
        'crease': 52,
        'rolled_pit': 31,
    }

    df_dist = pd.DataFrame([
        {'Classe': k, 'Português': CLASS_PT.get(k, k), 'Imagens': v}
        for k, v in dist_data.items()
    ])
    st.dataframe(df_dist, use_container_width=True, hide_index=True)
    st.caption("Dataset desbalanceado: silk_spot tem muito mais imagens que rolled_pit.")