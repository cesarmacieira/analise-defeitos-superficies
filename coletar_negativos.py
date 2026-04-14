"""
coletar_negativos.py
─────────────────────────────────────────────────────────────────────────────
Coleta imagens "negativas" (fora do domínio / OOD) para treinar o modelo
GC10-DET a rejeitar imagens que não são chapas de aço.

O bug reportado: imagens de dashboards, documentos e fotografias comuns
estão sendo classificadas como defeitos com 100% de confiança.
Este script resolve isso criando um banco de imagens negativas diversas.

FONTES UTILIZADAS (sem autenticação):
  1. COCO val2017   — fotos naturais variadas (via API pública)
  2. Open Images v7 — URLs públicas de imagens rotuladas
  3. Screenshots sintéticos — dashboards, gráficos, UIs geradas via PIL
  4. Ruído puro     — patches de textura aleatória

USO:
  python coletar_negativos.py [--n 500] [--dest images/negative] [--sources all]

ARGUMENTOS:
  --n       Número total de imagens negativas desejadas (padrão: 600)
  --dest    Pasta de destino (padrão: images/negative)
  --sources Fontes: all | coco | openimages | synthetic | noise
              (padrão: all)

REQUISITOS:
  pip install requests pillow tqdm numpy
"""

import argparse
import json
import os
import random
import sys
import time
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Configuração
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "_ood_cache"
CACHE_DIR.mkdir(exist_ok=True)

# Categorias COCO que podem lembrar superfícies metálicas — evitar
COCO_SKIP_CATS = {
    "knife", "scissors", "fork", "spoon", "bottle", "wire",
    "refrigerator", "oven", "microwave", "toaster",
}

# Categorias Open Images que queremos (claramente não-aço)
OPEN_IMAGES_LABELS = [
    "Flower", "Tree", "Dog", "Cat", "Bird", "Food", "Furniture",
    "Building", "Vehicle", "Book", "Computer keyboard", "Mobile phone",
    "Person", "Face", "Hat", "Clothing", "Sports equipment",
    "Musical instrument", "Toy",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. COCO val2017
# ─────────────────────────────────────────────────────────────────────────────
def collect_coco(dest_dir: Path, n: int = 200) -> int:
    """Baixa n imagens do COCO val2017 que não pertencem a categorias industriais."""
    ann_json = CACHE_DIR / "annotations" / "instances_val2017.json"

    if not ann_json.exists():
        zip_path = CACHE_DIR / "annotations.zip"
        url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        print(f"[COCO] Baixando anotações (~240 MB)...")
        try:
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(CACHE_DIR)
            zip_path.unlink(missing_ok=True)
        except Exception as e:
            print(f"[COCO] Erro no download das anotações: {e}")
            return 0

    with open(ann_json, encoding="utf-8") as f:
        coco = json.load(f)

    skip_ids = {c["id"] for c in coco["categories"] if c["name"] in COCO_SKIP_CATS}

    img_cats: dict[int, set] = {}
    for ann in coco["annotations"]:
        img_cats.setdefault(ann["image_id"], set()).add(ann["category_id"])

    valid_imgs = [
        img for img in coco["images"]
        if img["id"] in img_cats and not (img_cats[img["id"]] & skip_ids)
    ]
    random.shuffle(valid_imgs)

    downloaded = 0
    errors = 0
    print(f"[COCO] {len(valid_imgs)} candidatas. Baixando {n}...")

    for img_info in tqdm(valid_imgs, desc="COCO", unit="img"):
        if downloaded >= n:
            break
        out_path = dest_dir / f"coco_{img_info['id']}.jpg"
        if out_path.exists():
            downloaded += 1
            continue
        try:
            r = requests.get(img_info["coco_url"], timeout=15)
            if r.status_code == 200:
                # Converte para RGB e redimensiona
                img = Image.open(BytesIO(r.content)).convert("RGB")
                img = img.resize((640, 480), Image.LANCZOS)
                img.save(out_path, "JPEG", quality=85)
                downloaded += 1
            else:
                errors += 1
        except Exception:
            errors += 1
        if errors > 30:
            print(f"[COCO] Muitos erros ({errors}). Parando com {downloaded} imagens.")
            break

    print(f"[COCO] ✅ {downloaded} imagens coletadas ({errors} erros)")
    return downloaded


# ─────────────────────────────────────────────────────────────────────────────
# 2. Open Images v7 (subset público de URLs)
# ─────────────────────────────────────────────────────────────────────────────
def collect_open_images(dest_dir: Path, n: int = 150) -> int:
    """
    Baixa imagens do Open Images v7 via CSV público de URLs.
    Usa o arquivo de URLs de validação (sem login).
    """
    urls_csv = CACHE_DIR / "open_images_val_urls.txt"
    
    # Tenta baixar lista de URLs públicas do Open Images
    if not urls_csv.exists():
        csv_url = (
            "https://storage.googleapis.com/openimages/2018_04/"
            "validation/validation-images-boxable.csv"
        )
        print("[OpenImages] Baixando lista de URLs...")
        try:
            r = requests.get(csv_url, timeout=30, stream=True)
            if r.status_code == 200:
                with open(urls_csv, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                print(f"[OpenImages] HTTP {r.status_code}. Pulando fonte.")
                return 0
        except Exception as e:
            print(f"[OpenImages] Erro: {e}. Pulando fonte.")
            return 0

    # Lê URLs e filtra
    try:
        import csv
        image_urls = []
        with open(urls_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                url = row.get("OriginalURL", "")
                if url.startswith("http") and url.endswith((".jpg", ".jpeg", ".png")):
                    image_urls.append(url)
    except Exception as e:
        print(f"[OpenImages] Erro ao ler CSV: {e}")
        return 0

    random.shuffle(image_urls)
    downloaded = 0
    errors = 0
    print(f"[OpenImages] {len(image_urls)} URLs. Baixando {n}...")

    for i, url in enumerate(tqdm(image_urls[:n * 4], desc="OpenImages", unit="img")):
        if downloaded >= n:
            break
        img_id = url.split("/")[-1].split(".")[0]
        out_path = dest_dir / f"oi_{img_id}.jpg"
        if out_path.exists():
            downloaded += 1
            continue
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200 and len(r.content) > 5000:
                img = Image.open(BytesIO(r.content)).convert("RGB")
                # Filtra imagens muito escuras ou muito claras (podem ser artefatos)
                arr = np.array(img.resize((64, 64)))
                mean_brightness = arr.mean()
                if 20 < mean_brightness < 240:
                    img = img.resize((640, 480), Image.LANCZOS)
                    img.save(out_path, "JPEG", quality=85)
                    downloaded += 1
            else:
                errors += 1
        except Exception:
            errors += 1

    print(f"[OpenImages] ✅ {downloaded} imagens coletadas ({errors} erros)")
    return downloaded


# ─────────────────────────────────────────────────────────────────────────────
# 3. Imagens sintéticas — Dashboards, gráficos, UIs, texto
# ─────────────────────────────────────────────────────────────────────────────
def generate_synthetic(dest_dir: Path, n: int = 150) -> int:
    """
    Gera imagens sintéticas que imitam o tipo de conteúdo problemático:
    - Dashboards com gráficos de barra e pizza
    - Telas com texto e tabelas
    - Gradientes coloridos
    - Padrões geométricos
    - Imagens de "natureza" sintética (céus, grama)

    Essas imagens são fundamentais porque o bug reportado foi exatamente
    um dashboard sendo classificado como waist_folding.
    """
    generated = 0
    rng = np.random.default_rng(SEED)

    generators = [
        _gen_dashboard,
        _gen_bar_chart,
        _gen_pie_chart,
        _gen_text_document,
        _gen_geometric_pattern,
        _gen_color_gradient,
        _gen_noisy_nature,
        _gen_grid_table,
        _gen_ui_mockup,
        _gen_circular_gauges,
    ]

    n_per_gen = max(1, n // len(generators))
    extra     = n - n_per_gen * len(generators)

    print(f"[Sintético] Gerando {n} imagens sintéticas...")
    with tqdm(total=n, desc="Sintético", unit="img") as pbar:
        for i, gen_fn in enumerate(generators):
            count = n_per_gen + (1 if i < extra else 0)
            for j in range(count):
                try:
                    img = gen_fn(rng, j)
                    out = dest_dir / f"synth_{gen_fn.__name__}_{j:04d}.jpg"
                    img.convert("RGB").save(out, "JPEG", quality=85)
                    generated += 1
                    pbar.update(1)
                except Exception as e:
                    pass  # ignora falhas individuais

    print(f"[Sintético] ✅ {generated} imagens geradas")
    return generated


def _rand_color(rng, alpha=255) -> tuple:
    return tuple(rng.integers(0, 256, size=3).tolist())


def _gen_dashboard(rng, seed: int) -> Image.Image:
    """Dashboard com múltiplos painéis e gráficos."""
    w, h = 800, 600
    img  = Image.new("RGB", (w, h), color=_rand_color(rng))
    draw = ImageDraw.Draw(img)

    # Fundo cinza/azul escuro (típico de dashboards)
    bg = (rng.integers(20, 60), rng.integers(30, 70), rng.integers(50, 100))
    img.paste(Image.new("RGB", (w, h), bg))
    draw = ImageDraw.Draw(img)

    # Painéis
    for _ in range(rng.integers(3, 7)):
        x1 = int(rng.integers(10, w // 2))
        y1 = int(rng.integers(10, h // 2))
        x2 = int(rng.integers(x1 + 80, min(x1 + 300, w - 10)))
        y2 = int(rng.integers(y1 + 60, min(y1 + 200, h - 10)))
        panel_color = (rng.integers(35, 65), rng.integers(45, 80), rng.integers(60, 110))
        draw.rectangle([x1, y1, x2, y2], fill=panel_color, outline=(100, 150, 200), width=1)

        # Número grande no painel (KPI)
        val = rng.integers(0, 100000)
        draw.text((x1 + 10, y1 + 15), f"{val:,}", fill=(255, 255, 255))
        draw.text((x1 + 10, y1 + 35), f"Métrica #{seed % 10}", fill=(150, 180, 210))

    # Barras horizontais (gráfico)
    bar_x = 10
    for i in range(rng.integers(4, 9)):
        bar_y  = 350 + i * 25
        bar_w  = int(rng.integers(50, 500))
        color  = _rand_color(rng)
        draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + 18], fill=color)

    return img


def _gen_bar_chart(rng, seed: int) -> Image.Image:
    """Gráfico de barras verticais colorido."""
    w, h = 640, 480
    img  = Image.new("RGB", (w, h), (245, 247, 250))
    draw = ImageDraw.Draw(img)

    # Eixos
    draw.line([(60, 420), (580, 420)], fill=(80, 80, 80), width=2)
    draw.line([(60, 50),  (60, 420)],  fill=(80, 80, 80), width=2)

    n_bars = int(rng.integers(4, 12))
    bar_w  = (520 - n_bars * 5) // n_bars
    colors = [_rand_color(rng) for _ in range(n_bars)]

    for i in range(n_bars):
        x1    = 65 + i * (bar_w + 5)
        bar_h = int(rng.integers(30, 340))
        y1    = 420 - bar_h
        draw.rectangle([x1, y1, x1 + bar_w, 420], fill=colors[i])
        # Valor no topo
        draw.text((x1, y1 - 15), str(bar_h), fill=(50, 50, 50))

    # Título
    draw.text((200, 15), f"Relatório #{seed % 100}", fill=(30, 30, 30))
    # Linhas de grade horizontais
    for y_grid in range(50, 420, 60):
        draw.line([(61, y_grid), (580, y_grid)], fill=(200, 200, 200), width=1)

    return img


def _gen_pie_chart(rng, seed: int) -> Image.Image:
    """Gráfico de pizza."""
    w, h = 500, 500
    img  = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    cx, cy, r = 250, 250, 180
    n_slices  = int(rng.integers(3, 8))
    values    = rng.random(n_slices)
    values   /= values.sum()

    import math
    start = -90.0
    for i, v in enumerate(values):
        angle = v * 360
        color = _rand_color(rng)
        draw.pieslice([cx - r, cy - r, cx + r, cy + r],
                      start=start, end=start + angle, fill=color, outline="white")
        # Label no meio da fatia
        mid_angle = math.radians(start + angle / 2)
        lx = int(cx + r * 0.65 * math.cos(mid_angle))
        ly = int(cy + r * 0.65 * math.sin(mid_angle))
        draw.text((lx - 15, ly - 8), f"{v*100:.0f}%", fill=(255, 255, 255))
        start += angle

    draw.text((150, 10), f"Distribuição #{seed}", fill=(30, 30, 30))
    return img


def _gen_text_document(rng, seed: int) -> Image.Image:
    """Documento com texto (como o dashboard da Justiça Federal)."""
    w, h = 800, 600
    img  = Image.new("RGB", (w, h), (250, 250, 250))
    draw = ImageDraw.Draw(img)

    # Cabeçalho
    header_color = _rand_color(rng)
    draw.rectangle([0, 0, w, 60], fill=header_color)
    draw.text((20, 18), "Painel Integrado — Estatísticas", fill=(255, 255, 255))

    # Linhas de texto (imitam dados tabulares)
    y = 80
    for i in range(rng.integers(10, 25)):
        line_color = (rng.integers(30, 80),) * 3
        line_len   = int(rng.integers(100, 750))
        draw.line([(20, y), (20 + line_len, y)], fill=line_color, width=2)
        # Às vezes adiciona um "número KPI"
        if rng.random() > 0.6:
            val = f"{rng.integers(100, 100000):,}"
            draw.text((line_len + 25, y - 8), val, fill=(50, 100, 180))
        y += int(rng.integers(18, 30))
        if y > h - 30:
            break

    # Círculos (imitam gauges/metas do dashboard)
    for _ in range(rng.integers(2, 6)):
        cx = int(rng.integers(100, 700))
        cy = int(rng.integers(300, 550))
        r  = int(rng.integers(30, 60))
        color = _rand_color(rng)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=4)
        pct = rng.integers(30, 120)
        draw.text((cx - 20, cy - 10), f"{pct}%", fill=color)

    return img


def _gen_geometric_pattern(rng, seed: int) -> Image.Image:
    """Padrão geométrico abstrato — claramente não é aço."""
    w, h = 512, 512
    img  = Image.new("RGB", (w, h), _rand_color(rng))
    draw = ImageDraw.Draw(img)

    pattern = int(rng.integers(0, 4))
    if pattern == 0:  # Quadrados concêntricos
        for r in range(10, 250, 20):
            color = _rand_color(rng)
            draw.rectangle([256 - r, 256 - r, 256 + r, 256 + r], outline=color, width=3)
    elif pattern == 1:  # Círculos
        for r in range(10, 240, 18):
            draw.ellipse([256 - r, 256 - r, 256 + r, 256 + r],
                         outline=_rand_color(rng), width=3)
    elif pattern == 2:  # Grade
        spacing = int(rng.integers(20, 60))
        color   = _rand_color(rng)
        for x in range(0, w, spacing):
            draw.line([(x, 0), (x, h)], fill=color, width=1)
        for y in range(0, h, spacing):
            draw.line([(0, y), (w, y)], fill=color, width=1)
    else:  # Triângulos aleatórios
        for _ in range(rng.integers(5, 20)):
            pts = [(int(rng.integers(0, w)), int(rng.integers(0, h))) for _ in range(3)]
            draw.polygon(pts, fill=_rand_color(rng), outline=_rand_color(rng))

    return img


def _gen_color_gradient(rng, seed: int) -> Image.Image:
    """Gradiente de cores (simulando fotos de céu, natureza, etc.)."""
    w, h = 640, 480
    c1   = np.array(_rand_color(rng), dtype=float)
    c2   = np.array(_rand_color(rng), dtype=float)
    c3   = np.array(_rand_color(rng), dtype=float)

    direction = int(rng.integers(0, 2))  # 0=horizontal, 1=vertical
    arr = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(w if direction == 0 else h):
        t   = i / (w if direction == 0 else h)
        c   = (c1 * (1 - t) + c2 * t + c3 * np.sin(t * np.pi) * 0.3).clip(0, 255).astype(np.uint8)
        if direction == 0:
            arr[:, i] = c
        else:
            arr[i, :] = c

    # Adiciona ruído leve para parecer mais natural
    noise = rng.integers(-15, 15, arr.shape, dtype=np.int16)
    arr   = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _gen_noisy_nature(rng, seed: int) -> Image.Image:
    """Imagem que imita textura de natureza (grama, terra, céu nublado)."""
    w, h = 512, 512
    texture_type = int(rng.integers(0, 3))

    if texture_type == 0:  # Grama — tons de verde com variação
        base = np.array([
            rng.integers(30, 80),
            rng.integers(100, 180),
            rng.integers(20, 60),
        ], dtype=float)
    elif texture_type == 1:  # Céu — azul com gradiente
        base = np.array([
            rng.integers(100, 180),
            rng.integers(150, 220),
            rng.integers(200, 255),
        ], dtype=float)
    else:  # Terra — marrom
        base = np.array([
            rng.integers(100, 160),
            rng.integers(70, 120),
            rng.integers(30, 80),
        ], dtype=float)

    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(3):
        noise = rng.normal(base[c], 25, (h, w))
        arr[:, :, c] = np.clip(noise, 0, 255).astype(np.uint8)

    # Adiciona "ondas" suaves
    x = np.linspace(0, 4 * np.pi, w)
    y = np.linspace(0, 4 * np.pi, h)
    xx, yy = np.meshgrid(x, y)
    wave   = (np.sin(xx + seed) * np.cos(yy) * 20).astype(np.int16)
    arr    = np.clip(arr.astype(np.int16) + wave[:, :, np.newaxis], 0, 255).astype(np.uint8)

    return Image.fromarray(arr)


def _gen_grid_table(rng, seed: int) -> Image.Image:
    """Tabela com linhas e colunas (planilha/relatório)."""
    w, h  = 800, 500
    img   = Image.new("RGB", (w, h), (255, 255, 255))
    draw  = ImageDraw.Draw(img)
    
    n_cols = int(rng.integers(3, 8))
    n_rows = int(rng.integers(6, 16))
    
    col_w = (w - 40) // n_cols
    row_h = (h - 80) // n_rows
    
    # Cabeçalho
    header_color = _rand_color(rng)
    draw.rectangle([20, 40, w - 20, 40 + row_h], fill=header_color)
    
    for c in range(n_cols):
        x = 20 + c * col_w
        draw.text((x + 5, 50), f"Col {c + 1}", fill=(255, 255, 255))
    
    # Linhas de dados
    for r in range(1, n_rows):
        y = 40 + r * row_h
        row_bg = (245, 248, 252) if r % 2 == 0 else (255, 255, 255)
        draw.rectangle([20, y, w - 20, y + row_h], fill=row_bg)
        draw.line([(20, y), (w - 20, y)], fill=(210, 215, 220), width=1)
        
        for c in range(n_cols):
            x = 20 + c * col_w
            val = rng.integers(0, 99999)
            draw.text((x + 5, y + 5), f"{val:,}", fill=(50, 50, 70))
    
    # Bordas verticais
    for c in range(n_cols + 1):
        x = 20 + c * col_w
        draw.line([(x, 40), (x, 40 + n_rows * row_h)], fill=(190, 200, 210), width=1)
    
    return img


def _gen_ui_mockup(rng, seed: int) -> Image.Image:
    """Mockup de interface de usuário (app, website)."""
    w, h = 800, 600
    
    # Fundo
    bg_color = (rng.integers(240, 255), rng.integers(240, 255), rng.integers(245, 255))
    img  = Image.new("RGB", (w, h), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Navbar
    nav_color = _rand_color(rng)
    draw.rectangle([0, 0, w, 55], fill=nav_color)
    draw.text((15, 18), "◉  App Title", fill=(255, 255, 255))
    
    # Sidebar
    sidebar_color = (rng.integers(220, 240), rng.integers(225, 245), rng.integers(230, 250))
    draw.rectangle([0, 55, 200, h], fill=sidebar_color)
    for i, label in enumerate(["Dashboard", "Relatórios", "Usuários", "Configurações", "Sair"]):
        y = 80 + i * 40
        active = (i == seed % 5)
        if active:
            draw.rectangle([0, y - 5, 200, y + 30], fill=nav_color)
        draw.text((15, y + 5), f"▸ {label}", fill=(255, 255, 255) if active else (60, 80, 100))
    
    # Cards na área principal
    for row in range(2):
        for col in range(2):
            cx1 = 220 + col * 280
            cy1 = 80 + row * 240
            draw.rectangle([cx1, cy1, cx1 + 250, cy1 + 210],
                           fill=(255, 255, 255), outline=(210, 220, 230), width=2)
            # Mini gráfico dentro do card
            for b in range(5):
                bx = cx1 + 20 + b * 40
                bh = int(rng.integers(30, 130))
                draw.rectangle([bx, cy1 + 190 - bh, bx + 28, cy1 + 190],
                               fill=_rand_color(rng))
            draw.text((cx1 + 10, cy1 + 10), f"KPI {row*2+col+1}", fill=(30, 50, 80))
            val = rng.integers(1000, 99999)
            draw.text((cx1 + 10, cy1 + 35), f"{val:,}", fill=(20, 40, 100))
    
    return img


def _gen_circular_gauges(rng, seed: int) -> Image.Image:
    """Gauges circulares (como os do dashboard da Justiça Federal)."""
    import math
    
    w, h = 700, 500
    img  = Image.new("RGB", (w, h), (245, 248, 252))
    draw = ImageDraw.Draw(img)
    
    # Título
    draw.text((20, 15), f"Painel de Metas — {2020 + seed % 5}", fill=(30, 50, 80))
    draw.line([(20, 40), (w - 20, 40)], fill=(180, 195, 210), width=2)
    
    positions = [
        (120, 160), (280, 160), (440, 160), (600, 160),
        (120, 360), (280, 360), (440, 360), (600, 360),
    ]
    
    for i, (cx, cy) in enumerate(positions):
        r = 60
        pct = float(rng.uniform(30, 130))
        
        # Arco de fundo (cinza)
        draw.arc([cx - r, cy - r, cx + r, cy + r],
                 start=-210, end=30, fill=(200, 210, 220), width=8)
        
        # Arco de progresso (colorido)
        color = _rand_color(rng)
        end_angle = -210 + min(pct / 100 * 240, 240)
        draw.arc([cx - r, cy - r, cx + r, cy + r],
                 start=-210, end=int(end_angle), fill=color, width=8)
        
        # Texto central
        draw.text((cx - 22, cy - 12), f"{pct:.0f}%", fill=(40, 60, 80))
        draw.text((cx - 20, cy + 10), f"Meta {i+1}", fill=(100, 120, 140))
    
    return img


# ─────────────────────────────────────────────────────────────────────────────
# 4. Ruído puro e texturas aleatórias
# ─────────────────────────────────────────────────────────────────────────────
def generate_noise(dest_dir: Path, n: int = 50) -> int:
    """
    Gera patches de textura sintética: ruído, gradientes, padrões de Perlin-like.
    Esses exemplos garantem que o modelo aprenda que "imagem sem estrutura de aço"
    deve ser rejeitada — mesmo com textura superficial similar.
    """
    generated = 0
    rng = np.random.default_rng(SEED + 9999)

    print(f"[Ruído] Gerando {n} imagens de ruído/textura...")
    for i in tqdm(range(n), desc="Ruído", unit="img"):
        noise_type = i % 5

        if noise_type == 0:  # Ruído gaussiano puro
            arr = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)

        elif noise_type == 1:  # Ruído colorido suave
            base = rng.integers(0, 256, 3).astype(float)
            arr  = rng.normal(base, 40, (480, 640, 3)).clip(0, 255).astype(np.uint8)

        elif noise_type == 2:  # Xadrez colorido
            block = rng.integers(5, 40)
            c1    = rng.integers(0, 256, 3).astype(np.uint8)
            c2    = rng.integers(0, 256, 3).astype(np.uint8)
            arr   = np.zeros((480, 640, 3), dtype=np.uint8)
            for y in range(480):
                for x in range(640):
                    arr[y, x] = c1 if ((x // block + y // block) % 2 == 0) else c2

        elif noise_type == 3:  # Stripes (listras horizontais/verticais)
            arr = np.zeros((480, 640, 3), dtype=np.uint8)
            n_stripes = rng.integers(5, 30)
            colors    = [rng.integers(0, 256, 3).astype(np.uint8) for _ in range(n_stripes)]
            stripe_h  = 480 // n_stripes
            for s, c in enumerate(colors):
                arr[s * stripe_h : (s + 1) * stripe_h, :] = c

        else:  # Plasma/Perlin-like
            x = np.linspace(0, rng.uniform(2, 8) * np.pi, 640)
            y = np.linspace(0, rng.uniform(2, 8) * np.pi, 480)
            xx, yy = np.meshgrid(x, y)
            plasma  = (
                np.sin(xx + rng.uniform(0, 6)) +
                np.sin(yy + rng.uniform(0, 6)) +
                np.sin((xx + yy) / 2)
            )
            plasma = ((plasma - plasma.min()) / (plasma.max() - plasma.min() + 1e-8) * 255)
            arr = np.stack([
                plasma.astype(np.uint8),
                np.roll(plasma, 85, axis=0).astype(np.uint8),
                np.roll(plasma, 170, axis=1).astype(np.uint8),
            ], axis=2)

        img = Image.fromarray(arr, "RGB")
        img.save(dest_dir / f"noise_{i:04d}.jpg", "JPEG", quality=85)
        generated += 1

    print(f"[Ruído] ✅ {generated} imagens geradas")
    return generated


# ─────────────────────────────────────────────────────────────────────────────
# 5. Verificação e relatório final
# ─────────────────────────────────────────────────────────────────────────────
def verify_and_report(dest_dir: Path):
    """Verifica integridade das imagens e imprime relatório."""
    all_imgs  = list(dest_dir.glob("*.jpg")) + list(dest_dir.glob("*.png"))
    corrupted = []
    size_hist = {"<50KB": 0, "50-200KB": 0, ">200KB": 0}

    print(f"\n[Verificação] Checando {len(all_imgs)} imagens...")
    for p in tqdm(all_imgs, desc="Verificando", unit="img"):
        try:
            img  = Image.open(p)
            img.verify()
            size = p.stat().st_size
            if size < 50_000:
                size_hist["<50KB"] += 1
            elif size < 200_000:
                size_hist["50-200KB"] += 1
            else:
                size_hist[">200KB"] += 1
        except Exception:
            corrupted.append(p)

    if corrupted:
        print(f"⚠️  {len(corrupted)} imagens corrompidas — removendo...")
        for p in corrupted:
            p.unlink(missing_ok=True)

    # Prefix counts
    prefixes = {}
    for p in dest_dir.glob("*.jpg"):
        prefix = p.stem.split("_")[0]
        prefixes[prefix] = prefixes.get(prefix, 0) + 1

    print("\n" + "=" * 55)
    print("  RELATÓRIO FINAL — IMAGENS NEGATIVAS")
    print("=" * 55)
    print(f"  Total coletado : {len(all_imgs) - len(corrupted)}")
    print(f"  Corrompidas    : {len(corrupted)} (removidas)")
    print()
    print("  Por fonte:")
    for prefix, count in sorted(prefixes.items(), key=lambda x: -x[1]):
        label = {
            "coco":  "COCO val2017",
            "oi":    "Open Images v7",
            "synth": "Sintéticas (PIL)",
            "noise": "Ruído/Textura",
        }.get(prefix, prefix)
        print(f"    {label:<22}: {count:>4}")
    print()
    print("  Por tamanho de arquivo:")
    for label, count in size_hist.items():
        print(f"    {label:<12}: {count:>4}")
    print("=" * 55)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Coleta imagens negativas (OOD) para o GC10-DET",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--n", type=int, default=600,
        help="Número total de imagens desejadas (padrão: 600)",
    )
    parser.add_argument(
        "--dest", type=Path, default=BASE_DIR / "images" / "negative",
        help="Pasta de destino (padrão: images/negative)",
    )
    parser.add_argument(
        "--sources", type=str, default="all",
        choices=["all", "coco", "openimages", "synthetic", "noise"],
        help="Quais fontes usar (padrão: all)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dest_dir = args.dest
    dest_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(dest_dir.glob("*.jpg")) + list(dest_dir.glob("*.png")))
    if existing >= args.n:
        print(f"✅ Já existem {existing} imagens em {dest_dir}.")
        print("   Use --n para aumentar o alvo ou delete a pasta para recoletar.")
        verify_and_report(dest_dir)
        return

    print("=" * 55)
    print("  GC10-DET — Coleta de Imagens Negativas (OOD)")
    print("=" * 55)
    print(f"  Destino : {dest_dir}")
    print(f"  Alvo    : {args.n} imagens")
    print(f"  Fontes  : {args.sources}")
    print(f"  Já tem  : {existing} imagens")
    print()

    # Distribui o orçamento entre as fontes
    remaining = args.n - existing
    use_all   = args.sources == "all"

    sources_active = {
        "coco":       use_all or args.sources == "coco",
        "openimages": use_all or args.sources == "openimages",
        "synthetic":  use_all or args.sources == "synthetic",
        "noise":      use_all or args.sources == "noise",
    }
    n_active = sum(sources_active.values())

    # Alocação: sintéticas têm prioridade por incluir dashboards (o caso problemático)
    budgets = {}
    if sources_active["synthetic"]:
        budgets["synthetic"]  = max(int(remaining * 0.40), 50)   # 40% — dashboards
    if sources_active["coco"]:
        budgets["coco"]       = max(int(remaining * 0.35), 50)   # 35% — fotos naturais
    if sources_active["openimages"]:
        budgets["openimages"] = max(int(remaining * 0.15), 30)   # 15%
    if sources_active["noise"]:
        budgets["noise"]      = max(int(remaining * 0.10), 20)   # 10%

    collected = existing

    if sources_active["synthetic"]:
        collected += generate_synthetic(dest_dir, budgets.get("synthetic", 150))

    if sources_active["noise"]:
        collected += generate_noise(dest_dir, budgets.get("noise", 50))

    if sources_active["coco"]:
        collected += collect_coco(dest_dir, budgets.get("coco", 200))

    if sources_active["openimages"]:
        collected += collect_open_images(dest_dir, budgets.get("openimages", 100))

    verify_and_report(dest_dir)

    print("\nPróximos passos:")
    print("  1. Abra o notebook gc10det_treinamento_ood.ipynb")
    print("  2. Execute todas as células")
    print("  3. Substitua gc10det_cls_best.pt por gc10det_cls_ood.pt no app.py")
    print("  4. Reinicie: streamlit run app.py")


if __name__ == "__main__":
    main()
