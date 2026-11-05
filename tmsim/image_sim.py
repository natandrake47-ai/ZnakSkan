"""Сравнение изображений (изобразительных элементов знаков).

Критерии визуального сходства: общее зрительное впечатление, форма и
композиция, сочетание цветов. Реализовано без обязательных тяжёлых
зависимостей (Pillow + NumPy). Если установлен sentence-transformers,
дополнительно используется нейросеть CLIP — она лучше ловит «то же самое
изображённое» (два разных рисунка льва и т.п.).
"""
from __future__ import annotations

import io
import logging
from functools import lru_cache

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from .config import settings
from .text_sim import Criterion

log = logging.getLogger(__name__)


def load(data: bytes | str) -> Image.Image:
    img = Image.open(io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data)
    return load_pil(img)


def load_pil(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("RGB")


def crop_content(img: Image.Image, thr: int = 240) -> Image.Image:
    g = np.asarray(img.convert("L"))
    mask = g < thr
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    img = img.crop(box)
    w, h = img.size
    s = max(w, h)
    sq = Image.new("RGB", (s, s), (255, 255, 255))
    sq.paste(img, ((s - w) // 2, (s - h) // 2))
    return sq


def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n)
    m = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n))
    m[0] *= 1 / np.sqrt(2)
    return m * np.sqrt(2 / n)


_D32 = _dct_matrix(32)


def _phash(gray: Image.Image) -> int:
    a = np.asarray(gray.resize((32, 32), Image.LANCZOS), dtype=np.float64)
    d = _D32 @ a @ _D32.T
    low = d[:8, :8].flatten()
    med = np.median(low[1:])
    return int("".join("1" if v > med else "0" for v in low), 2)


def _dhash(gray: Image.Image) -> int:
    a = np.asarray(gray.resize((9, 8), Image.LANCZOS), dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _hog(gray: Image.Image, cells: int = 4, bins: int = 9) -> list[float]:
    a = np.asarray(gray.resize((64, 64), Image.LANCZOS), dtype=np.float64)
    gx = np.zeros_like(a); gy = np.zeros_like(a)
    gx[:, 1:-1] = a[:, 2:] - a[:, :-2]
    gy[1:-1, :] = a[2:, :] - a[:-2, :]
    mag = np.hypot(gx, gy)
    ang = (np.degrees(np.arctan2(gy, gx)) % 180) / (180 / bins)
    idx = np.minimum(ang.astype(int), bins - 1)
    feat = []
    step = 64 // cells
    for i in range(cells):
        for j in range(cells):
            m = mag[i * step:(i + 1) * step, j * step:(j + 1) * step].flatten()
            b = idx[i * step:(i + 1) * step, j * step:(j + 1) * step].flatten()
            feat.extend(np.bincount(b, weights=m, minlength=bins).tolist())
    v = np.array(feat)
    n = np.linalg.norm(v)
    return (v / n if n else v).round(5).tolist()


def _grid(gray: Image.Image, n: int = 12) -> list[float]:
    a = 1.0 - np.asarray(gray.resize((n, n), Image.BOX), dtype=np.float64) / 255.0
    return a.flatten().round(4).tolist()


def _color_hist(img: Image.Image) -> tuple[list[float], float]:
    hsv = np.asarray(img.resize((96, 96)).convert("HSV"), dtype=np.int32).reshape(-1, 3)
    rgb = np.asarray(img.resize((96, 96)), dtype=np.int32).reshape(-1, 3)
    ink = rgb.min(axis=1) < 235  # исключаем белый фон
    hsv = hsv[ink]
    if len(hsv) == 0:
        return [0.0] * 60, 0.0
    sat = hsv[:, 1]
    chroma = sat > 50
    # 12 оттенков × 4 яркости для цветных + 12 градаций серого
    h = (hsv[:, 0] * 12 // 256)
    v = (hsv[:, 2] * 4 // 256)
    hist = np.zeros(60)
    np.add.at(hist, (h * 4 + v)[chroma], 1)
    np.add.at(hist, 48 + (hsv[:, 2] * 12 // 256)[~chroma], 1)
    hist /= hist.sum()
    return hist.round(5).tolist(), float(chroma.mean())


@lru_cache(maxsize=1)
def _clip_model():
    s = settings()
    if not s.get("use_clip"):
        return None
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(s.get("clip_model", "clip-ViT-B-32"))
    except Exception as e:
        log.info("CLIP недоступен (%s) — используется классический анализ изображений", e)
        return None


def clip_embedding(img: Image.Image) -> np.ndarray | None:
    m = _clip_model()
    if m is None:
        return None
    v = m.encode([img], convert_to_numpy=True, normalize_embeddings=True)[0]
    return v.astype(np.float32)


def features(data: bytes | str | Image.Image, with_clip: bool = True) -> tuple[dict, bytes | None]:
    img = load_pil(data) if isinstance(data, Image.Image) else load(data)
    img = crop_content(img)
    gray = img.convert("L").filter(ImageFilter.GaussianBlur(0.6))
    mirror = ImageOps.mirror(gray)
    hist, chroma = _color_hist(img)
    feat = {
        "phash": f"{_phash(gray):016x}", "phash_m": f"{_phash(mirror):016x}",
        "dhash": f"{_dhash(gray):016x}", "hog": _hog(gray), "grid": _grid(gray),
        "color": hist, "chroma": round(chroma, 4),
        "aspect": 1.0,
    }
    clip = clip_embedding(img) if with_clip else None
    return feat, (clip.tobytes() if clip is not None else None)


def _ham_sim(a: str, b: str) -> float:
    d = bin(int(a, 16) ^ int(b, 16)).count("1")
    s = 1 - d / 64
    return max(0.0, (s - 0.5) / 0.5)  # случайные изображения ≈ 0.5 → 0


def _cos(u, v) -> float:
    u, v = np.asarray(u), np.asarray(v)
    n = np.linalg.norm(u) * np.linalg.norm(v)
    return float(u @ v / n) if n else 0.0


def _corr(u, v) -> float:
    u, v = np.asarray(u), np.asarray(v)
    if u.std() == 0 or v.std() == 0:
        return 0.0
    return float(np.corrcoef(u, v)[0, 1])


def compare(fa: dict, fb: dict, clip_a: bytes | None = None, clip_b: bytes | None = None,
            weights: dict | None = None, weights_no_clip: dict | None = None) -> Criterion:
    ph = max(_ham_sim(fa["phash"], fb["phash"]), _ham_sim(fa["phash_m"], fb["phash"]))
    dh = _ham_sim(fa["dhash"], fb["dhash"])
    hog = max(0.0, (_cos(fa["hog"], fb["hog"]) - 0.55) / 0.45)
    grid = max(0.0, _corr(fa["grid"], fb["grid"]))
    color = float(np.minimum(fa["color"], fb["color"]).sum())
    if abs(fa.get("chroma", 0) - fb.get("chroma", 0)) > 0.5:
        color *= 0.6  # цветное vs чёрно-белое
    parts = {
        "shape": 0.5 * ph + 0.2 * dh + 0.3 * hog,
        "structure": grid,
        "color": color,
    }
    det = []
    if clip_a and clip_b:
        ca = np.frombuffer(clip_a, dtype=np.float32)
        cb = np.frombuffer(clip_b, dtype=np.float32)
        c = float(ca @ cb)
        parts["clip"] = max(0.0, min(1.0, (c - 0.70) / 0.25))
        w = weights or {"clip": 0.45, "shape": 0.30, "structure": 0.10, "color": 0.15}
    else:
        w = weights_no_clip or {"shape": 0.55, "structure": 0.25, "color": 0.20}
        det.append("Нейросеть CLIP не установлена — оценка по форме, композиции и цвету")
    score = sum(parts[k] * w.get(k, 0) for k in parts) / (sum(w[k] for k in w if k in parts) or 1)
    det.insert(0, f"Форма/контур: {parts['shape']:.2f}; композиция: {parts['structure']:.2f}; цвет: {parts['color']:.2f}"
               + (f"; общее впечатление (CLIP): {parts['clip']:.2f}" if "clip" in parts else ""))
    if ph > 0.9 or dh > 0.9:
        det.append("Изображения практически совпадают (возможно зеркальное отражение или изменение масштаба)")
    return Criterion(min(1.0, score), det, parts)
