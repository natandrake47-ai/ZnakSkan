"""Итоговая оценка сходства двух товарных знаков.

Схема (все веса — в config/weights.json):

 1. Словесный элемент:
      фонетическое S_ф, смысловое S_с, графическое (написание) S_г
      взвешенное  = w_ф·S_ф + w_с·S_с + w_г·S_г
      доминирующее = max(S_ф, S_с, 0.9·S_г)
      S_слово = β·доминирующее + (1−β)·взвешенное,  β = max_blend
    Смешение «максимума» и «среднего» отражает практику экспертизы:
    сходство может быть признано и по одному признаку (например, только
    по звучанию), но совпадение по нескольким признакам усиливает вывод.
    Тождество (отличие только падежом, числом, уменьшительной формой,
    предлогом) → S_слово = 1.

 2. Изобразительный элемент: S_изобр (форма, композиция, цвет, CLIP).

 3. Комбинированный знак:
      S_знак = max(w_сл·S_слово + w_из·S_изобр, 0.85·S_слово, 0.75·S_изобр)
    (словесный элемент обычно сильнее: его запоминают и произносят).

 4. Однородность товаров/услуг (классы МКТУ) K ∈ [0..1]:
      Риск смешения = S_знак · (1 − λ + λ·K),  λ = classes.influence
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import image_sim
from .config import weights as load_weights
from .nice import class_similarity, goods_overlap
from .semantic import semantic_similarity
from .text_sim import (Criterion, identity_check, phonetic_similarity,
                       visual_text_similarity)


@dataclass
class Mark:
    name: str = ""
    classes: list[int] = field(default_factory=list)
    goods: str | None = None
    image_feat: dict | None = None
    image_clip: bytes | None = None
    image_desc: str | None = None   # что изображено (для смыслового сравнения с изображением)
    label: str = ""       # для отображения (номер регистрации и т.п.)
    meta: dict = field(default_factory=dict)

    @property
    def has_word(self) -> bool:
        return bool((self.name or "").strip())

    @property
    def has_image(self) -> bool:
        return bool(self.image_feat)


LEVELS = [
    ("high", "Высокая", "Обозначения, вероятно, сходны до степени смешения — высокий риск отказа/возражения"),
    ("medium", "Средняя", "Есть существенное сходство — нужна оценка эксперта, риск смешения возможен"),
    ("low", "Низкая", "Сходство незначительное, смешение маловероятно"),
    ("none", "Нет", "Обозначения не сходны"),
]


def level_of(score: float, th: dict) -> tuple[str, str, str]:
    if score >= th["high"]:
        return LEVELS[0]
    if score >= th["medium"]:
        return LEVELS[1]
    if score >= th["low"]:
        return LEVELS[2]
    return LEVELS[3]


def word_similarity(a: str, b: str, w: dict, use_ai: bool = True) -> dict:
    ident, ident_reason = identity_check(a, b)
    ph = phonetic_similarity(a, b, w.get("phonetic_parts"), w.get("weak_element_weight", 0.3))
    vi = visual_text_similarity(a, b, w.get("visual_text_parts"))
    se = semantic_similarity(a, b, use_ai=use_ai)
    ww = dict(w["word"])
    if not se.parts.get("applicable", 1.0):
        ww["semantic"] = 0.0  # фантазийные обозначения: смысл не сравнивается
    weighted = (ww["phonetic"] * ph.score + ww["semantic"] * se.score + ww["visual"] * vi.score) / (
        sum(ww.values()) or 1)
    dominant = max(ph.score, se.score, 0.9 * vi.score)
    beta = w.get("max_blend", 0.5)
    score = beta * dominant + (1 - beta) * weighted
    if ident:
        score = 1.0
    return {"score": score, "identity": ident, "identity_reason": ident_reason,
            "weighted": weighted, "dominant": dominant,
            "phonetic": ph.as_dict(), "visual": vi.as_dict(), "semantic": se.as_dict()}


def compare(a: Mark, b: Mark, use_ai: bool = True, w: dict | None = None) -> dict:
    w = w or load_weights()
    res: dict = {"a": a.label or a.name, "b": b.label or b.name, "notes": []}
    word = img = None
    if a.has_word and b.has_word:
        word = word_similarity(a.name, b.name, w, use_ai)
        res["word"] = word
    if a.has_image and b.has_image:
        c = image_sim.compare(a.image_feat, b.image_feat, a.image_clip, b.image_clip,
                              w.get("image_parts"), w.get("image_parts_no_clip"))
        img = c.as_dict()
        res["image"] = img
    if img and a.image_desc and b.image_desc:
        # одинаковые изображённые объекты (два разных рисунка льва) — смысловое сходство изображений
        sd = semantic_similarity(a.image_desc, b.image_desc, use_ai=use_ai)
        if sd.score >= 0.5 and 0.8 * sd.score > img["score"]:
            img["score"] = 0.8 * sd.score
            img["details"].append(f"Изображены сходные объекты: «{a.image_desc}» / «{b.image_desc}»")
    if not word and a.has_word != b.has_word:
        # слово против изображения: «Кобра» против рисунка кобры
        w_side, i_side = (a, b) if a.has_word else (b, a)
        if i_side.image_desc:
            sd = semantic_similarity(w_side.name, i_side.image_desc, use_ai=use_ai)
            if sd.score > 0:
                word = {"score": 0.8 * sd.score, "identity": False, "identity_reason": "",
                        "cross": True,
                        "phonetic": {"score": 0, "details": [], "parts": {}},
                        "visual": {"score": 0, "details": [], "parts": {}},
                        "semantic": {"score": sd.score, "parts": sd.parts,
                                     "details": [f"Слово «{w_side.name}» и изображение ({i_side.image_desc})"] + sd.details}}
                res["word"] = word
    if word and img:
        cw = w["combined"]
        s = max(cw["word"] * word["score"] + cw["image"] * img["score"],
                0.85 * word["score"], 0.75 * img["score"])
        kind = "комбинированные обозначения"
    elif word:
        s = word["score"]
        kind = "словесные элементы"
        if a.has_image != b.has_image:
            res["notes"].append("Изображение есть только у одного знака — сравнивались словесные элементы")
    elif img:
        s = img["score"]
        kind = "изобразительные элементы"
        if a.has_word != b.has_word:
            res["notes"].append("Словесный элемент есть только у одного знака — сравнивались изображения")
    else:
        s = 0.0
        kind = "—"
        res["notes"].append("Нечего сравнивать: у знаков нет общего типа элементов (текст/изображение)")
    res["mark_similarity"] = round(s, 4)

    k, cdet, cinfo = class_similarity(a.classes, b.classes, w["classes"])
    ov = goods_overlap(a.goods, b.goods)
    if ov is not None and cinfo.get("relation") in ("same", "related"):
        k = 0.7 * k + 0.3 * max(ov, 0.0)
        cdet.append(f"Пересечение перечней товаров/услуг: {ov:.0%}")
    lam = w["classes"].get("influence", 0.5)
    risk = s * (1 - lam + lam * k)
    res["classes"] = {"score": round(k, 4), "details": cdet, **cinfo}
    res["risk"] = round(risk, 4)
    lvl, lvl_name, lvl_text = level_of(risk, w["thresholds"])
    res["level"], res["level_name"], res["verdict"] = lvl, lvl_name, lvl_text
    res["compared"] = kind
    res["explanation"] = explain(res)
    return res


def explain(r: dict) -> list[str]:
    out = []
    wd = r.get("word")
    if wd:
        if wd["identity"]:
            out.append(f"Тождество словесных элементов: {wd['identity_reason']}.")
        crit = [("звуковому (фонетическому)", wd["phonetic"]["score"]),
                ("смысловому (семантическому)", wd["semantic"]["score"]),
                ("графическому (визуальному)", wd["visual"]["score"])]
        strong = [n for n, v in crit if v >= 0.75]
        if strong:
            out.append("Сходство по " + ", ".join(strong) + " признаку.")
    if r.get("image") and r["image"]["score"] >= 0.7:
        out.append("Изобразительные элементы сходны по общему зрительному впечатлению.")
    out.append(r["classes"]["details"][0] if r["classes"]["details"] else "")
    out.append(f"Итог: сходство знаков {r['mark_similarity']:.0%}, с учётом однородности товаров риск смешения {r['risk']:.0%} — {r['verdict'].lower()}.")
    return [x for x in out if x]
