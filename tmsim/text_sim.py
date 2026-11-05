"""Сравнение словесных обозначений: фонетика, графика (визуальное сходство
написания), тождество, сильные и слабые элементы."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from .translit import (clean, consonant_skeleton, phonetic_key, script_of,
                       vowel_count)

# --------------------------------------------------------------------------
# Базовые метрики
# --------------------------------------------------------------------------

def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def lev_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return 1.0 - levenshtein(a, b) / max(len(a), len(b))


def common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


# --------------------------------------------------------------------------
# Нормализация для проверки тождества
# --------------------------------------------------------------------------

STOP_WORDS = {
    # предлоги, союзы, артикли (рус/каз/англ/фр/нем/ит)
    "в", "во", "на", "с", "со", "у", "к", "ко", "из", "от", "до", "по", "за",
    "для", "о", "об", "и", "или", "да", "мен", "және", "пен", "бен", "үшін",
    "the", "a", "an", "of", "and", "for", "in", "on", "at", "by", "to",
    "de", "la", "le", "les", "du", "des", "el", "il", "der", "die", "das",
    "und", "et",
}

LEGAL_FORMS = {
    "тоо", "ао", "ооо", "зао", "оао", "ип", "llp", "llc", "ltd", "inc", "jsc",
    "gmbh", "corp", "co", "company", "компания", "group", "групп", "холдинг",
    "holding",
}

RU_ENDINGS = sorted([
    "иями", "ями", "ами", "ией", "ием", "иях", "ого", "его", "ому", "ему",
    "ыми", "ими", "ая", "яя", "ое", "ее", "ой", "ей", "ий", "ый", "ую", "юю",
    "ом", "ем", "ам", "ям", "ах", "ях", "ов", "ев", "ы", "и", "а", "я", "о",
    "е", "у", "ю", "ь",
], key=len, reverse=True)
RU_DIMINUTIVE = sorted(["очк", "ечк", "ушк", "юшк", "еньк", "оньк", "ичк",
                        "чик", "ик", "ок", "ек", "ец"], key=len, reverse=True)
KZ_SUFFIXES = sorted([
    "лардың", "лердің", "дардың", "дердің", "тардың", "тердің",
    "лар", "лер", "дар", "дер", "тар", "тер", "ның", "нің", "дың", "дің",
    "тың", "тің", "ға", "ге", "қа", "ке", "да", "де", "та", "те", "дан",
    "ден", "тан", "тен", "нан", "нен", "мен", "бен", "пен", "ым", "ім",
    "ың", "ің", "сы", "сі",
], key=len, reverse=True)
EN_ENDINGS = ["ies", "es", "s", "'s"]


@lru_cache(maxsize=100_000)
def stem(word: str) -> str:
    """Лёгкий стеммер: снимает падежные/числовые окончания и уменьшительные
    суффиксы (критерии тождества: различие только в падеже, числе,
    уменьшительной форме)."""
    w = word
    if re.search(r"[әғқңөұүһі]", w):
        for suf in KZ_SUFFIXES:
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                return w[: -len(suf)]
        return w
    if re.search(r"[а-яё]", w):
        w = w.replace("ё", "е")
        for end in RU_ENDINGS:
            if w.endswith(end) and len(w) - len(end) >= 3:
                w = w[: -len(end)]
                break
        for suf in RU_DIMINUTIVE:
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                w = w[: -len(suf)]
                break
        return w
    for end in EN_ENDINGS:
        if w.endswith(end) and len(w) - len(end) >= 3:
            return w[: -len(end)]
    return w


def tokens(text: str) -> list[str]:
    return [t for t in clean(text).split() if t]


def identity_form(text: str) -> tuple[str, ...]:
    toks = [t for t in tokens(text) if t not in STOP_WORDS and t not in LEGAL_FORMS]
    return tuple(stem(t) for t in toks)


# --------------------------------------------------------------------------
# Сильные и слабые элементы
# --------------------------------------------------------------------------

WEAK_WORDS = {
    # хвалебные / рекламные
    "super", "супер", "mega", "мега", "hyper", "гипер", "ultra", "ультра",
    "extra", "экстра", "premium", "премиум", "best", "лучший", "top", "топ",
    "lux", "люкс", "luxe", "gold", "голд", "golden", "royal", "роял",
    "prime", "прайм", "pro", "про", "plus", "плюс", "max", "макс", "maxi",
    "mini", "мини", "new", "новый", "нью", "classic", "классик", "original",
    "оригинал", "smart", "смарт", "star", "стар", "vip", "вип", "first",
    "number", "one", "1", "№1", "үздік", "жаңа", "ең",
    # описательные (эко, био, вкус, качество)
    "eco", "эко", "bio", "био", "organic", "органик", "natural", "натурал",
    "fresh", "фреш", "свежий", "market", "маркет",
    "shop", "шоп", "store", "стор", "service", "сервис", "сервис", "center",
    "центр", "city", "сити", "life", "лайф", "style", "стиль", "home", "хоум",
    "house", "хаус", "dom", "дом", "land", "ленд", "world", "мир", "art",
    "арт", "tech", "тех", "техно", "techno", "soft", "софт", "net", "нет",
    "online", "онлайн", "trade", "трейд", "invest", "инвест", "build",
    "строй", "auto", "авто", "med", "мед", "медикал", "medical", "pharm",
    "фарм", "kids", "кидс", "baby", "беби", "beauty", "бьюти", "fit", "фит",
    "express", "экспресс", "logistic", "логистик", "consulting", "консалтинг",
    "cafe", "кафе", "coffee", "кофе", "bar", "бар", "pizza", "пицца",
    # географические
    "kz", "кз", "kazakhstan", "казахстан", "қазақстан", "qazaq", "казах",
    "қазақ", "almaty", "алматы", "astana", "астана", "asia", "азия",
    "eurasia", "евразия", "steppe", "степь", "dala", "дала",
}


def is_weak(token: str) -> bool:
    t = token.lower()
    if t in WEAK_WORDS or t in LEGAL_FORMS or t in STOP_WORDS:
        return True
    if t.isdigit():
        return True
    if len(t) <= 1:
        return True
    return stem(t) in {stem(w) for w in WEAK_WORDS if len(w) > 3}


def split_compound(token: str) -> list[str]:
    """SuperFood → [super, food]; ЭкоДом → [эко, дом]. Работает по известным
    слабым словам в начале/конце слитного написания."""
    t = token.lower()
    for w in sorted(WEAK_WORDS, key=len, reverse=True):
        if len(w) >= 3 and t.startswith(w) and len(t) - len(w) >= 3:
            return [w] + split_compound(t[len(w):])
        if len(w) >= 3 and t.endswith(w) and len(t) - len(w) >= 3:
            return split_compound(t[: -len(w)]) + [w]
    return [t]


def camel_split(text: str) -> str:
    return re.sub(r"(?<=[a-zа-яё])(?=[A-ZА-ЯЁ])", " ", text or "")


def elements(text: str) -> list[tuple[str, bool]]:
    """Список (элемент, сильный?) с разбором слитных и CamelCase написаний."""
    out = []
    for t in tokens(camel_split(text)):
        if t in STOP_WORDS:
            continue
        parts = split_compound(t)
        for p in parts:
            out.append((p, not is_weak(p)))
    return out


# --------------------------------------------------------------------------
# Похожие по начертанию буквы (визуальное сходство написания)
# --------------------------------------------------------------------------

HOMOGLYPHS = {
    # кириллица → латиница одинакового начертания
    "а": "a", "в": "b", "е": "e", "ё": "e", "к": "k", "м": "m", "н": "h",
    "о": "o", "р": "p", "с": "c", "т": "t", "у": "y", "х": "x", "і": "i",
    "ә": "e", "ө": "o", "ү": "y", "ұ": "y", "қ": "k", "ң": "h", "һ": "h",
    "ғ": "f", "г": "r", "п": "n", "и": "u", "л": "n", "д": "g",
    # латиница: похожие по форме
    "0": "o", "1": "l", "i": "l", "j": "l", "rn": "m", "vv": "w", "u": "u",
    "q": "g", "5": "s", "3": "e", "8": "b",
}


def shape_form(text: str) -> str:
    t = clean(text).replace(" ", "")
    for a, b in sorted(HOMOGLYPHS.items(), key=lambda kv: -len(kv[0])):
        t = t.replace(a, b)
    return t


# --------------------------------------------------------------------------
# Результаты
# --------------------------------------------------------------------------

@dataclass
class Criterion:
    score: float
    details: list[str] = field(default_factory=list)
    parts: dict[str, float] = field(default_factory=dict)

    def as_dict(self):
        return {"score": round(self.score, 4), "details": self.details,
                "parts": {k: round(v, 4) for k, v in self.parts.items()}}


DEFAULT_PHONETIC_PARTS = {
    "strict": 0.35,      # совпадение звуков (строгая транскрипция)
    "loose": 0.20,       # с учётом редукции гласных / восприятия на слух
    "consonants": 0.15,  # совпадение согласного «скелета»
    "beginning": 0.20,   # совпадение начальных частей
    "syllables": 0.10,   # близость числа слогов и длины
}


def _phon_pair(a: str, b: str, w: dict) -> tuple[float, dict]:
    sa, sb = phonetic_key(a), phonetic_key(b)
    la, lb = phonetic_key(a, True), phonetic_key(b, True)
    sa_, sb_ = sa.replace(" ", ""), sb.replace(" ", "")
    la_, lb_ = la.replace(" ", ""), lb.replace(" ", "")
    parts = {
        "strict": lev_ratio(sa_, sb_),
        "loose": lev_ratio(la_, lb_),
        "consonants": lev_ratio(consonant_skeleton(sa_), consonant_skeleton(sb_)),
        "beginning": min(1.0, common_prefix(la_, lb_) / max(1, min(4, len(la_), len(lb_)))),
    }
    va, vb = vowel_count(la_), vowel_count(lb_)
    syl = 1.0 - abs(va - vb) / max(1, va, vb)
    ln = min(len(la_), len(lb_)) / max(1, len(la_), len(lb_))
    parts["syllables"] = 0.6 * syl + 0.4 * ln
    total_w = sum(w.values()) or 1
    score = sum(parts[k] * w.get(k, 0) for k in parts) / total_w
    # вхождение одного обозначения в другое
    short, long_ = sorted([la_, lb_], key=len)
    # вхождение: короткое обозначение целиком читается внутри длинного.
    # Для очень коротких (3 буквы) учитываем только как отдельное слово —
    # иначе «ГОР» «входит» в «Гарнир».
    words_long = set(max([la, lb], key=len).split())
    if short != long_ and short in long_ and (len(short) >= 4 or short in words_long):
        inc = 0.6 + 0.35 * len(short) / len(long_)
        if inc > score:
            score = inc
            parts["inclusion"] = inc
    return score, parts


def phonetic_similarity(a: str, b: str, weights: dict | None = None,
                        weak_weight: float = 0.3) -> Criterion:
    w = weights or DEFAULT_PHONETIC_PARTS
    whole, parts = _phon_pair(a, b, w)
    det = [f"Транскрипция: «{phonetic_key(a)}» ↔ «{phonetic_key(b)}»"]

    # сравнение по сильным элементам (экспертиза отдельных слов словосочетания)
    ea, eb = elements(a), elements(b)
    strong_a = [e for e, s in ea if s] or [e for e, _ in ea]
    strong_b = [e for e, s in eb if s] or [e for e, _ in eb]
    best_strong = 0.0
    best_pair = None
    if (len(ea) > 1 or len(eb) > 1) and strong_a and strong_b:
        # средневзвешенное лучшее совпадение сильных элементов обеих сторон
        def side(xs, ys):
            vals = []
            for x in xs:
                m = max(((_phon_pair(x, y, w)[0], y) for y in ys), key=lambda t: t[0])
                vals.append(m)
            return vals
        va, vb = side(strong_a, strong_b), side(strong_b, strong_a)
        sa = sum(v for v, _ in va) / len(va)
        sb = sum(v for v, _ in vb) / len(vb)
        best_strong = max(sa, sb)
        top = max(va, key=lambda t: t[0])
        best_pair = (strong_a[va.index(top)], top[1], top[0])
        # слабые элементы не должны тянуть оценку: смешиваем
        best_strong = best_strong * (1 - weak_weight * 0.3)
        parts["strong_elements"] = best_strong
    score = max(whole, best_strong)
    if best_pair and best_strong >= whole:
        det.append(f"Решающее сравнение по сильным элементам: «{best_pair[0]}» ↔ «{best_pair[1]}» ({best_pair[2]:.2f})")
    weak_a = [e for e, s in ea if not s]
    weak_b = [e for e, s in eb if not s]
    if weak_a or weak_b:
        det.append("Слабые (неохраняемые/описательные) элементы: "
                   + ", ".join(sorted(set(weak_a + weak_b))))
    if parts.get("beginning", 0) >= 0.99:
        det.append("Совпадают начальные части обозначений (на них падает основное внимание потребителя)")
    if "inclusion" in parts:
        det.append("Одно обозначение фонетически входит в другое")
    return Criterion(min(1.0, score), det, parts)


DEFAULT_VISUAL_TEXT_PARTS = {
    "letters": 0.50,   # совпадение букв с учётом похожего начертания
    "alphabet": 0.20,  # тип алфавита
    "length": 0.15,    # длина / общее зрительное впечатление
    "structure": 0.15, # число слов, регистр, расположение
}


def visual_text_similarity(a: str, b: str, weights: dict | None = None) -> Criterion:
    w = weights or DEFAULT_VISUAL_TEXT_PARTS
    fa, fb = shape_form(a), shape_form(b)
    sca, scb = script_of(a), script_of(b)
    alph = 1.0 if sca == scb else 0.6 if "mixed" in (sca, scb) else 0.3
    ta, tb = tokens(a), tokens(b)
    ra, rb = (a or "").strip(), (b or "").strip()
    case_a = "upper" if ra.isupper() else "lower" if ra.islower() else "mixed"
    case_b = "upper" if rb.isupper() else "lower" if rb.islower() else "mixed"
    structure = (1.0 if len(ta) == len(tb) else 0.6) * (1.0 if case_a == case_b else 0.85)
    parts = {
        "letters": lev_ratio(fa, fb),
        "alphabet": alph,
        "length": min(len(fa), len(fb)) / max(1, len(fa), len(fb)),
        "structure": structure,
    }
    score = sum(parts[k] * w.get(k, 0) for k in parts) / (sum(w.values()) or 1)
    names = {"cyr": "кириллица", "lat": "латиница", "mixed": "смешанный", "none": "—"}
    det = [f"Алфавит: {names[sca]} / {names[scb]}",
           f"Совпадение начертания букв: {parts['letters']:.2f}"]
    return Criterion(score, det, parts)


def identity_check(a: str, b: str) -> tuple[bool, str]:
    """Тождество: та же грамматическая форма или отличие только в падеже,
    числе, уменьшительной форме, наличии/отсутствии предлога."""
    if clean(a) == clean(b) and clean(a):
        return True, "Обозначения совпадают полностью"
    ia, ib = identity_form(a), identity_form(b)
    if ia and ia == ib:
        return True, "Обозначения отличаются только падежом, числом, уменьшительной формой или предлогом"
    if clean(a).replace(" ", "") == clean(b).replace(" ", "") and clean(a):
        return True, "Обозначения отличаются только пробелами/знаками препинания"
    return False, ""
