"""Транслитерация и фонетическое кодирование для русского, казахского и латиницы.

Все обозначения приводятся к единому «фонетическому алфавиту», чтобы
сравнивать звучание независимо от алфавита (Altyn ~ Алтын, Тойказан ~ Toykazan).
Диграфы кодируются одной заглавной буквой:
  J = ж/дж, W = ш/щ, X = ч, C = ц, H = х/һ
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

CYR_MAP = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "o",
    "ж": "J", "з": "z", "и": "i", "й": "i", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "H", "ц": "C", "ч": "X", "ш": "W", "щ": "W", "ъ": "",
    "ы": "i", "ь": "", "э": "e", "ю": "u", "я": "a",
    # казахские буквы
    "ә": "e", "ғ": "g", "қ": "k", "ң": "n", "ө": "o", "ұ": "u", "ү": "u",
    "һ": "H", "і": "i",
}

# Латинские правила: порядок важен (длинные сочетания раньше)
LAT_RULES = [
    ("shch", "W"), ("sch", "W"), ("tch", "X"), ("sh", "W"), ("ch", "X"),
    ("zh", "J"), ("kh", "H"), ("dj", "J"), ("dzh", "J"), ("tz", "C"),
    ("ts", "C"), ("ph", "f"), ("th", "t"), ("ck", "k"), ("qu", "kv"),
    ("ee", "i"), ("oo", "u"), ("ou", "u"), ("ea", "i"),
    ("ya", "a"), ("yu", "u"), ("yo", "o"), ("ye", "e"),
    ("ja", "a"), ("ju", "u"), ("jo", "o"),
    ("x", "ks"), ("w", "v"), ("q", "k"), ("j", "J"), ("y", "i"), ("h", "H"),
    # казахская латиница 2021
    ("ä", "e"), ("ğ", "g"), ("ñ", "n"), ("ö", "o"), ("ū", "u"), ("ü", "u"),
    ("ı", "i"), ("ş", "W"), ("ç", "X"),
]

VOWELS = set("aeiou")
VOICED_TO_VOICELESS = {"b": "p", "v": "f", "g": "k", "d": "t", "z": "s", "J": "W"}
VOICELESS = set("pfktsWXCH")

CYRILLIC_RE = re.compile(r"[а-яёәғқңөұүһі]")
LATIN_RE = re.compile(r"[a-zäğñöūüış]")


def clean(text: str) -> str:
    """Нижний регистр, NFKC, только буквы/цифры/пробелы."""
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = text.replace("&", " and ").replace("+", " plus ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = text.replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


def script_of(text: str) -> str:
    """'cyr', 'lat', 'mixed' или 'none'."""
    t = clean(text)
    c, l = bool(CYRILLIC_RE.search(t)), bool(LATIN_RE.search(t))
    if c and l:
        return "mixed"
    return "cyr" if c else "lat" if l else "none"


def _lat_word(word: str) -> str:
    out = []
    i = 0
    while i < len(word):
        for src, dst in LAT_RULES:
            if word.startswith(src, i):
                out.append(dst)
                i += len(src)
                break
        else:
            ch = word[i]
            if ch == "c":
                nxt = word[i + 1] if i + 1 < len(word) else ""
                out.append("s" if nxt in "eiy" else "k")
            else:
                out.append(ch)
            i += 1
    return "".join(out)


def _word_to_phon(word: str) -> str:
    if CYRILLIC_RE.search(word):
        word = "".join(CYR_MAP.get(ch, ch) for ch in word)
    return _lat_word(word)


@lru_cache(maxsize=200_000)
def phonetic_key(text: str, loose: bool = False) -> str:
    """Фонетический ключ обозначения.

    strict: транскрипция + оглушение на конце слова и перед глухими,
            удвоенные согласные схлопываются.
    loose:  дополнительно редукция безударных гласных (о→а, е→и),
            выпадение «h», i/e в одну группу — моделирует то,
            как обозначение воспринимается на слух в потоке речи.
    """
    words = [_word_to_phon(w) for w in clean(text).split()]
    res = []
    for w in words:
        chars = list(w)
        # оглушение перед глухими и на конце слова
        for i, ch in enumerate(chars):
            nxt = chars[i + 1] if i + 1 < len(chars) else None
            if ch in VOICED_TO_VOICELESS and (nxt is None or nxt in VOICELESS):
                chars[i] = VOICED_TO_VOICELESS[ch]
        w = "".join(chars)
        w = re.sub(r"(.)\1+", r"\1", w)  # удвоения
        if loose:
            w = w.replace("o", "a").replace("e", "i").replace("H", "")
            w = re.sub(r"(.)\1+", r"\1", w)
        res.append(w)
    return " ".join(x for x in res if x)


def consonant_skeleton(key: str) -> str:
    return "".join(ch for ch in key if ch not in VOWELS and ch != " ")


def vowel_count(key: str) -> int:
    """Приблизительное число слогов = число гласных групп."""
    return len(re.findall(r"[aeiou]+", key))


def to_cyrillic(text: str) -> str:
    """Обратная грубая транслитерация фонетического ключа в кириллицу (для показа)."""
    m = {"J": "ж", "W": "ш", "X": "ч", "C": "ц", "H": "х", "a": "а", "b": "б",
         "v": "в", "g": "г", "d": "д", "e": "е", "z": "з", "i": "и", "k": "к",
         "l": "л", "m": "м", "n": "н", "o": "о", "p": "п", "r": "р", "s": "с",
         "t": "т", "u": "у", "f": "ф"}
    return "".join(m.get(ch, ch) for ch in text)
