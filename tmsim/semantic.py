"""Смысловое (семантическое) сходство: подобие заложенных понятий и идей,
в том числе когда обозначения написаны на разных языках.

Уровни (используются все доступные, берётся максимум):
 1. встроенный многоязычный словарь понятий (рус/каз/англ) — работает всегда;
 2. эмбеддинги многоязычной модели через Ollama (bge-m3 и т.п.);
 3. оценка локальной LLM с обоснованием (кэшируется в базе).
"""
from __future__ import annotations

import hashlib
import json
import logging
import math

from . import db, ollama
from .config import settings
from .text_sim import Criterion, elements, is_weak, stem, tokens
from .translit import phonetic_key

log = logging.getLogger(__name__)

# Каждая строка — одно понятие: варианты на русском, казахском, английском
# (и иногда других языках). Список можно расширять в data/concepts.txt
CONCEPTS_RAW = """
солнце|солнышко|күн|sun|sunny|sol|soleil
луна|ай|moon|luna|lune
звезда|звёздочка|жұлдыз|star|stella|etoile|stern
небо|аспан|көк|sky|heaven
земля|жер|earth|land|terra
вода|су|water|aqua|agua
огонь|от|fire|fuego|ignis
ветер|жел|wind
гора|тау|mountain|mount
степь|дала|steppe
река|өзен|river
море|теңіз|sea|marine|mare
золото|золотой|алтын|gold|golden|oro|aurum
серебро|күміс|silver|argent
железо|темір|iron
лев|арыстан|lion|leo|leon
волк|қасқыр|бөрі|wolf|lupus
орёл|орел|бүркіт|қыран|eagle|aquila
барс|ирбис|снежный барс|барыс|leopard|snow leopard
медведь|аю|bear
конь|лошадь|ат|жылқы|horse|equus|тұлпар|скакун
верблюд|түйе|camel
сокол|сұңқар|лочин|falcon
птица|құс|bird
рыба|балық|fish
пчела|ара|bee
мёд|мед|бал|honey|miel
молоко|сүт|milk|lacto
хлеб|нан|bread|pan
мясо|ет|meat
еда|пища|тамақ|food|foods|фуд
сыр|ірімшік|cheese|fromage
яблоко|алма|apple|pomme|apfel
вишня|шие|cherry
виноград|жүзім|grape
чай|шай|tea|chai
кофе|кофе|coffee|caffe
сахар|қант|sugar
соль|тұз|salt
вкус|дәм|taste|flavor|flavour
дом|үй|house|home|domus|casa
город|қала|city|town
мир|әлем|world|mundo
мир|бейбітшілік|peace|pax
свет|жарық|light|lux|lumen
жизнь|өмір|life|vita|vida
здоровье|денсаулық|health|salus
красота|сұлулық|ару|beauty|belle|bella
любовь|махаббат|love|amor|amour
радость|қуаныш|joy
счастье|бақыт|happiness|happy|lucky
друг|дос|friend|amigo
семья|отбасы|family|familia
ребёнок|ребенок|дети|бала|балалар|kid|kids|child|baby|bebe
мама|ана|mother|mom|mama
отец|ата|әке|father|papa
король|патша|хан|king|rex|roi|rey
королева|ханшайым|queen|regina|reine
принц|ханзада|prince
корона|тәж|crown|corona
путь|жол|way|road|path|route
время|уақыт|time|tempo
новый|жаңа|new|nova|novo|neo
старый|ескі|old|vintage
большой|үлкен|big|grand|maxi|magnum
маленький|кіші|small|mini|little|petit
лучший|үздік|best|optimum|optima
первый|бірінші|first|primo|prima
быстрый|жылдам|fast|quick|rapid|express|экспресс
сильный|күшті|strong|power|forte
умный|ақылды|smart|clever
чистый|таза|clean|pure|clear
зелёный|зеленый|жасыл|green|verde
белый|ақ|white|blanc|bianco
чёрный|черный|қара|black|noir|nero
красный|қызыл|red|rouge|rosso
синий|голубой|көк|blue|bleu|azul
цветок|гүл|flower|flora|fleur
сад|бақ|garden|jardin
лес|орман|forest|wood|woods
дерево|ағаш|tree
ключ|кілт|key
сердце|жүрек|heart|cor|coeur
глаз|көз|eye
рука|қол|hand
бизнес|іскер|business
деньги|ақша|money|cash
банк|bank|банкі
оазис|oasis
снег|қар|snow
лёд|лед|мұз|ice
жемчуг|інжу|pearl|perla
алмаз|бриллиант|гауһар|diamond|diamant
""".strip()


def _build_concepts():
    idx: dict[str, set[int]] = {}
    for i, line in enumerate(CONCEPTS_RAW.splitlines()):
        for w in line.split("|"):
            w = w.strip().lower()
            if w:
                idx.setdefault(stem(w), set()).add(i)
                idx.setdefault(w, set()).add(i)
    return idx, CONCEPTS_RAW.splitlines()


CONCEPT_INDEX, CONCEPT_LINES = _build_concepts()


def concepts_of(text: str) -> dict[int, str]:
    out = {}
    for t in tokens(text):
        for key in (t, stem(t)):
            for cid in CONCEPT_INDEX.get(key, ()):
                out[cid] = t
    # слитные написания: SuperFood → food
    for e, _strong in elements(text):
        for cid in CONCEPT_INDEX.get(e, set()) | CONCEPT_INDEX.get(stem(e), set()):
            out.setdefault(cid, e)
    return out


def dictionary_similarity(a: str, b: str) -> tuple[float, list[str]]:
    ca, cb = concepts_of(a), concepts_of(b)
    if not ca or not cb:
        return 0.0, []
    common = set(ca) & set(cb)
    if not common:
        return 0.0, []
    score = len(common) / max(len(ca), len(cb))
    strong_common = [c for c in common if not (is_weak(ca[c]) and is_weak(cb[c]))]
    # совпадение главного (сильного) понятия — уже значительное смысловое сходство;
    # совпадение только по слабым словам (gold, new, best...) — незначительное
    score = max(score, 0.75) if strong_common else min(score, 0.4)
    det = [f"Общее понятие: «{ca[c]}» ≈ «{cb[c]}» ({CONCEPT_LINES[c].split('|')[0]})"
           for c in sorted(common)]
    return min(1.0, score), det


def _cos(u, v) -> float:
    num = sum(x * y for x, y in zip(u, v))
    den = math.sqrt(sum(x * x for x in u)) * math.sqrt(sum(y * y for y in v))
    return num / den if den else 0.0


def embedding_similarity(a: str, b: str) -> float | None:
    if not settings().get("use_embeddings"):
        return None
    key = "emb:" + hashlib.sha1(f"{a}\x00{b}".encode()).hexdigest()
    cached = db.cache_get(key)
    if cached is not None:
        return cached
    vecs = ollama.embed([a, b])
    if not vecs:
        return None
    cos = _cos(vecs[0], vecs[1])
    # у эмбеддингов даже несвязанные слова дают cos≈0.3–0.5: растягиваем
    val = max(0.0, min(1.0, (cos - 0.55) / 0.40))
    db.cache_set(key, val)
    return val


LLM_PROMPT = """Ты эксперт по товарным знакам (Казпатент, Республика Казахстан).
Оцени ТОЛЬКО смысловое (семантическое) сходство двух обозначений: подобие заложенных понятий и идей,
совпадение значения при переводе с одного языка на другой (русский, казахский, английский и др.),
противоположность идей. Звучание и написание НЕ учитывай.
Обозначение 1: «{a}»
Обозначение 2: «{b}»
Если хотя бы одно обозначение фантазийное (не имеет смыслового значения ни на одном языке), поставь "applicable": false.
Ответь строго JSON: {{"score": число от 0 до 1, "applicable": true/false, "meaning_1": "...", "meaning_2": "...", "reason": "кратко по-русски"}}"""


def llm_similarity(a: str, b: str) -> tuple[float, str] | None:
    s = settings()
    if not s.get("use_llm_semantic"):
        return None
    key = "llmsem:" + s["chat_model"] + ":" + hashlib.sha1(
        "\x00".join(sorted([a.lower(), b.lower()])).encode()).hexdigest()
    cached = db.cache_get(key)
    if cached is not None:
        return tuple(cached)
    if not ollama.available():
        return None
    try:
        msg = ollama.chat([{"role": "user", "content": LLM_PROMPT.format(a=a, b=b)}],
                          fmt="json", options={"temperature": 0})
        d = json.loads(msg["content"])
        val = (max(0.0, min(1.0, float(d.get("score", 0)))),
               f"ИИ: «{a}» — {d.get('meaning_1', '?')}; «{b}» — {d.get('meaning_2', '?')}. {d.get('reason', '')}",
               bool(d.get("applicable", True)))
        db.cache_set(key, list(val))
        return val
    except Exception as e:  # модель не скачана, таймаут и т.п.
        log.warning("LLM semantic failed: %s", e)
        return None


def semantic_similarity(a: str, b: str, use_ai: bool = True) -> Criterion:
    parts: dict[str, float] = {}
    det: list[str] = []
    if phonetic_key(a) == phonetic_key(b):
        # одно и то же слово (в т.ч. транслитерация) — один и тот же смысл
        return Criterion(1.0, ["Одно и то же слово (возможно, в другой транслитерации)"],
                         {"same_word": 1.0})
    d, ddet = dictionary_similarity(a, b)
    parts["dictionary"] = d
    det += ddet
    applicable = bool(concepts_of(a)) and bool(concepts_of(b))
    if use_ai:
        e = embedding_similarity(a, b)
        if e is not None:
            parts["embeddings"] = e
        l = llm_similarity(a, b)
        if l is not None:
            parts["llm"] = l[0]
            det.append(l[1])
            applicable = applicable or (len(l) < 3 or l[2])
    if "llm" in parts:
        # LLM видит контекст лучше всего; словарь страхует от «галлюцинаций» в ноль
        score = max(parts["llm"], d * 0.9)
        if l is not None and (len(l) < 3 or l[2]):
            score = max(score, 0.5 * parts["llm"] + 0.5 * parts.get("embeddings", parts["llm"]))
    else:
        score = max(d, parts.get("embeddings", 0.0))
    if not use_ai or ("llm" not in parts and "embeddings" not in parts):
        det.append("Локальная ИИ-модель недоступна — смысл оценён только по встроенному словарю")
    if not applicable and score < 0.5:
        det.insert(0, "Смысловое сравнение неприменимо: обозначение фантазийное или его значение не установлено — критерий исключён из расчёта")
    parts["applicable"] = 1.0 if (applicable or score >= 0.5) else 0.0
    return Criterion(score, det, parts)
