"""Поиск сходных знаков в локальной базе."""
from __future__ import annotations

import json

from . import db, image_sim
from .config import settings, weights
from .nice import expand_classes
from .scoring import Mark, compare
from .text_sim import common_prefix, lev_ratio
from .translit import phonetic_key


def mark_from_row(r) -> Mark:
    feat = json.loads(r["image_feat"]) if r["image_feat"] else None
    return Mark(name=r["name"] or "", classes=db.parse_classes(r["classes"]),
                goods=r["goods"], image_feat=feat, image_clip=r["image_clip"],
                image_desc=r["image_desc"],
                label=r["reg_number"] or r["app_number"] or str(r["id"]),
                meta={"id": r["id"], "reg_number": r["reg_number"], "app_number": r["app_number"],
                      "owner": r["owner"], "image_path": r["image_path"], "name": r["name"],
                      "image_desc": r["image_desc"], "source": r["source"],
                      "classes": db.parse_classes(r["classes"]), "reg_date": r["reg_date"],
                      "expiry_date": r["expiry_date"]})


def _cheap_score(q: Mark, qkey: str, r) -> float:
    s = 0.0
    if q.has_word and r["phon_loose"]:
        k = r["phon_loose"].replace(" ", "")
        s = lev_ratio(qkey, k)
        if common_prefix(qkey, k) >= 3:
            s += 0.15
        if len(qkey) >= 3 and (qkey in k or (len(k) >= 3 and k in qkey)):
            s += 0.3
        # совпадение отдельных слов
        qw = set(phonetic_key(q.name, True).split())
        if qw & set(r["phon_loose"].split()):
            s += 0.4
    if q.has_image and r["image_feat"]:
        f = json.loads(r["image_feat"])
        ph = 1 - bin(int(q.image_feat["phash"], 16) ^ int(f["phash"], 16)).count("1") / 64
        s = max(s, ph * 1.1)
        if q.image_clip and r["image_clip"]:
            import numpy as np
            c = float(np.frombuffer(q.image_clip, np.float32) @ np.frombuffer(r["image_clip"], np.float32))
            s = max(s, c * 1.2)
    return s


def search(q: Mark, include_related: bool = True, use_ai: bool = False,
           limit: int | None = None) -> list[dict]:
    st = settings()
    limit = limit or st.get("results_limit", 50)
    classes = expand_classes(q.classes, include_related) if q.classes else None
    qkey = phonetic_key(q.name, True).replace(" ", "")
    scored = []
    for r in db.iter_candidates(classes):
        scored.append((_cheap_score(q, qkey, r), r))
    scored.sort(key=lambda t: -t[0])
    top = scored[: st.get("candidates_prefilter", 300)]
    w = weights()
    results = []
    for _, r in top:
        m = mark_from_row(r)
        res = compare(q, m, use_ai=False, w=w)
        res["mark"] = m.meta
        results.append(res)
    results.sort(key=lambda x: -x["risk"])
    # несходные знаки не показываем, кроме первых 10 для ориентира
    low = w["thresholds"]["low"]
    results = [r for i, r in enumerate(results) if r["risk"] >= low or i < 10][:limit]
    if use_ai:
        # уточняем смысловое сходство моделью только для лучших кандидатов
        for res in results[:10]:
            m = mark_from_row(db.conn().execute("SELECT * FROM marks WHERE id=?",
                                                (res["mark"]["id"],)).fetchone())
            meta = res["mark"]
            res.clear()
            res.update(compare(q, m, use_ai=True, w=w))
            res["mark"] = meta
        results.sort(key=lambda x: -x["risk"])
    return results
