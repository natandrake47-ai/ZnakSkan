"""Подбор весов по размеченным парам (решения Казпатента / апелляционного совета).

Каждая пара: два знака и метка «сходны до степени смешения» (1) или нет (0).
Обучается логистическая регрессия на признаках фонетика/смысл/графика,
положительные коэффициенты нормируются в веса, порог «высокой» степени
подбирается по максимальной точности (accuracy)."""
from __future__ import annotations

import json

import numpy as np

from . import db
from .scoring import Mark, compare


def _features(a: dict, b: dict) -> list[float] | None:
    r = compare(Mark(a.get("name", ""), a.get("classes") or []),
                Mark(b.get("name", ""), b.get("classes") or []), use_ai=True)
    w = r.get("word")
    if not w:
        return None
    return [w["phonetic"]["score"], w["semantic"]["score"], w["visual"]["score"]]


def calibrate(current: dict) -> dict:
    rows = db.conn().execute("SELECT a_json, b_json, confusing FROM labeled_pairs").fetchall()
    X, y = [], []
    for r in rows:
        f = _features(json.loads(r[0]), json.loads(r[1]))
        if f:
            X.append(f)
            y.append(r[2])
    if len(X) < 10 or len(set(y)) < 2:
        return {"ok": False, "message": f"Нужно минимум 10 размеченных пар обоих типов (сейчас {len(X)})."}
    X, y = np.array(X), np.array(y, dtype=float)
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(5000):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        g = p - y
        w -= 0.5 * (X.T @ g / len(y) + 0.01 * w)
        b -= 0.5 * g.mean()
    wp = np.clip(w, 0.01, None)
    wp = wp / wp.sum()
    new = json.loads(json.dumps(current))
    new["word"] = {"phonetic": round(float(wp[0]), 3), "semantic": round(float(wp[1]), 3),
                   "visual": round(float(wp[2]), 3)}
    s = X @ wp
    best_t, best_acc = 0.8, 0.0
    for t in np.linspace(0.3, 0.95, 66):
        acc = ((s >= t) == (y == 1)).mean()
        if acc > best_acc:
            best_t, best_acc = float(t), float(acc)
    new["thresholds"]["high"] = round(best_t, 3)
    return {"ok": True, "weights": new, "accuracy": round(best_acc, 3), "pairs": len(y),
            "message": f"Подобрано по {len(y)} парам, точность на них {best_acc:.0%}."}
