"""Локальная база зарегистрированных товарных знаков (SQLite)."""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager

from .config import DATA_DIR
from .translit import clean, phonetic_key

DB_PATH = DATA_DIR / "marks.sqlite3"
_local = threading.local()

SCHEMA = """
CREATE TABLE IF NOT EXISTS marks (
    id INTEGER PRIMARY KEY,
    reg_number TEXT,            -- (111) номер регистрации
    app_number TEXT,            -- (210) номер заявки
    app_date TEXT,              -- (220) дата подачи
    reg_date TEXT,              -- (151) дата регистрации
    expiry_date TEXT,           -- (181) срок действия
    priority TEXT,              -- (310)-(330) приоритет
    name TEXT,                  -- словесный элемент (540)
    name_clean TEXT,
    phon TEXT,                  -- строгий фонетический ключ
    phon_loose TEXT,            -- нестрогий фонетический ключ
    owner TEXT,                 -- (732)/(731) владелец
    classes TEXT,               -- "3,5,35"
    goods TEXT,                 -- (511) перечень товаров/услуг
    colors TEXT,                -- (591) цветовое сочетание
    disclaimer TEXT,            -- (526) неохраняемые элементы
    kind TEXT,                  -- word / figurative / combined
    image_path TEXT,
    image_desc TEXT,            -- что изображено (описание ИИ)
    image_feat TEXT,            -- JSON признаков изображения
    image_clip BLOB,            -- эмбеддинг CLIP (float32)
    status TEXT DEFAULT 'registered',
    source TEXT,                -- бюллетень / файл импорта
    raw TEXT,
    UNIQUE(reg_number, app_number)
);
CREATE TABLE IF NOT EXISTS mark_classes (
    mark_id INTEGER REFERENCES marks(id) ON DELETE CASCADE,
    cls INTEGER,
    PRIMARY KEY (mark_id, cls)
);
CREATE INDEX IF NOT EXISTS ix_mark_classes_cls ON mark_classes(cls);
CREATE INDEX IF NOT EXISTS ix_marks_phon ON marks(phon_loose);
CREATE TABLE IF NOT EXISTS bulletins (
    id INTEGER PRIMARY KEY,
    filename TEXT UNIQUE,
    imported_at TEXT DEFAULT CURRENT_TIMESTAMP,
    marks_count INTEGER,
    log TEXT
);
CREATE TABLE IF NOT EXISTS cache (
    k TEXT PRIMARY KEY,
    v TEXT
);
CREATE TABLE IF NOT EXISTS labeled_pairs (
    id INTEGER PRIMARY KEY,
    a_json TEXT, b_json TEXT,
    confusing INTEGER,          -- 1 = сходны до степени смешения, 0 = нет
    note TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


def conn() -> sqlite3.Connection:
    c = getattr(_local, "conn", None)
    if c is None:
        c = sqlite3.connect(DB_PATH, check_same_thread=False)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys=ON")
        c.execute("PRAGMA journal_mode=WAL")
        c.executescript(SCHEMA)
        cols = {r[1] for r in c.execute("PRAGMA table_info(marks)")}
        if "image_desc" not in cols:  # миграция баз первой версии
            c.execute("ALTER TABLE marks ADD COLUMN image_desc TEXT")
        _local.conn = c
    return c


@contextmanager
def tx():
    c = conn()
    try:
        yield c
        c.commit()
    except Exception:
        c.rollback()
        raise


def parse_classes(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items = value
    else:
        import re
        items = re.findall(r"\d{1,2}", str(value))
    out = sorted({int(x) for x in items if 1 <= int(x) <= 45})
    return out


def upsert_mark(m: dict) -> int:
    name = m.get("name") or ""
    classes = parse_classes(m.get("classes"))
    row = {
        "reg_number": m.get("reg_number"), "app_number": m.get("app_number"),
        "app_date": m.get("app_date"), "reg_date": m.get("reg_date"),
        "expiry_date": m.get("expiry_date"), "priority": m.get("priority"),
        "name": name, "name_clean": clean(name), "phon": phonetic_key(name),
        "phon_loose": phonetic_key(name, True), "owner": m.get("owner"),
        "classes": ",".join(map(str, classes)), "goods": m.get("goods"),
        "colors": m.get("colors"), "disclaimer": m.get("disclaimer"),
        "kind": m.get("kind") or ("combined" if name and m.get("image_path") else
                                  "figurative" if m.get("image_path") else "word"),
        "image_path": m.get("image_path"), "image_desc": m.get("image_desc"),
        "image_feat": json.dumps(m["image_feat"]) if m.get("image_feat") else None,
        "image_clip": m.get("image_clip"), "status": m.get("status") or "registered",
        "source": m.get("source"), "raw": m.get("raw"),
    }
    with tx() as c:
        cur = c.execute(
            "SELECT id FROM marks WHERE IFNULL(reg_number,'')=IFNULL(?,'') AND IFNULL(app_number,'')=IFNULL(?,'')",
            (row["reg_number"], row["app_number"]))
        ex = cur.fetchone()
        cols = list(row)
        if ex and (row["reg_number"] or row["app_number"]):
            mid = ex["id"]
            c.execute(f"UPDATE marks SET {', '.join(f'{k}=?' for k in cols)} WHERE id=?",
                      [row[k] for k in cols] + [mid])
            c.execute("DELETE FROM mark_classes WHERE mark_id=?", (mid,))
        else:
            cur = c.execute(f"INSERT INTO marks ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                            [row[k] for k in cols])
            mid = cur.lastrowid
        c.executemany("INSERT OR IGNORE INTO mark_classes VALUES (?, ?)", [(mid, k) for k in classes])
    return mid


def row_to_dict(r: sqlite3.Row) -> dict:
    d = dict(r)
    d.pop("image_clip", None)
    d.pop("raw", None)
    if d.get("image_feat"):
        d["image_feat"] = json.loads(d["image_feat"])
    d["classes"] = parse_classes(d.get("classes"))
    return d


def get_mark(mark_id: int) -> dict | None:
    r = conn().execute("SELECT * FROM marks WHERE id=?", (mark_id,)).fetchone()
    return row_to_dict(r) if r else None


def find_by_number(num: str) -> dict | None:
    r = conn().execute("SELECT * FROM marks WHERE reg_number=? OR app_number=?",
                       (num, num)).fetchone()
    return row_to_dict(r) if r else None


def iter_candidates(classes: list[int] | None):
    """Строки-кандидаты для сравнения; при заданных классах — только из этих
    классов (список уже расширен смежными классами вызывающим кодом)."""
    c = conn()
    if classes:
        q = ("SELECT DISTINCT m.* FROM marks m JOIN mark_classes mc ON mc.mark_id=m.id "
             f"WHERE mc.cls IN ({','.join('?' * len(classes))})")
        return c.execute(q, classes)
    return c.execute("SELECT * FROM marks")


def list_marks(q: str = "", cls: int | None = None, offset: int = 0, limit: int = 50,
               no_name: bool = False):
    c = conn()
    where, args = [], []
    if no_name:
        where.append("IFNULL(name,'')=''")
    if q:
        where.append("(name_clean LIKE ? OR reg_number LIKE ? OR app_number LIKE ? OR owner LIKE ?)")
        like = f"%{q.lower()}%"
        args += [like, f"%{q}%", f"%{q}%", f"%{q}%"]
    if cls:
        where.append("id IN (SELECT mark_id FROM mark_classes WHERE cls=?)")
        args.append(cls)
    w = ("WHERE " + " AND ".join(where)) if where else ""
    total = c.execute(f"SELECT COUNT(*) FROM marks {w}", args).fetchone()[0]
    rows = c.execute(f"SELECT * FROM marks {w} ORDER BY id DESC LIMIT ? OFFSET ?",
                     args + [limit, offset]).fetchall()
    return total, [row_to_dict(r) for r in rows]


def stats() -> dict:
    c = conn()
    return {
        "marks": c.execute("SELECT COUNT(*) FROM marks").fetchone()[0],
        "with_images": c.execute("SELECT COUNT(*) FROM marks WHERE image_path IS NOT NULL").fetchone()[0],
        "without_name": c.execute("SELECT COUNT(*) FROM marks WHERE IFNULL(name,'')='' "
                                  "AND IFNULL(image_desc,'')=''").fetchone()[0],
        "bulletins": c.execute("SELECT COUNT(*) FROM bulletins").fetchone()[0],
        "by_class": {r[0]: r[1] for r in c.execute(
            "SELECT cls, COUNT(*) FROM mark_classes GROUP BY cls ORDER BY cls")},
        "labeled_pairs": c.execute("SELECT COUNT(*) FROM labeled_pairs").fetchone()[0],
    }


def cache_get(k: str):
    r = conn().execute("SELECT v FROM cache WHERE k=?", (k,)).fetchone()
    return json.loads(r[0]) if r else None


def cache_set(k: str, v) -> None:
    with tx() as c:
        c.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (k, json.dumps(v, ensure_ascii=False)))


def update_name(mark_id: int, name: str, image_desc: str | None = None) -> None:
    with tx() as c:
        c.execute("UPDATE marks SET name=?, name_clean=?, phon=?, phon_loose=?, "
                  "image_desc=COALESCE(?, image_desc), kind=CASE WHEN image_path IS NOT NULL AND ?<>'' "
                  "THEN 'combined' ELSE kind END WHERE id=?",
                  (name, clean(name), phonetic_key(name), phonetic_key(name, True), image_desc, name, mark_id))
