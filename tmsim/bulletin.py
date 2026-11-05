"""Парсер бюллетеней Казпатента (НИИС РК) с товарными знаками.

Записи в бюллетене размечены международными кодами ИНИД (ВОИС ST.60):
  (111) номер регистрации      (151) дата регистрации   (181) срок действия
  (210) номер заявки           (220) дата подачи        (310-330) приоритет
  (540) изображение/обозначение (591) цвета             (526) неохраняемые элементы
  (511) классы МКТУ и товары   (731)/(732) заявитель/владелец  (740) представитель
Парсер не привязан к вёрстке конкретного года: он собирает текст и картинки
страницы в порядке чтения (с учётом двух колонок) и режет поток по кодам.

Поддерживаемые входы: PDF, TXT (текст бюллетеня), CSV, JSON, XLSX.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import DATA_DIR

log = logging.getLogger(__name__)

INID_RE = re.compile(r"\((\d{3})\)")
ENTRY_START_CODES = ("111", "210")


@dataclass
class Item:
    page: int
    col: int
    top: float
    text: str = ""
    image: object = None   # PIL.Image


@dataclass
class Entry:
    fields: dict = field(default_factory=dict)
    order: list = field(default_factory=list)
    images: list = field(default_factory=list)
    saw_540: bool = False
    page: int = 0

    def add(self, code: str, text: str):
        if code in self.fields:
            self.fields[code] += "\n" + text
        else:
            self.fields[code] = text
            self.order.append(code)


# --------------------------------------------------------------------------
# Чтение PDF в поток элементов
# --------------------------------------------------------------------------

def _lines_from_words(words, tol=3.0):
    words = sorted(words, key=lambda w: (round(w["top"]), w["x0"]))
    lines = []
    for w in words:
        if lines and abs(lines[-1]["top"] - w["top"]) <= tol:
            lines[-1]["words"].append(w)
        else:
            lines.append({"top": w["top"], "words": [w]})
    out = []
    for ln in lines:
        ws = sorted(ln["words"], key=lambda w: w["x0"])
        out.append((ln["top"], " ".join(w["text"] for w in ws)))
    return out


def _detect_columns(words, width) -> list[tuple[float, float]]:
    """Возвращает границы колонок: одна или две."""
    if not words:
        return [(0, width)]
    mid = width / 2
    crossing = sum(1 for w in words if w["x0"] < mid - 2 and w["x1"] > mid + 2)
    left = sum(1 for w in words if w["x1"] <= mid)
    right = sum(1 for w in words if w["x0"] >= mid)
    if crossing <= max(2, 0.02 * len(words)) and left > 10 and right > 10:
        return [(0, mid), (mid, width)]
    return [(0, width)]


def pdf_items(path: str | Path, with_images: bool = True, progress=None) -> list[Item]:
    import pdfplumber
    items: list[Item] = []
    with pdfplumber.open(str(path)) as pdf:
        n = len(pdf.pages)
        for pno, page in enumerate(pdf.pages):
            if progress:
                progress(pno + 1, n)
            words = page.extract_words(keep_blank_chars=False, use_text_flow=False, x_tolerance=1.5)
            cols = _detect_columns(words, page.width)
            for ci, (x0, x1) in enumerate(cols):
                cw = [w for w in words if x0 <= (w["x0"] + w["x1"]) / 2 < x1]
                for top, text in _lines_from_words(cw):
                    items.append(Item(pno, ci, top, text=text))
                if with_images:
                    for im in page.images:
                        cx = (im["x0"] + im["x1"]) / 2
                        if not (x0 <= cx < x1):
                            continue
                        w_, h_ = im["x1"] - im["x0"], im["bottom"] - im["top"]
                        if w_ < 15 or h_ < 15:      # декоративные элементы
                            continue
                        if w_ > page.width * 0.9 and h_ > page.height * 0.9:  # скан страницы
                            continue
                        try:
                            bbox = (max(0, im["x0"]), max(0, im["top"]),
                                    min(page.width, im["x1"]), min(page.height, im["bottom"]))
                            pil = page.crop(bbox).to_image(resolution=150).original.convert("RGB")
                            items.append(Item(pno, ci, im["top"], image=pil))
                        except Exception as e:
                            log.warning("image extract failed p%d: %s", pno + 1, e)
    items.sort(key=lambda it: (it.page, it.col, it.top, 0 if it.image is None else 1))
    return items


def text_items(text: str) -> list[Item]:
    return [Item(0, 0, i, text=line) for i, line in enumerate(text.splitlines())]


# --------------------------------------------------------------------------
# Разбор потока на записи
# --------------------------------------------------------------------------

def split_entries(items: list[Item]) -> list[Entry]:
    all_text = "\n".join(it.text for it in items if it.text)
    present = set(INID_RE.findall(all_text))
    start = next((c for c in ENTRY_START_CODES if c in present), None)
    if not start:
        return []
    entries: list[Entry] = []
    cur: Entry | None = None
    code: str | None = None
    pending_images = []
    for it in items:
        if it.image is not None:
            if cur is not None and cur.saw_540 and not cur.images:
                cur.images.append(it.image)
            else:
                pending_images.append(it.image)
            continue
        line = it.text
        pos = 0
        matches = list(INID_RE.finditer(line))
        if not matches:
            if cur is not None and code:
                cur.add(code, line.strip())
            continue
        if matches[0].start() > 0 and cur is not None and code:
            pre = line[: matches[0].start()].strip()
            if pre:
                cur.add(code, pre)
        for i, m in enumerate(matches):
            c = m.group(1)
            end = matches[i + 1].start() if i + 1 < len(matches) else len(line)
            val = line[m.end():end].strip()
            if c == start:
                cur = Entry(page=it.page)
                entries.append(cur)
            if cur is None:
                continue
            code = c
            cur.add(c, val)
            if c == "540":
                cur.saw_540 = True
                if pending_images and not cur.images:
                    cur.images.append(pending_images.pop(0))
        pos = len(line)
    return entries


DATE_RE = re.compile(r"\d{2}[./]\d{2}[./]\d{4}|\d{4}[./-]\d{2}[./-]\d{2}")


def _first(s: str | None, rx=None) -> str | None:
    if not s:
        return None
    s = s.strip()
    if rx:
        m = rx.search(s)
        return m.group(0) if m else None
    return s.splitlines()[0].strip() if s else None


def parse_classes_goods(text: str | None) -> tuple[list[int], str]:
    if not text:
        return [], ""
    classes = set()
    for m in re.finditer(r"(?m)(?:^|;|\n)\s*(?:класс|кл\.?|сынып|class)?\s*(\d{1,2})\s*(?:[-–—:.]|\s+-|класс|сынып)", text, re.I):
        n = int(m.group(1))
        if 1 <= n <= 45:
            classes.add(n)
    first = text.strip().splitlines()[0] if text.strip() else ""
    if not classes and re.fullmatch(r"[\d\s,;.]+", first or "-"):
        classes |= {int(x) for x in re.findall(r"\d{1,2}", first) if 1 <= int(x) <= 45}
    goods = re.sub(r"\s+", " ", text).strip()
    return sorted(classes), goods


def clean_name(text: str | None) -> str:
    if not text:
        return ""
    t = re.sub(r"\s+", " ", text).strip().strip("«»\"'")
    # служебные пометки вместо текста
    if re.fullmatch(r"(?i)(изображение|см\.? изображение|бейне|рисунок|image|-|—)?", t):
        return ""
    return t[:300]


def entry_to_mark(e: Entry, source: str) -> dict:
    f = e.fields
    classes, goods = parse_classes_goods(f.get("511"))
    owner = f.get("732") or f.get("731") or ""
    m = {
        "reg_number": _first(f.get("111"), re.compile(r"[\d\-/]+")),
        "app_number": _first(f.get("210"), re.compile(r"[\w./\-]+")),
        "reg_date": _first(f.get("151"), DATE_RE),
        "expiry_date": _first(f.get("181"), DATE_RE),
        "app_date": _first(f.get("220"), DATE_RE),
        "priority": " ".join(filter(None, [f.get("310"), f.get("320"), f.get("330")])) or None,
        "name": clean_name(f.get("540")),
        "colors": _first(f.get("591")),
        "disclaimer": re.sub(r"\s+", " ", f.get("526", "")).strip() or None,
        "owner": re.sub(r"\s+", " ", owner).strip() or None,
        "classes": classes, "goods": goods, "source": source,
        "raw": json.dumps(f, ensure_ascii=False),
    }
    return m


def save_image(img, mark_key: str) -> str:
    safe = re.sub(r"[^\w.-]", "_", mark_key)[:80]
    p = DATA_DIR / "images" / f"{safe}.png"
    img.save(p)
    return str(p.relative_to(DATA_DIR))


def parse_pdf(path: str | Path, progress=None) -> list[dict]:
    items = pdf_items(path, progress=progress)
    entries = split_entries(items)
    src = Path(path).name
    out = []
    for e in entries:
        m = entry_to_mark(e, src)
        if e.images:
            m["_image"] = e.images[0]
        if m["reg_number"] or m["app_number"]:
            out.append(m)
    return out


def parse_text(text: str, source: str) -> list[dict]:
    return [entry_to_mark(e, source) for e in split_entries(text_items(text))]


# --------------------------------------------------------------------------
# Табличные форматы (выгрузки реестра, ручные списки)
# --------------------------------------------------------------------------

COLUMN_ALIASES = {
    "reg_number": ["reg_number", "номер регистрации", "рег. номер", "№ регистрации", "111", "номер свидетельства", "registration"],
    "app_number": ["app_number", "номер заявки", "заявка", "210", "application"],
    "name": ["name", "обозначение", "товарный знак", "наименование", "знак", "540", "mark"],
    "classes": ["classes", "классы", "класс", "мкту", "511", "class"],
    "goods": ["goods", "товары", "перечень товаров", "товары и услуги"],
    "owner": ["owner", "владелец", "правообладатель", "732", "заявитель"],
    "reg_date": ["reg_date", "дата регистрации", "151"],
    "app_date": ["app_date", "дата подачи", "220"],
    "expiry_date": ["expiry_date", "срок действия", "181"],
    "colors": ["colors", "цвета", "591"],
    "image_path": ["image_path", "изображение", "image", "файл изображения"],
}


def _map_row(row: dict, source: str) -> dict:
    low = {str(k).strip().lower(): v for k, v in row.items() if k is not None}
    m = {"source": source}
    for field_, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            if a in low and low[a] not in (None, ""):
                m[field_] = str(low[a]).strip()
                break
    return m


def parse_table(data: bytes, filename: str) -> list[dict]:
    ext = Path(filename).suffix.lower()
    rows: list[dict] = []
    if ext == ".json":
        d = json.loads(data.decode("utf-8-sig"))
        rows = d if isinstance(d, list) else d.get("marks", [])
    elif ext in (".csv", ".tsv"):
        text = data.decode("utf-8-sig", errors="replace")
        dialect = csv.Sniffer().sniff(text[:5000], delimiters=",;\t")
        rows = list(csv.DictReader(io.StringIO(text), dialect=dialect))
    elif ext in (".xlsx", ".xlsm"):
        import openpyxl  # опционально
        wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True)
        ws = wb.active
        it = ws.iter_rows(values_only=True)
        head = [str(h or "") for h in next(it)]
        rows = [dict(zip(head, r)) for r in it]
    return [_map_row(r, filename) for r in rows]
