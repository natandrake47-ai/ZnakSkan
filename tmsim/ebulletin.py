"""Разбор официального электронного бюллетеня Казпатента (ebulletin.kazpatent.kz),
сохранённого из браузера (Ctrl+S → «Веб-страница»).

Берётся только раздел «Товарные знаки» (div id="tz"). Изобретения, полезные
модели, промышленные образцы, извещения и указатели пропускаются.

Запись знака в бюллетене:
  (111) номер регистрации  (151) дата регистрации  (181) срок действия
  (210) номер заявки       (220) дата подачи        (730) владелец
  (540) изображение знака (картинка; словесный элемент — только на картинке)
  (526) неохраноспособные элементы   (591) цвета
  (511)(510): список классов, далее перечень товаров по классам
"""
from __future__ import annotations

import base64
import html
import io
import re
from pathlib import Path

SECTION_ID = 'id="tz"'
DIV_RE = re.compile(r"<div[^>]*>([^<]*)</div>")
IMG_RE = re.compile(r'<img[^>]*>')
DATE_RE = re.compile(r"\d{2}\.\d{2}\.\d{4}")


def _text(s: str) -> str:
    s = re.sub(r"<br[^>]*>", "\n", s)
    s = re.sub(r"<[^>]+>", "", s)
    return html.unescape(s).strip()


def _section(doc: str) -> str:
    i = doc.find(SECTION_ID)
    if i < 0:
        # запасной вариант: по заголовку раздела (рус/каз)
        m = re.search(r'bulletin-section-title[^>]*>\s*(Товарные знаки|Тауар таңбалары)\s*\[', doc)
        if not m:
            return ""
        i = doc.rfind('<div', 0, m.start())
    j = doc.find('class="bulletin-section"', i + 10)
    return doc[i: j if j > 0 else len(doc)]


def publication(doc: str) -> str:
    m = re.search(r"(?:Опубликовано|Жарияланды)[^:]*:\s*(\d{4}-\d{2}-\d{2})\s*(?:<br[^>]*>)?\s*#?\s*(\d+)?", doc)
    if not m:
        return ""
    return f"Бюллетень {m.group(1)}" + (f" №{m.group(2)}" if m.group(2) else "")


def _load_image(src: str, base_dir: Path | None):
    from PIL import Image
    if src.startswith("data:"):
        data = base64.b64decode(src.split(",", 1)[1])
        return Image.open(io.BytesIO(data))
    if base_dir is not None:
        p = (base_dir / html.unescape(src)).resolve()
        if p.exists():
            return Image.open(p)
    return None


def _field(texts: list[str], code: str) -> str | None:
    for t in texts:
        if t.startswith(f"({code})"):
            return t[len(code) + 2:].strip()
    return None


def parse_entry(chunk: str, source: str, base_dir: Path | None) -> dict | None:
    head, _, rest = chunk.partition('class="tz-classes"')
    flat = IMG_RE.sub("", head)
    texts = [html.unescape(t).strip() for t in DIV_RE.findall(flat)]
    texts = [t for t in texts if t]
    reg = _field(texts, "111")
    app = _field(texts, "210")
    if not (reg or app):
        return None
    classes_line, goods = "", ""
    if rest:
        m = re.match(r"[^>]*>(.*?)</div>", rest, re.S)
        classes_line = _text(m.group(1)) if m else ""
        g = re.search(r'class="hiden"[^>]*>(.*?)</div>', rest, re.S)
        goods = _text(g.group(1)) if g else ""
    classes = sorted({int(x) for x in re.findall(r"\b\d{1,2}\b", classes_line.split(":", 1)[-1])
                      if 1 <= int(x) <= 45})
    goods = re.sub(r"^\(511\)\(510\)\s*", "", goods)
    goods = re.sub(r"[ \t]+", " ", goods)
    disclaimer = _field(texts, "526")
    if disclaimer:
        disclaimer = re.sub(r"^Неохраноспособные элементы:\s*", "", disclaimer)
    colors = _field(texts, "591")
    if colors:
        colors = re.sub(r"^Указание цветов:\s*", "", colors).rstrip(".")
    m = {
        "reg_number": reg, "app_number": app,
        "reg_date": (DATE_RE.search(_field(texts, "151") or "") or [None])[0],
        "expiry_date": (DATE_RE.search(_field(texts, "181") or "") or [None])[0],
        "app_date": (DATE_RE.search(_field(texts, "220") or "") or [None])[0],
        "owner": re.sub(r"\s+", " ", _field(texts, "730") or _field(texts, "732") or "").strip() or None,
        "priority": " ".join(filter(None, [_field(texts, c) for c in ("310", "320", "330")])) or None,
        "disclaimer": disclaimer, "colors": colors,
        "classes": classes, "goods": goods, "source": source, "name": "",
        "raw": "\n".join(texts),
    }
    t540 = _field(texts, "540")
    if t540:
        m["name"] = t540
    img_m = re.search(r'\(540\)\s*<img[^>]*src="([^"]+)"', head)
    if img_m:
        try:
            img = _load_image(img_m.group(1), base_dir)
            if img is not None:
                m["_image"] = img.convert("RGBA") if img.mode in ("P", "LA") else img
        except Exception:
            pass
    return m


def is_ebulletin(doc_head: str) -> bool:
    return "eBulletin" in doc_head or "bulletin-section" in doc_head


def parse_html(path: str | Path, progress=None) -> list[dict]:
    path = Path(path)
    doc = path.read_text("utf-8", errors="replace")
    src = publication(doc) or path.name
    sec = _section(doc)
    if not sec:
        return []
    chunks = sec.split('<div class="div-content"')[1:]
    out = []
    for k, ch in enumerate(chunks):
        if progress:
            progress(k + 1, len(chunks))
        m = parse_entry(ch, src, path.parent)
        if m:
            out.append(m)
    return out
