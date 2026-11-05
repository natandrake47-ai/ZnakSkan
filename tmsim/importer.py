"""Импорт бюллетеней и таблиц в базу: извлечение, признаки изображений,
распознавание словесного элемента с картинки знака.

В электронном бюллетене Казпатента обозначение (540) дано только картинкой,
поэтому название знака нужно распознать. Порядок:
  1. готовые названия из папки seed/ (по номеру регистрации);
  2. локальная vision-модель Ollama (qwen2.5vl и т.п.) — читает и кириллицу,
     и латиницу, и описывает изображение;
  3. Tesseract OCR, если установлен (pytesseract + tesseract с языками rus, kaz, eng).
Нераспознанные знаки можно подписать вручную на вкладке «База знаков»."""
from __future__ import annotations

import base64
import io
import json
import logging
import threading
import uuid
from functools import lru_cache
from pathlib import Path

from . import bulletin, db, ebulletin, image_sim, ollama
from .config import DATA_DIR, ROOT, settings

log = logging.getLogger(__name__)

JOBS: dict[str, dict] = {}

OCR_PROMPT = (
    "Это изображение товарного знака из официального бюллетеня. Задачи:\n"
    "1) Выпиши словесные элементы (слова, буквы, цифры) ровно так, как они написаны, с сохранением алфавита "
    "(кириллица/латиница/казахские буквы). Главное (самое крупное) слово — первым, мелкие подписи — после.\n"
    "2) Кратко по-русски опиши изобразительные элементы (что нарисовано), без цветов.\n"
    "Если слов нет, text — пустая строка. Ответ строго JSON: "
    "{\"text\": \"...\", \"main\": \"главное слово\", \"description\": \"...\"}")


@lru_cache(maxsize=1)
def seed_names() -> dict:
    out = {}
    for p in sorted((ROOT / "seed").glob("*.json")):
        try:
            out.update(json.loads(p.read_text("utf-8")).get("marks", {}))
        except Exception as e:
            log.warning("seed %s: %s", p, e)
    return out


def _vision(img) -> tuple[str, str] | None:
    s = settings()
    model = s.get("vision_model")
    if not model or model not in ollama.models():
        return None
    buf = io.BytesIO()
    im = img.convert("RGB")
    im.thumbnail((768, 768))
    im.save(buf, format="PNG")
    try:
        msg = ollama.chat([{"role": "user", "content": OCR_PROMPT,
                            "images": [base64.b64encode(buf.getvalue()).decode()]}],
                          model=model, fmt="json", options={"temperature": 0})
        d = json.loads(msg["content"])
        text = (d.get("text") or d.get("main") or "").strip()
        return text, (d.get("description") or "").strip()
    except Exception as e:
        log.warning("vision OCR failed: %s", e)
        return None


def _tesseract(img) -> tuple[str, str] | None:
    try:
        import pytesseract
        txt = pytesseract.image_to_string(img.convert("L"), lang="rus+kaz+eng", config="--psm 6")
    except Exception:
        return None
    txt = " ".join(txt.split())
    return (txt, "") if txt else None


def ocr_image(img) -> tuple[str, str] | None:
    return _vision(img) or _tesseract(img)


def ocr_available() -> dict:
    s = settings()
    tess = False
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        tess = True
    except Exception:
        pass
    return {"vision": bool(s.get("vision_model")) and s.get("vision_model") in ollama.models(),
            "tesseract": tess}


def store_marks(marks: list[dict], use_ocr: bool = True, job: dict | None = None) -> int:
    seeds = seed_names()
    n = 0
    for i, m in enumerate(marks):
        img = m.pop("_image", None)
        if img is None and m.get("image_path"):
            p = Path(m["image_path"])
            p = p if p.is_absolute() else DATA_DIR / p
            if p.exists():
                img = image_sim.load(str(p))
        sd = seeds.get(m.get("reg_number") or "")
        if sd and not m.get("name"):
            m["name"] = sd.get("name", "")
            m["image_desc"] = sd.get("image_desc") or None
        if img is not None:
            key = m.get("reg_number") or m.get("app_number") or uuid.uuid4().hex
            if img.mode not in ("RGB", "L"):
                img = image_sim.load_pil(img)
            m["image_path"] = bulletin.save_image(img, key)
            feat, clip = image_sim.features(img)
            m["image_feat"], m["image_clip"] = feat, clip
            if use_ocr and not m.get("name"):
                if job is not None:
                    job["stage"] = "Распознавание названий на изображениях"
                r = ocr_image(img)
                if r:
                    m["name"], m["image_desc"] = r[0], (r[1] or None)
        db.upsert_mark(m)
        n += 1
        if job is not None:
            job.update(done=i + 1, total=len(marks))
    return n


def _run(job: dict, fn):
    def run():
        try:
            fn()
            job.update(status="done", stage="Готово")
        except Exception as e:
            log.exception("job failed")
            job.update(status="error", stage=f"Ошибка: {e}")
    threading.Thread(target=run, daemon=True).start()


def _new_job(name: str) -> dict:
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"id": job_id, "file": name, "status": "running", "stage": "Чтение файла",
                    "done": 0, "total": 0, "count": 0, "log": []}
    return JOBS[job_id]


def import_file(path: Path, filename: str, use_ocr: bool = True) -> str:
    job = _new_job(filename)

    def work():
        ext = path.suffix.lower()

        def prog(i, n):
            job.update(done=i, total=n)
        if ext in (".htm", ".html", ".mhtml"):
            job["stage"] = "Разбор раздела «Товарные знаки»"
            marks = ebulletin.parse_html(path, progress=prog)
            if marks:
                job["log"].append(f"{marks[0]['source']}: товарных знаков {len(marks)} "
                                  "(остальные разделы бюллетеня пропущены)")
        elif ext == ".pdf":
            job["stage"] = "Разбор страниц PDF"
            marks = bulletin.parse_pdf(path, progress=prog)
        elif ext == ".txt":
            marks = bulletin.parse_text(path.read_text("utf-8", errors="replace"), filename)
        else:
            marks = bulletin.parse_table(path.read_bytes(), filename)
        if not ext.startswith(".htm"):
            job["log"].append(f"Найдено записей: {len(marks)}")
        if not marks:
            job["log"].append("Товарные знаки не найдены. Если это бюллетень Казпатента, сохраните страницу "
                              "выпуска целиком (Ctrl+S) и загрузите файл .htm.")
        job["stage"] = "Сохранение в базу"
        n = store_marks(marks, use_ocr=use_ocr, job=job)
        unnamed = sum(1 for m in marks if not m.get("name") and not m.get("image_desc"))
        if unnamed:
            av = ocr_available()
            hint = "" if (av["vision"] or av["tesseract"]) else (
                " Модель распознавания не установлена: выполните «ollama pull "
                + (settings().get("vision_model") or "qwen2.5vl:7b") + "» и нажмите «Распознать названия».")
            job["log"].append(f"Без распознанного названия: {unnamed}.{hint}")
        with db.tx() as c:
            c.execute("INSERT OR REPLACE INTO bulletins (filename, marks_count, log) VALUES (?,?,?)",
                      (marks[0]["source"] if marks else filename, n, "\n".join(job["log"])))
        job["count"] = n

    _run(job, work)
    return job["id"]


def reocr_unnamed() -> str:
    """Повторно распознать названия у знаков, где их нет (например, после установки модели)."""
    job = _new_job("Распознавание названий")

    def work():
        rows = db.conn().execute("SELECT id, image_path FROM marks WHERE IFNULL(name,'')='' "
                                 "AND IFNULL(image_desc,'')='' AND image_path IS NOT NULL").fetchall()
        job.update(total=len(rows), stage="Распознавание названий на изображениях")
        if rows and not any(ocr_available().values()):
            job["log"].append("Нет модели распознавания: установите vision-модель в Ollama.")
            return
        ok = 0
        for i, r in enumerate(rows):
            res = ocr_image(image_sim.load(str(DATA_DIR / r["image_path"])))
            if res and res[0]:
                db.update_name(r["id"], res[0], res[1] or None)
                ok += 1
            job.update(done=i + 1, count=ok)
        job["log"].append(f"Распознано: {ok} из {len(rows)}")

    _run(job, work)
    return job["id"]
