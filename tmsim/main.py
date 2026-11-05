"""Веб-сервер программы (FastAPI). Запуск: python -m tmsim"""
from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__, absolute, agent, db, image_sim, importer, ollama
from .calibrate import calibrate
from .config import DATA_DIR, save_settings, save_weights, settings, weights
from .nice import NICE
from .scoring import Mark, compare
from .search import search

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = FastAPI(title="TM-Sim", version=__version__)
STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/images/{path:path}")
def images(path: str):
    p = (DATA_DIR / path).resolve()
    if not str(p).startswith(str(DATA_DIR.resolve())) or not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/api/status")
def status():
    s = settings()
    av = ollama.available()
    ms = ollama.models() if av else []
    return {"version": __version__, "ollama": av, "models": ms,
            "chat_model_ready": s["chat_model"] in ms,
            "vision_model_ready": s.get("vision_model") in ms,
            "embed_model_ready": any(m.split(":")[0] == s.get("embed_model", "").split(":")[0] for m in ms),
            "clip": image_sim._clip_model() is not None, "ocr": importer.ocr_available(),
            "db": db.stats(), "settings": s}


async def _mark(name: str, classes: str, goods: str, image: UploadFile | None, label: str) -> Mark:
    feat = clip = None
    if image is not None and image.filename:
        data = await image.read()
        if data:
            try:
                feat, clip = image_sim.features(data)
            except Exception as e:
                raise HTTPException(400, f"Не удалось прочитать изображение «{image.filename}»: {e}")
    return Mark(name=name.strip(), classes=db.parse_classes(classes), goods=goods or None,
                image_feat=feat, image_clip=clip, label=label)


@app.post("/api/compare")
async def api_compare(name_a: str = Form(""), classes_a: str = Form(""), goods_a: str = Form(""),
                      image_a: UploadFile | None = File(None),
                      name_b: str = Form(""), classes_b: str = Form(""), goods_b: str = Form(""),
                      image_b: UploadFile | None = File(None), use_ai: bool = Form(True)):
    a = await _mark(name_a, classes_a, goods_a, image_a, name_a or "Знак 1")
    b = await _mark(name_b, classes_b, goods_b, image_b, name_b or "Знак 2")
    if not (a.has_word or a.has_image) or not (b.has_word or b.has_image):
        raise HTTPException(400, "Для каждого знака укажите обозначение или загрузите изображение")
    r = compare(a, b, use_ai=use_ai)
    r["absolute_a"] = absolute.check(a.name, a.classes) if a.has_word else []
    r["absolute_b"] = absolute.check(b.name, b.classes) if b.has_word else []
    r["input"] = {"a": {"name": a.name, "classes": a.classes}, "b": {"name": b.name, "classes": b.classes}}
    return r


@app.post("/api/search")
async def api_search(name: str = Form(""), classes: str = Form(""), goods: str = Form(""),
                     image: UploadFile | None = File(None), related: bool = Form(True),
                     use_ai: bool = Form(False), limit: int = Form(50)):
    q = await _mark(name, classes, goods, image, name or "Заявляемое обозначение")
    if not (q.has_word or q.has_image):
        raise HTTPException(400, "Укажите обозначение или загрузите изображение")
    res = search(q, include_related=related, use_ai=use_ai, limit=limit)
    return {"results": res, "absolute": absolute.check(q.name, q.classes) if q.has_word else [],
            "total_in_db": db.stats()["marks"]}


@app.post("/api/import")
async def api_import(file: UploadFile = File(...), use_ocr: bool = Form(True)):
    suffix = Path(file.filename or "x").suffix.lower()
    if suffix not in (".htm", ".html", ".pdf", ".txt", ".csv", ".tsv", ".json", ".xlsx"):
        raise HTTPException(400, "Поддерживаются: сохранённая страница бюллетеня (.htm), PDF, TXT, CSV, JSON, XLSX")
    dest = DATA_DIR / "bulletins" / Path(file.filename).name
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"job": importer.import_file(dest, file.filename, use_ocr=use_ocr)}


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    j = importer.JOBS.get(job_id)
    if not j:
        raise HTTPException(404)
    return j


@app.get("/api/bulletins")
def api_bulletins():
    return [dict(r) for r in db.conn().execute("SELECT * FROM bulletins ORDER BY id DESC")]


@app.get("/api/marks")
def api_marks(q: str = "", cls: int | None = None, offset: int = 0, limit: int = 50,
              no_name: bool = False):
    total, rows = db.list_marks(q, cls, offset, limit, no_name)
    for r in rows:
        r.pop("image_feat", None)
    return {"total": total, "items": rows}


@app.post("/api/marks")
async def api_add_mark(name: str = Form(""), classes: str = Form(""), goods: str = Form(""),
                       reg_number: str = Form(""), app_number: str = Form(""), owner: str = Form(""),
                       image: UploadFile | None = File(None)):
    m = {"name": name, "classes": classes, "goods": goods, "reg_number": reg_number or None,
         "app_number": app_number or None, "owner": owner, "source": "вручную"}
    if image is not None and image.filename:
        m["_image"] = image_sim.load(await image.read())
    importer.store_marks([m], use_ocr=not name)
    return {"ok": True}


@app.patch("/api/marks/{mark_id}")
async def api_patch_mark(mark_id: int, data: dict):
    if db.get_mark(mark_id) is None:
        raise HTTPException(404)
    db.update_name(mark_id, (data.get("name") or "").strip(), data.get("image_desc"))
    return db.get_mark(mark_id)


@app.post("/api/ocr")
def api_ocr():
    return {"job": importer.reocr_unnamed()}


@app.delete("/api/marks/{mark_id}")
def api_del_mark(mark_id: int):
    with db.tx() as c:
        c.execute("DELETE FROM marks WHERE id=?", (mark_id,))
    return {"ok": True}


@app.get("/api/weights")
def api_weights():
    return weights()


@app.put("/api/weights")
async def api_put_weights(data: dict):
    cur = weights()
    for k, v in data.items():
        if k in cur:
            cur[k] = v
    save_weights(cur)
    return cur


@app.post("/api/weights/reset")
def api_reset_weights():
    src = Path(__file__).resolve().parent.parent / "config" / "weights.default.json"
    data = json.loads(src.read_text("utf-8"))
    save_weights(data)
    return data


@app.put("/api/settings")
async def api_put_settings(data: dict):
    allowed = set(settings())
    save_settings({k: v for k, v in data.items() if k in allowed})
    image_sim._clip_model.cache_clear()
    return settings()


@app.post("/api/chat")
async def api_chat(data: dict):
    return agent.chat(data.get("history", []), data.get("context"))


@app.get("/api/nice")
def api_nice():
    return NICE


@app.post("/api/labeled")
async def api_labeled(data: dict):
    with db.tx() as c:
        c.execute("INSERT INTO labeled_pairs (a_json, b_json, confusing, note) VALUES (?,?,?,?)",
                  (json.dumps(data["a"], ensure_ascii=False), json.dumps(data["b"], ensure_ascii=False),
                   1 if data.get("confusing") else 0, data.get("note", "")))
    return {"ok": True, "count": db.stats()["labeled_pairs"]}


@app.post("/api/calibrate")
def api_calibrate(apply: bool = False):
    r = calibrate(weights())
    if r.get("ok") and apply:
        save_weights(r["weights"])
    return JSONResponse(r)
