import io
import os
import tempfile
import time
from pathlib import Path

os.environ["TMSIM_DATA"] = tempfile.mkdtemp()

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from tmsim.main import app  # noqa: E402
from tmsim.scoring import Mark, compare  # noqa: E402

c = TestClient(app)
HERE = Path(__file__).parent


def risk(a, b, ca=(30,), cb=(30,)):
    return compare(Mark(a, list(ca)), Mark(b, list(cb)), use_ai=False)


def test_examples_from_task():
    assert risk("Тойказан", "Тайказан")["level"] == "high"
    assert risk("Звёздочка", "Звезда")["word"]["identity"]
    assert risk("В лесу", "Лес")["word"]["identity"]
    assert risk("SuperFood", "Food Master")["mark_similarity"] > risk("SuperFood", "SuperMax")["mark_similarity"]


def test_transliteration_and_translation():
    assert risk("Altyn", "Алтын")["word"]["phonetic"]["score"] > 0.95
    assert risk("Golden Lion", "Алтын Арыстан")["word"]["semantic"]["score"] >= 0.75
    assert risk("Coca-Cola", "Pepsi")["level"] == "none"


def test_classes_reduce_risk():
    same = risk("Samsung", "Samsong", [9], [9])["risk"]
    other = risk("Samsung", "Samsong", [9], [31])["risk"]
    assert other < same


def _png(color, shape):
    im = Image.new("RGB", (200, 200), "white")
    d = ImageDraw.Draw(im)
    (d.ellipse if shape == "circle" else d.rectangle)((30, 30, 170, 170), fill=color)
    b = io.BytesIO(); im.save(b, "PNG"); return b.getvalue()


def test_compare_endpoint_with_images():
    r = c.post("/api/compare", data={"name_a": "Арыстан", "classes_a": "33", "name_b": "Arystan",
                                     "classes_b": "33", "use_ai": "false"},
               files={"image_a": ("a.png", _png("red", "circle")), "image_b": ("b.png", _png("red", "circle"))})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["image"]["score"] > 0.9 and j["level"] == "high"


def test_import_bulletin_and_search():
    with open(HERE / "sample_bulletin.pdf", "rb") as f:
        r = c.post("/api/import", files={"file": ("sample_bulletin.pdf", f)}, data={"use_ocr": "false"})
    job = r.json()["job"]
    for _ in range(100):
        j = c.get(f"/api/jobs/{job}").json()
        if j["status"] != "running":
            break
        time.sleep(0.1)
    assert j["status"] == "done" and j["count"] == 4, j
    r = c.post("/api/search", data={"name": "Тайказан", "classes": "30"}).json()
    assert r["results"][0]["mark"]["reg_number"] == "70001"
    assert r["results"][0]["level"] == "high"
    r = c.post("/api/search", data={"name": "Samsung", "classes": "9"}).json()
    assert r["results"][0]["mark"]["name"] == "SAMSONG"
    r = c.post("/api/search", data={"name": "", "classes": "32"},
               files={"image": ("x.png", _png("red", "circle"))}).json()
    assert r["results"][0]["mark"]["reg_number"] == "70002"
    marks = c.get("/api/marks").json()
    assert marks["total"] == 4


def test_chat_without_ollama():
    r = c.post("/api/chat", json={"history": [{"role": "user", "content": "привет"}]}).json()
    assert r.get("error") == "ollama_unavailable"


def test_index():
    assert "TM-Sim" in c.get("/").text
    assert c.get("/api/status").status_code == 200


def test_ebulletin_html_only_trademarks():
    from tmsim.ebulletin import parse_html
    ms = parse_html(HERE / "data" / "ebulletin_sample.htm")
    assert [m["reg_number"] for m in ms] == ["109485", "109486", "109487"]
    m = ms[0]
    assert m["source"] == "Бюллетень 2026-01-09 №1"
    assert m["classes"] == [30, 35, 39] and m["app_number"] == "137083"
    assert m["owner"].startswith("Товарищество") and "_image" in m
    assert m["disclaimer"].startswith("«АДЖИКА»")


def test_ebulletin_import_uses_seed_names():
    with open(HERE / "data" / "ebulletin_sample.htm", "rb") as f:
        job = c.post("/api/import", files={"file": ("eb.htm", f)}, data={"use_ocr": "false"}).json()["job"]
    for _ in range(100):
        j = c.get(f"/api/jobs/{job}").json()
        if j["status"] != "running":
            break
        time.sleep(0.1)
    assert j["status"] == "done" and j["count"] == 3, j
    r = c.post("/api/search", data={"name": "Kobra", "classes": "30"}).json()
    assert r["results"][0]["mark"]["name"] == "Кобра" and r["results"][0]["level"] == "high"
