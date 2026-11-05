"""Минимальный клиент локального LLM-сервера Ollama (https://ollama.com)."""
from __future__ import annotations

import json
import logging

import httpx

from .config import settings

log = logging.getLogger(__name__)


def _url(path: str) -> str:
    return settings()["ollama_url"].rstrip("/") + path


def available() -> bool:
    try:
        return httpx.get(_url("/api/tags"), timeout=2).status_code == 200
    except Exception:
        return False


def models() -> list[str]:
    try:
        r = httpx.get(_url("/api/tags"), timeout=3)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def chat(messages: list[dict], tools: list[dict] | None = None,
         model: str | None = None, fmt: str | dict | None = None,
         options: dict | None = None) -> dict:
    s = settings()
    payload = {"model": model or s["chat_model"], "messages": messages,
               "stream": False, "options": options or {"temperature": 0.2}}
    if tools:
        payload["tools"] = tools
    if fmt:
        payload["format"] = fmt
    r = httpx.post(_url("/api/chat"), json=payload, timeout=s.get("llm_timeout_sec", 120))
    r.raise_for_status()
    return r.json()["message"]


def chat_stream(messages: list[dict], model: str | None = None):
    s = settings()
    payload = {"model": model or s["chat_model"], "messages": messages,
               "stream": True, "options": {"temperature": 0.3}}
    with httpx.stream("POST", _url("/api/chat"), json=payload,
                      timeout=s.get("llm_timeout_sec", 120)) as r:
        for line in r.iter_lines():
            if line:
                d = json.loads(line)
                yield d.get("message", {}).get("content", "")


def embed(texts: list[str], model: str | None = None) -> list[list[float]] | None:
    s = settings()
    try:
        r = httpx.post(_url("/api/embed"), json={"model": model or s["embed_model"],
                                                 "input": texts}, timeout=60)
        r.raise_for_status()
        return r.json()["embeddings"]
    except Exception as e:
        log.debug("embed failed: %s", e)
        return None
