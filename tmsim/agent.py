"""Локальный ИИ-агент: диалог с пользователем + инструменты (поиск по базе,
сравнение знаков, проверка охраноспособности)."""
from __future__ import annotations

import json
import logging

from . import absolute, db, ollama
from .config import settings
from .nice import NICE
from .scoring import Mark, compare
from .search import search

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """Ты — ИИ-помощник патентного поверенного Республики Казахстан. Работаешь локально,
внутри программы оценки сходства товарных знаков. Отвечай по-русски (или на казахском, если пользователь пишет по-казахски),
понятно и по делу.

Что ты знаешь:
- Экспертиза Казпатента (РГП «НИИС») сравнивает обозначения по признакам сходства: звуковому (фонетическому),
  графическому (визуальному) и смысловому (семантическому). Сходство может быть признано и по одному признаку.
- Звуковое: близость звуков, совпадение начальных частей, вхождение одного обозначения в другое, число слогов, ударение.
- Графическое: общее зрительное впечатление, вид шрифта, алфавит, цвет и сочетание цветов, расположение элементов.
- Смысловое: подобие заложенных понятий и идей, совпадение значения при переводе, противоположность идей.
- Сильные элементы (запоминаемые, фантазийные) важнее слабых (описательные, хвалебные, географические, неохраняемые).
- Тождество — отличие только падежом, числом, уменьшительной формой, наличием/отсутствием предлога.
- Однородность товаров/услуг по классам МКТУ (род, назначение, круг потребителей, каналы сбыта) — без неё
  сходство обозначений само по себе не ведёт к смешению.
- Абсолютные основания (ст. 6 Закона РК о товарных знаках): отсутствие различительной способности, госсимволика,
  введение в заблуждение, противоречие общественным интересам, принципам гуманности и морали, объекты культурного наследия и т.д.

Правила:
- Для фактов о базе и для расчётов ВСЕГДА вызывай инструменты, не выдумывай номера регистраций и проценты.
- Проценты, которые выдаёт программа, — расчётная оценка с предложенными весами, а не решение Казпатента. Так и говори.
- Если данных не хватает (класс МКТУ, точное написание), задай уточняющий вопрос.
"""

TOOLS = [
    {"type": "function", "function": {
        "name": "search_similar_marks",
        "description": "Найти в локальной базе зарегистрированные товарные знаки, сходные с обозначением",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "Словесное обозначение"},
            "classes": {"type": "array", "items": {"type": "integer"}, "description": "Классы МКТУ"}},
            "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "compare_marks",
        "description": "Рассчитать степень сходства двух словесных обозначений по критериям Казпатента",
        "parameters": {"type": "object", "properties": {
            "name_a": {"type": "string"}, "name_b": {"type": "string"},
            "classes_a": {"type": "array", "items": {"type": "integer"}},
            "classes_b": {"type": "array", "items": {"type": "integer"}}},
            "required": ["name_a", "name_b"]}}},
    {"type": "function", "function": {
        "name": "get_mark",
        "description": "Получить сведения о знаке из базы по номеру регистрации или заявки",
        "parameters": {"type": "object", "properties": {"number": {"type": "string"}},
                       "required": ["number"]}}},
    {"type": "function", "function": {
        "name": "check_absolute_grounds",
        "description": "Предварительная проверка обозначения на абсолютные основания для отказа",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"},
            "classes": {"type": "array", "items": {"type": "integer"}}},
            "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "nice_classes",
        "description": "Названия классов МКТУ по номерам",
        "parameters": {"type": "object", "properties": {
            "classes": {"type": "array", "items": {"type": "integer"}}}, "required": ["classes"]}}},
    {"type": "function", "function": {
        "name": "database_stats",
        "description": "Сколько знаков в локальной базе и по каким классам",
        "parameters": {"type": "object", "properties": {}}}},
]


def _short(res: dict) -> dict:
    out = {"сходство_знаков": f"{res['mark_similarity']:.0%}", "риск_смешения": f"{res['risk']:.0%}",
           "уровень": res["level_name"], "вывод": res["verdict"], "пояснение": res["explanation"],
           "классы": res["classes"]["details"]}
    if res.get("word"):
        w = res["word"]
        out["фонетика"] = f"{w['phonetic']['score']:.0%}"
        out["семантика"] = f"{w['semantic']['score']:.0%}"
        out["графика"] = f"{w['visual']['score']:.0%}"
        out["тождество"] = w["identity"]
        out["детали"] = w["phonetic"]["details"] + w["semantic"]["details"]
    if res.get("mark"):
        m = res["mark"]
        out["знак"] = {"рег_номер": m.get("reg_number"), "заявка": m.get("app_number"),
                       "обозначение": m.get("name"), "владелец": m.get("owner"), "классы": m.get("classes")}
    return out


def run_tool(name: str, args: dict):
    if name == "search_similar_marks":
        res = search(Mark(name=args.get("name", ""), classes=args.get("classes") or []), limit=10)
        return [_short(r) for r in res] or "В базе ничего похожего не найдено (или база пуста)."
    if name == "compare_marks":
        r = compare(Mark(args["name_a"], args.get("classes_a") or []),
                    Mark(args["name_b"], args.get("classes_b") or []), use_ai=True)
        return _short(r)
    if name == "get_mark":
        return db.find_by_number(str(args.get("number"))) or "Не найдено"
    if name == "check_absolute_grounds":
        return absolute.check(args.get("name", ""), args.get("classes")) or "Явных абсолютных оснований не найдено."
    if name == "nice_classes":
        return {c: NICE.get(int(c), "?") for c in args.get("classes", [])}
    if name == "database_stats":
        return db.stats()
    return f"Неизвестный инструмент {name}"


def chat(history: list[dict], context: dict | None = None, max_steps: int = 6) -> dict:
    if not ollama.available():
        return {"content": "Локальная модель не запущена. Установите Ollama (https://ollama.com), затем выполните "
                           f"`ollama pull {settings()['chat_model']}` и перезапустите программу.",
                "tool_calls": [], "error": "ollama_unavailable"}
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context:
        msgs.append({"role": "system", "content": "Последний результат сравнения в программе (контекст разговора):\n"
                     + json.dumps(_short(context) if "risk" in context else context, ensure_ascii=False)[:6000]})
    msgs += [{"role": m["role"], "content": m["content"]} for m in history if m.get("role") in ("user", "assistant")]
    used = []
    for _ in range(max_steps):
        try:
            msg = ollama.chat(msgs, tools=TOOLS)
        except Exception as e:
            return {"content": f"Ошибка обращения к модели: {e}", "tool_calls": used, "error": "llm_error"}
        calls = msg.get("tool_calls") or []
        if not calls:
            return {"content": msg.get("content", ""), "tool_calls": used}
        msgs.append(msg)
        for c in calls:
            fn = c["function"]["name"]
            args = c["function"].get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            try:
                result = run_tool(fn, args)
            except Exception as e:
                result = f"Ошибка инструмента: {e}"
            used.append({"tool": fn, "args": args})
            msgs.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False, default=str)[:8000],
                         "tool_name": fn})
    return {"content": "Не удалось завершить ответ за отведённое число шагов.", "tool_calls": used}
