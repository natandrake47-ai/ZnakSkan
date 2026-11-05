"use strict";
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const pct = v => Math.round((v || 0) * 100) + "%";
const LVL_COLOR = {high: "var(--high)", medium: "var(--medium)", low: "var(--low)", none: "var(--none)"};

async function api(url, opts = {}) {
  const r = await fetch(url, opts);
  const ct = r.headers.get("content-type") || "";
  const data = ct.includes("json") ? await r.json() : await r.text();
  if (!r.ok) throw new Error(data.detail || data || r.statusText);
  return data;
}
const jsonOpts = (method, body) => ({method, headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});

/* ---------------- вкладки ---------------- */
$$("#tabs button").forEach(b => b.onclick = () => showTab(b.dataset.tab));
function showTab(name) {
  $$("#tabs button").forEach(x => x.classList.toggle("active", x.dataset.tab === name));
  $$(".tab").forEach(x => x.classList.toggle("active", x.id === "tab-" + name));
  if (name === "db") loadDb();
  if (name === "settings") loadSettings();
}

/* ---------------- статус ---------------- */
let STATUS = null;
async function loadStatus() {
  try {
    STATUS = await api("/api/status");
    $("#sys").innerHTML =
      `<span><i class="dot ${STATUS.ollama && STATUS.chat_model_ready ? "ok" : "bad"}"></i>ИИ ${STATUS.ollama ? (STATUS.chat_model_ready ? "готов" : "модель не скачана") : "не запущен"}</span>` +
      `<span><i class="dot ok"></i>База: ${STATUS.db.marks} знаков</span>`;
  } catch (e) { $("#sys").textContent = "Сервер недоступен"; }
}
loadStatus(); setInterval(loadStatus, 30000);

/* ---------------- превью изображений ---------------- */
$$(".mark-card input[type=file]").forEach(inp => inp.onchange = () => {
  const img = $(".preview", inp.closest(".mark-card"));
  if (inp.files[0]) { img.src = URL.createObjectURL(inp.files[0]); img.hidden = false; } else img.hidden = true;
});

/* ---------------- сравнение ---------------- */
let LAST_COMPARE = null;
$("#btn-compare").onclick = async () => {
  const fd = new FormData();
  for (const card of $$(".mark-card")) {
    const s = card.dataset.side;
    fd.append("name_" + s, $("[name=name]", card).value);
    fd.append("classes_" + s, $("[name=classes]", card).value);
    fd.append("goods_" + s, $("[name=goods]", card).value);
    const f = $("[name=image]", card).files[0];
    if (f) fd.append("image_" + s, f);
  }
  fd.append("use_ai", $("#cmp-ai").checked);
  const btn = $("#btn-compare"); btn.disabled = true; btn.textContent = "Сравниваю…";
  try {
    const r = await api("/api/compare", {method: "POST", body: fd});
    LAST_COMPARE = r;
    $("#cmp-result").innerHTML = renderResult(r, true);
    bindResultButtons();
  } catch (e) { $("#cmp-result").innerHTML = `<div class="warn error">${esc(e.message)}</div>`; }
  btn.disabled = false; btn.textContent = "Сравнить";
};

function bar(label, v, details, sub) {
  return `<div class="bar"><div class="lbl"><span>${label}${sub ? ` <small class="muted">${sub}</small>` : ""}</span><b>${pct(v)}</b></div>
    <div class="track"><div class="fill" style="width:${pct(v)}"></div></div>
    ${details && details.length ? `<ul>${details.map(d => `<li>${esc(d)}</li>`).join("")}</ul>` : ""}</div>`;
}
function absBlock(list, title) {
  if (!list || !list.length) return "";
  return `<h4>${esc(title)}</h4>` + list.map(h => `<div class="warn ${h.level}"><b>${esc(h.ground)}.</b> ${esc(h.text)}</div>`).join("");
}
function renderResult(r, full) {
  const c = LVL_COLOR[r.level];
  const w = r.word, img = r.image;
  let bars = "";
  if (w) {
    const ws = STATUS_WEIGHTS?.word || {};
    if (w.identity) bars += `<div class="identity">Тождество: ${esc(w.identity_reason)}</div>`;
    bars += `<h4>Словесный элемент — ${pct(w.score)}</h4>`;
    bars += bar("Звуковое (фонетическое) сходство", w.phonetic.score, w.phonetic.details, ws.phonetic ? `вес ${pct(ws.phonetic)}` : "");
    const semNA = w.semantic.parts && w.semantic.parts.applicable === 0;
    bars += bar("Смысловое (семантическое) сходство" + (semNA ? " — не применимо" : ""), w.semantic.score, w.semantic.details, ws.semantic && !semNA ? `вес ${pct(ws.semantic)}` : "");
    bars += bar("Графическое сходство написания", w.visual.score, w.visual.details, ws.visual ? `вес ${pct(ws.visual)}` : "");
  }
  if (img) { bars += `<h4>Изобразительный элемент</h4>` + bar("Визуальное сходство изображений", img.score, img.details); }
  bars += `<h4>Однородность товаров и услуг</h4>` + bar("Классы МКТУ", r.classes.score, r.classes.details);
  return `<div class="card"><div class="result">
    <div class="gauge">
      <div class="ring" style="background:conic-gradient(${c} ${r.risk * 360}deg, var(--line) 0)"><div class="inner"><div><div class="pct">${pct(r.risk)}</div><div class="sub">риск смешения</div></div></div></div>
      <p><span class="badge ${r.level}">${esc(r.level_name)} степень сходства</span></p>
      <p class="muted">Сходство обозначений: <b>${pct(r.mark_similarity)}</b><br>Сравнивались: ${esc(r.compared)}</p>
      ${full ? `<div class="actions" style="justify-content:center;flex-direction:column;gap:8px">
        <button id="btn-discuss">Обсудить с ИИ</button>
        <button id="btn-lab1" title="Сохранить как размеченный пример для подбора весов">Казпатент признал сходными</button>
        <button id="btn-lab0">Казпатент признал НЕ сходными</button></div>` : ""}
    </div>
    <div>
      <div class="verdict"><b>${esc(r.verdict)}.</b></div>
      <ul class="explain">${r.explanation.map(x => `<li>${esc(x)}</li>`).join("")}</ul>
      ${r.notes && r.notes.length ? r.notes.map(n => `<div class="warn info">${esc(n)}</div>`).join("") : ""}
      <div class="bars">${bars}</div>
      ${absBlock(r.absolute_a, "Охраноспособность знака 1 (предварительно)")}
      ${absBlock(r.absolute_b, "Охраноспособность знака 2 (предварительно)")}
      <p class="muted">Оценка рассчитана по предложенным весам и не заменяет экспертизу Казпатента.</p>
    </div></div></div>`;
}
function bindResultButtons() {
  const d = $("#btn-discuss");
  if (d) d.onclick = () => { setChatContext(LAST_COMPARE, `${LAST_COMPARE.a} ↔ ${LAST_COMPARE.b}`); showTab("chat"); };
  for (const [id, v] of [["#btn-lab1", 1], ["#btn-lab0", 0]]) {
    const b = $(id); if (!b) continue;
    b.onclick = async () => {
      const r = await api("/api/labeled", jsonOpts("POST", {a: LAST_COMPARE.input.a, b: LAST_COMPARE.input.b, confusing: v}));
      b.textContent = `Сохранено (всего примеров: ${r.count})`; b.disabled = true;
    };
  }
}

/* ---------------- поиск ---------------- */
let SEARCH = [];
$("#btn-search").onclick = async () => {
  const fd = new FormData();
  fd.append("name", $("#s-name").value);
  fd.append("classes", $("#s-classes").value);
  fd.append("related", $("#s-related").checked);
  fd.append("use_ai", $("#s-ai").checked);
  const f = $("#s-image").files[0]; if (f) fd.append("image", f);
  const btn = $("#btn-search"); btn.disabled = true; btn.textContent = "Ищу…";
  try {
    const r = await api("/api/search", {method: "POST", body: fd});
    SEARCH = r.results;
    $("#s-abs").innerHTML = r.absolute.length ? `<div class="card">${absBlock(r.absolute, "Предварительная проверка охраноспособности")}</div>` : "";
    renderSearch(r);
  } catch (e) { $("#s-result").innerHTML = `<div class="warn error">${esc(e.message)}</div>`; }
  btn.disabled = false; btn.textContent = "Найти";
};
const mini = v => `<span class="mini"><i style="width:${pct(v)}"></i></span>${v == null ? "—" : pct(v)}`;
function renderSearch(r) {
  if (!r.total_in_db) { $("#s-result").innerHTML = `<div class="warn info">База пуста. Загрузите бюллетени на вкладке «База знаков».</div>`; return; }
  if (!SEARCH.length) { $("#s-result").innerHTML = `<div class="card">Сходных знаков не найдено (в базе ${r.total_in_db}).</div>`; return; }
  const rows = SEARCH.map((x, i) => {
    const m = x.mark, w = x.word;
    return `<tr class="clickable" data-i="${i}">
      <td>${m.image_path ? `<img src="/images/${esc(m.image_path)}">` : ""}</td>
      <td><b>${esc(m.name || "(изобразительный)")}</b><br><small class="muted">${esc(m.owner || "")}</small></td>
      <td>${esc(m.reg_number || "")}<br><small class="muted">${esc(m.app_number || "")}</small></td>
      <td>${esc((m.classes || []).join(", "))}</td>
      <td>${w ? mini(w.phonetic.score) : "—"}</td>
      <td>${w ? mini(w.semantic.score) : "—"}</td>
      <td>${w ? mini(w.visual.score) : "—"}</td>
      <td>${x.image ? mini(x.image.score) : "—"}</td>
      <td><b style="color:${LVL_COLOR[x.level]}">${pct(x.risk)}</b><br><span class="badge ${x.level}">${esc(x.level_name)}</span></td></tr>`;
  }).join("");
  $("#s-result").innerHTML = `<div class="card"><h3>Найдено: ${SEARCH.length} <small>из ${r.total_in_db} знаков в базе, по убыванию риска</small></h3>
    <table><thead><tr><th></th><th>Обозначение / владелец</th><th>Рег. № / заявка</th><th>Классы</th><th>Фонетика</th><th>Смысл</th><th>Графика</th><th>Изобр.</th><th>Риск</th></tr></thead><tbody>${rows}</tbody></table></div>`;
  $$("#s-result tr.clickable").forEach(tr => tr.onclick = () => {
    const nx = tr.nextElementSibling;
    if (nx && nx.classList.contains("detail-row")) { nx.remove(); return; }
    const x = SEARCH[+tr.dataset.i];
    tr.insertAdjacentHTML("afterend", `<tr class="detail-row"><td colspan="9">${renderResult(x, false)}
      <button class="discuss-row">Обсудить этот знак с ИИ</button></td></tr>`);
    $(".discuss-row", tr.nextElementSibling).onclick = () => { setChatContext(x, `${x.a} ↔ ${x.mark.name || x.b}`); showTab("chat"); };
  });
}

/* ---------------- база ---------------- */
let DB_OFF = 0; const DB_LIM = 30;
async function loadDb() {
  const s = (await api("/api/status")).db;
  const bul = await api("/api/bulletins");
  $("#db-stats").innerHTML = `<div class="stat"><b>${s.marks}</b>знаков</div><div class="stat"><b>${s.with_images}</b>с изображением</div><div class="stat"><b>${s.bulletins}</b>бюллетеней</div>
    ${s.without_name ? `<div class="warn info">У ${s.without_name} знаков не распознано название (в бюллетене оно только на картинке). Без названия знак находится только поиском по изображению. <button id="btn-ocr">Распознать названия</button> или подпишите вручную, щёлкнув по названию в таблице.</div><div id="ocr-job"></div>` : ""}
    <div class="classes-cloud">${Object.entries(s.by_class).map(([k, v]) => `<span title="класс ${k}">${k}: ${v}</span>`).join("")}</div>
    ${bul.length ? `<details><summary>Загруженные файлы</summary><table>${bul.map(b => `<tr><td>${esc(b.filename)}</td><td>${b.marks_count} зн.</td><td class="muted">${esc(b.imported_at)}</td></tr>`).join("")}</table></details>` : ""}`;
  const ob = $("#btn-ocr");
  if (ob) ob.onclick = async () => { const {job} = await api("/api/ocr", {method: "POST"}); pollJob(job, $("#ocr-job")); };
  loadList();
}
async function loadList() {
  const q = encodeURIComponent($("#db-q").value), c = $("#db-cls").value.trim();
  const r = await api(`/api/marks?q=${q}&offset=${DB_OFF}&limit=${DB_LIM}${c ? "&cls=" + c : ""}${$("#db-noname").checked ? "&no_name=true" : ""}`);
  $("#db-list").innerHTML = r.items.length ? `<table><thead><tr><th></th><th>Обозначение</th><th>Рег. №</th><th>Заявка</th><th>Классы</th><th>Владелец</th><th>Источник</th><th></th></tr></thead><tbody>
    ${r.items.map(m => `<tr><td>${m.image_path ? `<img src="/images/${esc(m.image_path)}">` : ""}</td><td><b class="editable" data-edit="${m.id}" title="Щёлкните, чтобы исправить название">${m.name ? esc(m.name) : '<span class="muted">(нет названия)</span>'}</b>${m.image_desc ? `<br><small class="muted">${esc(m.image_desc)}</small>` : ""}</td><td>${esc(m.reg_number)}</td><td>${esc(m.app_number)}</td>
      <td>${esc(m.classes.join(", "))}</td><td>${esc(m.owner)}</td><td class="muted">${esc(m.source)}</td><td><button data-del="${m.id}" title="Удалить">✕</button></td></tr>`).join("")}</tbody></table>`
    : `<p class="muted">Пусто.</p>`;
  $("#db-page").textContent = `${r.total ? DB_OFF + 1 : 0}–${Math.min(DB_OFF + DB_LIM, r.total)} из ${r.total}`;
  $("#db-prev").disabled = DB_OFF === 0; $("#db-next").disabled = DB_OFF + DB_LIM >= r.total;
  $$("[data-edit]").forEach(b => b.onclick = async () => {
    const cur = b.querySelector(".muted") ? "" : b.textContent;
    const v = prompt("Словесный элемент знака (как написан на изображении):", cur);
    if (v === null) return;
    await api("/api/marks/" + b.dataset.edit, jsonOpts("PATCH", {name: v}));
    loadList(); loadStatus();
  });
  $$("[data-del]").forEach(b => b.onclick = async () => { if (confirm("Удалить знак из базы?")) { await api("/api/marks/" + b.dataset.del, {method: "DELETE"}); loadList(); } });
}
$("#db-noname").onchange = () => { DB_OFF = 0; loadList(); };
let dbT; $("#db-q").oninput = $("#db-cls").oninput = () => { clearTimeout(dbT); dbT = setTimeout(() => { DB_OFF = 0; loadList(); }, 300); };
$("#db-prev").onclick = () => { DB_OFF = Math.max(0, DB_OFF - DB_LIM); loadList(); };
$("#db-next").onclick = () => { DB_OFF += DB_LIM; loadList(); };
$("#add-form").onsubmit = async e => {
  e.preventDefault();
  await api("/api/marks", {method: "POST", body: new FormData(e.target)});
  e.target.reset(); loadDb(); loadStatus();
};

const drop = $("#drop");
drop.ondragover = e => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = e => { e.preventDefault(); drop.classList.remove("over"); uploadFiles(e.dataTransfer.files); };
$("#imp-file").onchange = e => uploadFiles(e.target.files);
async function uploadFiles(files) {
  for (const f of files) {
    const fd = new FormData(); fd.append("file", f); fd.append("use_ocr", $("#imp-ocr").checked);
    const el = document.createElement("div"); el.className = "job"; el.textContent = f.name + ": загрузка…";
    $("#jobs").prepend(el);
    try { const {job} = await api("/api/import", {method: "POST", body: fd}); pollJob(job, el); }
    catch (e) { el.innerHTML = `${esc(f.name)}: <span style="color:var(--high)">${esc(e.message)}</span>`; }
  }
}
async function pollJob(id, el) {
  const j = await api("/api/jobs/" + id);
  const p = j.total ? j.done / j.total : 0;
  el.innerHTML = `<b>${esc(j.file)}</b> — ${esc(j.stage)} ${j.total ? `(${j.done}/${j.total})` : ""}
    ${j.status === "running" ? `<div class="progress"><i style="width:${pct(p)}"></i></div>` : ""}
    ${j.status === "done" ? `<br>Добавлено/обновлено знаков: <b>${j.count}</b>` : ""}
    ${j.log.map(l => `<div class="muted">${esc(l)}</div>`).join("")}`;
  if (j.status === "running") setTimeout(() => pollJob(id, el), 1000); else { loadDb(); loadStatus(); }
}

/* ---------------- чат ---------------- */
let HISTORY = [], CHAT_CTX = null;
function setChatContext(ctx, title) {
  CHAT_CTX = ctx;
  const el = $("#chat-ctx");
  el.hidden = false;
  el.innerHTML = `<span>В контексте разговора: результат сравнения <b>${esc(title)}</b> (${pct(ctx.risk)})</span><a href="#" id="ctx-x">убрать</a>`;
  $("#ctx-x").onclick = e => { e.preventDefault(); CHAT_CTX = null; el.hidden = true; };
  $("#chat-input").value = $("#chat-input").value || "Объясни этот результат и оцени риски регистрации.";
}
function addMsg(role, text, tools) {
  const d = document.createElement("div");
  d.className = "msg " + role; d.textContent = text;
  if (tools && tools.length) d.insertAdjacentHTML("beforeend", `<div class="tools">Использовано: ${tools.map(t => esc(t.tool)).join(", ")}</div>`);
  $("#chat-log").append(d); $("#chat-log").scrollTop = 1e9; return d;
}
$("#chat-form").onsubmit = async e => {
  e.preventDefault();
  const t = $("#chat-input").value.trim(); if (!t) return;
  $("#chat-input").value = "";
  addMsg("user", t); HISTORY.push({role: "user", content: t});
  const typing = addMsg("assistant typing", "Думаю…");
  try {
    const r = await api("/api/chat", jsonOpts("POST", {history: HISTORY, context: CHAT_CTX}));
    typing.remove(); addMsg("assistant", r.content, r.tool_calls);
    if (!r.error) HISTORY.push({role: "assistant", content: r.content});
  } catch (err) { typing.remove(); addMsg("assistant", "Ошибка: " + err.message); }
};
$("#chat-input").onkeydown = e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#chat-form").requestSubmit(); } };

/* ---------------- настройки и веса ---------------- */
let STATUS_WEIGHTS = null;
api("/api/weights").then(w => STATUS_WEIGHTS = w);
const W_LABELS = {
  word: ["Словесный элемент", {phonetic: "Фонетическое (звуковое)", semantic: "Смысловое (семантическое)", visual: "Графическое (написание)"}],
  phonetic_parts: ["Фонетика: составляющие", {strict: "Совпадение звуков", loose: "Восприятие на слух (редукция гласных)", consonants: "Согласный «скелет»", beginning: "Совпадение начала", syllables: "Число слогов и длина"}],
  combined: ["Комбинированный знак", {word: "Словесный элемент", image: "Изобразительный элемент"}],
  image_parts: ["Изображение (с CLIP)", {clip: "Общее впечатление (CLIP)", shape: "Форма/контур", structure: "Композиция", color: "Цвет"}],
  image_parts_no_clip: ["Изображение (без CLIP)", {shape: "Форма/контур", structure: "Композиция", color: "Цвет"}],
  classes: ["Однородность классов", {same: "Тот же класс", related: "Смежные классы", retail_35: "35 класс и товары", unrelated: "Неоднородные", influence: "Влияние классов на итог (λ)"}],
  thresholds: ["Пороги степени сходства", {high: "Высокая от", medium: "Средняя от", low: "Низкая от"}],
};
async function loadSettings() {
  const w = STATUS_WEIGHTS = await api("/api/weights");
  let h = "";
  for (const [g, [title, items]] of Object.entries(W_LABELS)) {
    h += `<div class="wgroup"><b>${title}</b>` + Object.entries(items).map(([k, l]) =>
      `<div class="wrow"><span>${l}</span><input type="number" step="0.01" min="0" max="1" data-g="${g}" data-k="${k}" value="${w[g]?.[k] ?? 0}"></div>`).join("") + "</div>";
  }
  h += `<div class="wgroup"><div class="wrow"><span>Доля «максимума» в итоге (β): сходство по одному признаку</span><input type="number" step="0.05" min="0" max="1" id="w-beta" value="${w.max_blend}"></div>
    <div class="wrow"><span>Вес слабых элементов</span><input type="number" step="0.05" min="0" max="1" id="w-weak" value="${w.weak_element_weight}"></div></div>`;
  $("#weights-form").innerHTML = h;
  const st = await api("/api/status");
  const s = st.settings, f = $("#settings-form");
  for (const k of ["ollama_url", "chat_model", "embed_model", "vision_model"]) f[k].value = s[k] || "";
  for (const k of ["use_llm_semantic", "use_embeddings", "use_clip"]) f[k].checked = !!s[k];
  $("#models").innerHTML = st.models.map(m => `<option value="${esc(m)}">`).join("");
  const y = (ok, t) => `<div><i class="dot ${ok ? "ok" : "bad"}"></i>${t}</div>`;
  $("#ai-status").innerHTML = y(st.ollama, "Ollama " + (st.ollama ? "запущен" : "не найден — установите с ollama.com")) +
    y(st.chat_model_ready, `Модель диалога ${esc(s.chat_model)}` + (st.chat_model_ready ? "" : ` — выполните: ollama pull ${esc(s.chat_model)}`)) +
    y(st.embed_model_ready, `Эмбеддинги ${esc(s.embed_model)}` + (st.embed_model_ready ? "" : ` — ollama pull ${esc(s.embed_model)}`)) +
    y(st.vision_model_ready, `Vision ${esc(s.vision_model)}` + (st.vision_model_ready ? "" : ` — ollama pull ${esc(s.vision_model)}`)) +
    y(st.clip, "CLIP для изображений" + (st.clip ? "" : " — необязательно: pip install sentence-transformers")) + "<br>";
}
$("#w-save").onclick = async () => {
  const w = STATUS_WEIGHTS;
  $$("#weights-form [data-g]").forEach(i => { w[i.dataset.g][i.dataset.k] = parseFloat(i.value) || 0; });
  w.max_blend = parseFloat($("#w-beta").value); w.weak_element_weight = parseFloat($("#w-weak").value);
  STATUS_WEIGHTS = await api("/api/weights", jsonOpts("PUT", w));
  $("#w-save").textContent = "Сохранено ✓"; setTimeout(() => $("#w-save").textContent = "Сохранить", 1500);
};
$("#w-reset").onclick = async () => { if (confirm("Вернуть веса по умолчанию?")) { await api("/api/weights/reset", {method: "POST"}); loadSettings(); } };
$("#w-calib").onclick = async () => {
  const r = await api("/api/calibrate", {method: "POST"});
  if (!r.ok) { $("#calib-res").innerHTML = `<div class="warn info">${esc(r.message)}</div>`; return; }
  $("#calib-res").innerHTML = `<div class="warn info">${esc(r.message)} Предлагаемые веса: фонетика ${pct(r.weights.word.phonetic)}, смысл ${pct(r.weights.word.semantic)}, графика ${pct(r.weights.word.visual)}; порог высокой степени ${pct(r.weights.thresholds.high)}.
    <button id="calib-apply">Применить</button></div>`;
  $("#calib-apply").onclick = async () => { await api("/api/calibrate?apply=true", {method: "POST"}); loadSettings(); };
};
$("#settings-form").onsubmit = async e => {
  e.preventDefault();
  const f = e.target, d = {};
  for (const k of ["ollama_url", "chat_model", "embed_model", "vision_model"]) d[k] = f[k].value.trim();
  for (const k of ["use_llm_semantic", "use_embeddings", "use_clip"]) d[k] = f[k].checked;
  await api("/api/settings", jsonOpts("PUT", d)); loadSettings(); loadStatus();
};
