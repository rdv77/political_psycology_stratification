"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const STAGES = {relevance: "Отбор материалов", extract: "Извлечение фактов", describe: "Подготовка признаков", score: "Оценивание"};
  const state = {corpora: [], studies: [], corpus: null, modelConfig: {}, busy: false, dirty: false, offset: 0, total: 0, timer: null, selection: 0};
  const clone = (v) => JSON.parse(JSON.stringify(v));
  const number = (v) => new Intl.NumberFormat("ru-RU", {maximumFractionDigits: 5}).format(Number(v) || 0);
  function node(tag, attrs = {}, ...children) {
    const item = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs)) {
      if (value === null || value === undefined) continue;
      if (key === "text") item.textContent = String(value);
      else if (key === "class") item.className = value;
      else if (key.startsWith("on") && typeof value === "function") item.addEventListener(key.slice(2), value);
      else if (["value", "disabled", "checked"].includes(key)) item[key] = value;
      else item.setAttribute(key, String(value));
    }
    for (const child of children.flat(Infinity)) if (child !== null && child !== undefined) item.append(child instanceof Node ? child : document.createTextNode(String(child)));
    return item;
  }
  function message(text, error = false) {
    $("message").hidden = !text;
    $("message").className = error ? "error" : "notice";
    $("message").textContent = text || "";
  }
  async function api(path, method = "GET", data) {
    const options = {method, cache: "no-store", credentials: "same-origin", headers: {Accept: "application/json"}};
    if (data !== undefined) { options.headers["Content-Type"] = "application/json"; options.body = JSON.stringify(data); }
    const response = await fetch(path, options);
    let result;
    try { result = await response.json(); } catch (_) { throw new Error("Сервис вернул непонятный ответ. Проверьте локальный сервер."); }
    if (!response.ok) throw new Error(result.error || `Ошибка сервиса: ${response.status}`);
    return result;
  }
  const path = (suffix = "", id = state.corpus?.id) => `/api/corpora/${encodeURIComponent(id)}${suffix}`;
  const running = () => state.corpus?.http_job?.status === "running";
  function controls() {
    const locked = state.busy || running();
    for (const id of ["import-submit", "save-config", "copy-study", "export", "use-draft"]) $(id).disabled = locked;
    $("prepare").disabled = locked || state.dirty;
    $("run-api").disabled = locked || state.dirty || !state.modelConfig.configured || !(state.corpus?.jobs?.pending > 0);
    $("refresh").disabled = state.busy;
  }
  async function perform(action) {
    if (state.busy) return;
    state.busy = true; message(""); controls();
    try { await action(); } catch (error) { message(error.message || "Операция не завершена", true); }
    finally { state.busy = false; controls(); }
  }
  function renderList() {
    const list = $("corpus-list"); list.replaceChildren();
    if (!state.corpora.length) list.append(node("p", {class: "hint", text: "Пока нет корпусов"}));
    for (const corpus of state.corpora) list.append(node("button", {type: "button", class: `corpus-item${corpus.id === state.corpus?.id ? " active" : ""}`, onclick: () => perform(() => select(corpus.id))}, corpus.title || corpus.id, node("small", {text: corpus.source_name || corpus.id})));
  }
  async function refreshList() {
    const result = await api("/api/corpora"); state.corpora = result.corpora || []; state.modelConfig = result.model_config || {};
    $("connection").textContent = state.modelConfig.configured ? "API подключён · запуск вручную" : "Локальная обработка · API не подключён";
    renderList(); controls();
  }
  function renderProfile() {
    const c = state.corpus, counts = c.counts || {}, profile = c.profile || {}, jobs = c.jobs || {}, usage = c.usage || {};
    $("corpus-workspace").hidden = false;
    $("corpus-title").textContent = c.title || c.id;
    $("source-note").textContent = [c.source_name, c.created_at?.slice(0, 10)].filter(Boolean).join(" · ");
    $("run-status").textContent = running() ? "Идёт обработка API" : c.http_job?.status === "error" ? "Запуск остановлен" : "Локальный корпус";
    $("metrics").replaceChildren(...[[counts.documents ?? profile.rows_total, "Документов"], [counts.unique_documents, "Уникальных"], [counts.exact_duplicates, "Точных дублей"], [counts.empty_documents ?? profile.empty_rows, "Без текста"]].map(([value, label]) => node("div", {class: "metric"}, node("strong", {text: number(value)}), node("span", {text: label}))));
    $("profile-detail").textContent = [profile.encoding && `Кодировка: ${profile.encoding}`, profile.delimiter && `Разделитель: ${profile.delimiter === "\t" ? "табуляция" : profile.delimiter}`, `Кандидатов близких дублей: ${number(counts.near_duplicate_candidates)}`, `Фактов: ${number(counts.facts)}`, `Оценок: ${number(counts.ratings)}`].filter(Boolean).join(" · ");
    $("profile-columns").textContent = (profile.columns || []).join(" · ") || "Столбцы не определены";
    $("profile-warnings").hidden = !profile.warnings?.length;
    $("profile-warnings").textContent = (profile.warnings || []).join(" · ");
    $("queue-label").textContent = `В очереди: ${number(jobs.pending)}`;
    $("job-counts").replaceChildren(...[["done", "Готово"], ["running", "В работе"], ["review", "На проверку"], ["uncertain", "Неопределённо"], ["error", "Ошибок"]].map(([key, label]) => node("span", {}, label + ": ", node("strong", {text: number(jobs[key])}))));
    $("usage").textContent = `Учтено за всю историю: $${number(usage.estimated_usd)} · вход ${number(usage.input_tokens)} / выход ${number(usage.output_tokens)} токенов${c.config?.budget_limit_usd != null ? ` · общий лимит $${number(c.config.budget_limit_usd)}` : ""}${usage.uncertain_reserved_usd ? ` · неопределённый резерв $${number(usage.uncertain_reserved_usd)}` : ""}`;
    $("api-note").textContent = state.modelConfig.configured ? "Кнопка ниже отправляет подготовленные материалы провайдеру моделей. Бюджет ограничивает оценку расходов, но не задаёт фиксированную цену." : "Для вызовов моделей нужен ключ на сервере в локальном .env. Импорт, просмотр и подготовка доступны без ключа.";
    $("progress-note").textContent = running() ? "Запуск выполняется в фоне. Статус обновляется автоматически; страницу можно оставить открытой." : c.http_job?.status === "error" ? `Ошибка: ${c.http_job.error || "запуск не завершён"}` : c.http_job?.status === "complete" ? "Последний запуск завершён. Проверьте очередь и результаты перед продолжением." : "API ещё не запускался с этой страницы.";
    if (c.http_job?.status === "complete" && c.http_job.result?.stopped_reason) {
      const result = c.http_job.result;
      const reason = {queue_empty: "очередь пуста", job_limit: "достигнут лимит заданий", budget_reservation: "следующее задание превышает оставшийся бюджет", corpus_budget_reservation: "достигнут общий лимит корпуса с учётом предыдущих запусков", configuration_changed: "настройки изменились", uncertain_request: "результат или оплата запроса требуют проверки перед повтором"}[result.stopped_reason] || result.stopped_reason;
      $("progress-note").textContent = `Запуск завершён: готово ${number(result.completed)}, на проверку ${number(result.review)}, ошибок ${number(result.failed)}, неопределённых ${number(result.uncertain)}. Причина остановки: ${reason}.`;
    }
    const draft = Array.isArray(c.draft_scales) ? c.draft_scales : c.draft_scales?.scales || [];
    $("draft-card").hidden = !draft.length;
    $("draft-scales-json").value = JSON.stringify(draft, null, 2);
    const statusNames = {pending: "В очереди", running: "В работе", done: "Готово", review: "Нужна проверка", error: "Ошибка", uncertain: "Неопределённо", obsolete: "Устарело после изменения настроек"};
    $("recent-jobs").replaceChildren(...(c.recent_jobs || []).slice(0, 15).map(job => node("details", {},
      node("summary", {text: `${STAGES[job.stage] || job.stage || job.kind || "Задание"} · ${statusNames[job.status] || job.status || "—"} · ${job.id || ""}`}),
      node("pre", {text: JSON.stringify({error: job.error || null, result: job.result || job.output || null, model: job.model || null}, null, 2)}))));
    if (!(c.recent_jobs || []).length) $("recent-jobs").append(node("p", {class: "hint", text: "Пока нет заданий"}));
    controls();
  }
  function readPricing() {
    const values = {};
    for (const row of $("pricing").querySelectorAll(".price-row")) {
      const record = {};
      for (const input of row.querySelectorAll("input")) if (input.value.trim()) record[input.dataset.field] = Number(input.value);
      if (Object.keys(record).length) values[row.dataset.model] = record;
    }
    return values;
  }
  function renderPricing(pricing = {}) {
    const merged = {...state.modelConfig.pricing, ...pricing};
    const models = [...new Set([...$("routes").querySelectorAll("input")].map(x => x.value.trim()).filter(Boolean))];
    $("pricing").replaceChildren(...models.map(model => {
      const row = node("div", {class: "price-row", "data-model": model}, node("strong", {text: model}));
      for (const [field, label] of [["input", "Вход"], ["output", "Выход"], ["cached_input", "Кэш входа"]]) row.append(node("label", {}, label, node("input", {type: "number", min: "0", step: "0.001", "data-field": field, value: merged[model]?.[field] ?? "", placeholder: "Не задано"})));
      return row;
    }));
  }
  function fillConfig() {
    const config = state.corpus.config || {};
    $("scope").value = config.scope || "";
    if (![...$("aspect").options].some(o => o.value === config.aspect) && config.aspect) $("aspect").append(node("option", {value: config.aspect, text: config.aspect}));
    $("aspect").value = config.aspect || "public_positions";
    $("languages").value = (config.languages || []).join(", ");
    $("country-terms").value = (config.country_terms || []).join(", ");
    $("date-from").value = config.date_from || ""; $("date-to").value = config.date_to || "";
    $("objects-json").value = JSON.stringify(config.objects || [], null, 2);
    $("scales-json").value = JSON.stringify(config.scales || [], null, 2);
    $("scales-approved").checked = config.scales_approved === true;
    $("routes").replaceChildren(...Object.entries(STAGES).map(([stage, title]) => {
      const input = node("input", {"data-stage": stage, value: config.routes?.[stage] || state.modelConfig.routes?.[stage] || (stage === "relevance" ? "gpt-4.1-nano" : "gpt-5.1"), spellcheck: "false"});
      input.addEventListener("change", () => renderPricing(readPricing()));
      return node("label", {}, title, input);
    }));
    renderPricing(config.pricing || {});
    $("config-summary").textContent = `Объектов: ${number(config.objects?.length)} · шкал: ${number(config.scales?.length)} · ${config.scales_approved ? "шкалы утверждены для корпуса" : "шкалы для корпуса не утверждены"}`;
    state.dirty = false; controls();
  }
  function parseArray(id, title) {
    let value;
    try { value = JSON.parse($(id).value || "[]"); } catch (_) { throw new Error(`${title}: проверьте JSON`); }
    if (!Array.isArray(value)) throw new Error(`${title}: ожидается массив JSON`);
    return value;
  }
  function configPayload() {
    const config = {...clone(state.corpus.config || {}), scope: $("scope").value.trim(), aspect: $("aspect").value,
      languages: $("languages").value.split(/[,;\n]/).map(x => x.trim()).filter(Boolean), date_from: $("date-from").value,
      country_terms: $("country-terms").value.split(/[,;\n]/).map(x => x.trim()).filter(Boolean),
      date_to: $("date-to").value, objects: parseArray("objects-json", "Объекты"), scales: parseArray("scales-json", "Шкалы"),
      scales_approved: $("scales-approved").checked, routes: {}, pricing: readPricing()};
    for (const input of $("routes").querySelectorAll("input")) { if (!input.value.trim()) throw new Error("Укажите модель для каждого этапа"); config.routes[input.dataset.stage] = input.value.trim(); }
    return config;
  }
  function safeUrl(value, download = false) {
    if (typeof value !== "string" || !value.trim()) return null;
    try {
      const url = new URL(value, location.origin);
      if (!["http:", "https:"].includes(url.protocol) || url.username || url.password) return null;
      if (download && (url.origin !== location.origin || !url.pathname.startsWith(path("/files/")))) return null;
      return url.href;
    } catch (_) { return null; }
  }
  async function renderDocuments() {
    const id = state.corpus.id, selected = state.selection;
    const query = new URLSearchParams({limit: "50", offset: String(state.offset)});
    if ($("document-status").value) query.set("status", $("document-status").value);
    const result = await api(path(`/documents?${query}`, id));
    if (selected !== state.selection) return;
    const docs = result.documents || []; state.total = result.total || 0;
    $("documents-total").textContent = `Найдено ${number(state.total)} · на странице до 50 материалов`;
    const statusNames = {unique: "Уникальный", exact_duplicate: "Точный дубль", empty: "Нет текста"};
    $("documents").replaceChildren(...docs.map(doc => {
      const source = safeUrl(doc.url);
      const title = node(source ? "a" : "span", {class: "document-title", text: doc.title || `Материал ${doc.source_row ?? doc.id}`, ...(source ? {href: source, target: "_blank", rel: "noopener noreferrer"} : {})});
      return node("tr", {}, node("td", {}, title, node("span", {class: "document-preview", text: String(doc.text || "").slice(0, 220)}), node("details", {}, node("summary", {text: "Полный текст"}), node("p", {text: doc.text || "Текст отсутствует"}))), node("td", {}, doc.published_at || "Дата не определена", node("br"), doc.source_name || "Источник не указан", doc.author ? node("div", {class: "muted", text: doc.author}) : null), node("td", {}, statusNames[doc.status] || doc.status || "—", doc.duplicate_of ? node("div", {class: "hint", text: `Дубль ${doc.duplicate_of}`}) : null));
    }));
    if (!docs.length) $("documents").append(node("tr", {}, node("td", {colspan: "3", text: "По этому фильтру документов нет"})));
    $("previous").disabled = state.offset === 0; $("next").disabled = state.offset + docs.length >= state.total;
    $("page-label").textContent = state.total ? `${number(state.offset + 1)}–${number(state.offset + docs.length)} из ${number(state.total)}` : "0 документов";
  }
  function schedule() {
    clearTimeout(state.timer);
    if (running()) state.timer = setTimeout(async () => {
      try { await refreshCurrent(false); } catch (error) { message(error.message, true); }
    }, 1500);
  }
  async function refreshCurrent(withDocuments = true) {
    if (!state.corpus) return;
    const id = state.corpus.id, selected = state.selection, wasRunning = running();
    const result = await api(path("", id));
    if (selected !== state.selection) return;
    state.corpus = result; renderProfile();
    if (withDocuments || (wasRunning && !running())) await renderDocuments();
    if (wasRunning && !running()) await refreshList();
    schedule();
  }
  async function select(id) {
    clearTimeout(state.timer); state.selection += 1;
    const selected = state.selection, result = await api(path("", id));
    if (selected !== state.selection) return;
    state.corpus = result; state.offset = 0;
    $("import-card").open = false; $("prepare-result").textContent = ""; $("downloads").replaceChildren();
    renderList(); renderProfile(); fillConfig(); await renderDocuments(); schedule();
    const url = new URL(location.href); url.searchParams.set("id", id); history.replaceState(null, "", url);
  }
  function stageNote() {
    $("stage-note").textContent = {relevance: "Проверка связи документа с выбранными людьми и областью исследования.", extract: "Используются прошедшие отбор материалы; факты сохраняются вместе с источниками.", describe: "Нужны накопленные факты. Подготовленные признаки требуют проверки и утверждения.", score: "Нужны проверяемые факты и явно утверждённые для этого корпуса шкалы. Отсутствие данных остаётся неизвестностью."}[$("stage").value];
  }
  $("import-form").addEventListener("submit", event => {
    event.preventDefault(); perform(async () => {
      let mapping;
      if ($("import-mapping").value.trim()) { try { mapping = JSON.parse($("import-mapping").value); } catch (_) { throw new Error("Проверьте JSON сопоставления столбцов"); } }
      const payload = {path: $("import-path").value.trim(), title: $("import-title").value.trim(), ...(mapping ? {mapping} : {}), ...($("import-study").value ? {study_id: $("import-study").value} : {})};
      const result = await api("/api/corpora", "POST", payload); await refreshList(); await select(result.id || result.corpus?.id); message("Корпус импортирован. Проверьте профиль и настройки; API не вызывался.");
    });
  });
  $("config-form").addEventListener("input", () => {state.dirty = true; $("config-summary").textContent = "Есть несохранённые настройки. Сохраните их перед подготовкой или запуском."; controls();});
  $("config-form").addEventListener("submit", event => {event.preventDefault(); perform(async () => {state.corpus = await api(path("/configure"), "POST", {config: configPayload()}); fillConfig(); renderProfile(); message("Настройки корпуса сохранены.");});});
  $("copy-study").addEventListener("click", () => perform(async () => {if (!$("config-study").value) throw new Error("Выберите исследование"); state.corpus = await api(path("/configure"), "POST", {study_id: $("config-study").value}); fillConfig(); renderProfile(); message("Настройки скопированы. Шкалы требуют отдельного утверждения для нового корпуса.");}));
  $("use-draft").addEventListener("click", () => {
    $("scales-json").value = $("draft-scales-json").value;
    $("scales-approved").checked = false; state.dirty = true;
    $("settings-card").open = true;
    $("config-summary").textContent = "Черновик перенесён. Проверьте шкалы и сохраните настройки. Утверждение не перенесено автоматически.";
    $("settings-card").scrollIntoView({behavior: "smooth", block: "start"}); controls();
  });
  $("prepare").addEventListener("click", () => perform(async () => {
    const limit = Number($("prepare-limit").value); if (!Number.isInteger(limit) || limit < 1 || limit > 5000) throw new Error("Укажите целый лимит от 1 до 5000 заданий");
    const result = await api(path("/prepare"), "POST", {stage: $("stage").value, limit});
    $("prepare-result").textContent = `Добавлено: ${number(result.created)} · оценка верхней границы: $${number(result.total_estimated_upper_usd)}`;
    await refreshCurrent(); message("Задания подготовлены. API ещё не запускался.");
  }));
  $("run-api").addEventListener("click", () => perform(async () => {
    const max_jobs = Number($("run-limit").value), budget_usd = Number($("run-budget").value);
    if (!Number.isInteger(max_jobs) || max_jobs < 1 || max_jobs > 1000 || !Number.isFinite(budget_usd) || budget_usd <= 0) throw new Error("Проверьте лимит 1–1000 заданий и положительный бюджет API");
    state.corpus = await api(path("/run"), "POST", {max_jobs, budget_usd}); renderProfile(); schedule();
  }));
  $("export").addEventListener("click", () => perform(async () => {
    const result = await api(path("/export"), "POST", {}); $("downloads").replaceChildren();
    for (const file of result.files || []) {const url = safeUrl(file.url, true); if (url) $("downloads").append(node("a", {class: "button secondary small", href: url, download: "", text: file.name}));}
    message(result.files?.length ? "Локальная выгрузка готова." : "Выгрузка пока не содержит файлов.");
  }));
  $("previous").addEventListener("click", () => perform(async () => {state.offset = Math.max(0, state.offset - 50); await renderDocuments();}));
  $("next").addEventListener("click", () => perform(async () => {state.offset += 50; await renderDocuments();}));
  $("document-status").addEventListener("change", () => perform(async () => {state.offset = 0; await renderDocuments();}));
  $("stage").addEventListener("change", stageNote);
  $("refresh").addEventListener("click", () => perform(async () => {await refreshList(); await refreshCurrent();}));
  async function init() {
    stageNote();
    const results = await Promise.allSettled([refreshList(), api("/api/studies")]);
    if (results[1].status === "fulfilled") {
      state.studies = results[1].value.studies || [];
      for (const id of ["import-study", "config-study"]) for (const study of state.studies) $(id).append(node("option", {value: study.id, text: study.title || study.id}));
    }
    if (results[0].status === "rejected") throw results[0].reason;
    const requested = new URL(location.href).searchParams.get("id");
    if (requested && state.corpora.some(c => c.id === requested)) await select(requested);
    else if (state.corpora.length) await select(state.corpora[0].id);
  }
  perform(init);
})();
