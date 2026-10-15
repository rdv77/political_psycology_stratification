"use strict";

(() => {
  const $ = (selector) => document.querySelector(selector);
  const ACTIVE = new Set(["discovering", "generating_scales", "running"]);
  const COMPLETE = new Set(["complete", "completed", "done"]);
  const STEP_NAMES = ["Постановка", "Объекты", "Шкалы", "Анализ", "Отчёт"];
  const STATUS_NAMES = {
    draft: "Черновик", created: "Черновик", discovering: "Поиск объектов",
    await_objects_approval: "Проверка объектов", awaiting_objects: "Проверка объектов",
    objects_ready: "Проверка объектов", objects_approved: "Объекты утверждены",
    generating_scales: "Подготовка шкал", scales_ready: "Проверка шкал",
    await_scales_approval: "Проверка шкал", awaiting_scales: "Проверка шкал",
    scales_approved: "Готово к анализу", ready: "Готово к анализу",
    running: "Выполняется анализ", complete: "Анализ завершён",
    completed: "Анализ завершён", done: "Анализ завершён",
    error: "Требуется внимание", failed: "Требуется внимание",
    interrupted: "Этап прерван",
    await_factor_selection: "Выбор факторного решения",
    needs_measurement: "Недостаточно сопоставимых оценок",
  };
  const FORMAT_NAMES = { md: "MD", markdown: "MD", pdf: "PDF", pptx: "PPTX", docx: "DOCX", html: "HTML", zip: "ZIP", csv: "CSV", json: "JSON" };
  const STAGE_NAMES = { draft: "Постановка", discover: "Подбор объектов", objects: "Проверка объектов",
    sources: "Сбор материалов", scales: "Проверка шкал", ratings: "Оценивание объектов",
    analysis: "Факторный и кластерный анализ", cluster_review: "Сравнение группировок", coverage_completion: "Уточнение неполных профилей",
    factor_preview: "Сравнение факторных решений", factor_selection: "Короткий отчёт готов",
    measurement_review: "Аудит покрытия готов",
    interpretation: "Интерпретация", export: "Подготовка файлов", complete: "Отчёт подготовлен" };
  const state = {
    config: {}, studies: [], study: null, step: 1, busy: false, error: "",
    inputMode: "query", newDraft: null, objectDraft: null, scaleDraft: null,
    objectsDirty: false, scalesDirty: false, pollTimer: null, selectedRequest: 0,
    factorCount: "", clusterCount: "", connectionFailed: false,
  };

  // All API-provided text is inserted through textContent or input.value.
  // Links are restricted to HTTP(S); artifact links additionally remain in this study.
  function el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs)) {
      if (value === null || value === undefined || value === false) continue;
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = String(value);
      else if (key.startsWith("on") && typeof value === "function") node.addEventListener(key.slice(2), value);
      else if (["value", "checked", "disabled", "readOnly", "required", "selected"].includes(key)) node[key] = value;
      else node.setAttribute(key, String(value));
    }
    for (const child of children.flat(Infinity)) {
      if (child === null || child === undefined || child === false) continue;
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return node;
  }

  function clone(value) { return JSON.parse(JSON.stringify(value)); }
  function studyPath(suffix = "", id = state.study?.id) { return `/api/studies/${encodeURIComponent(id)}${suffix}`; }
  function unwrap(value) { return value?.study || value; }
  function approved(kind) { return state.study?.approvals?.[kind]?.approved === true; }
  function isActive(study = state.study) { return !!study && ACTIVE.has(study.status); }
  function isComplete(study = state.study) { return !!study && !ACTIVE.has(study.status) && !["error", "failed"].includes(study.status) && (COMPLETE.has(study.status) || (study.analysis && study.artifacts?.length > 0)); }
  function countObjects(study) { return study?.objects?.length ?? study?.object_count ?? 0; }

  function safeSourceUrl(value) {
    if (typeof value !== "string" || !value.trim()) return null;
    try {
      const url = new URL(value, location.origin);
      if (!["http:", "https:"].includes(url.protocol) || url.username || url.password) return null;
      return url.href;
    } catch (_) { return null; }
  }

  function safeArtifactUrl(value) {
    const url = safeSourceUrl(value);
    if (!url || !state.study) return null;
    const parsed = new URL(url);
    const prefix = `${studyPath("/files/")}`;
    if (parsed.origin !== location.origin || !parsed.pathname.startsWith(prefix)) return null;
    return `${parsed.pathname}${parsed.search}${parsed.hash}`;
  }

  async function api(path, method = "GET", data) {
    const options = { method, credentials: "same-origin", headers: { Accept: "application/json" }, cache: "no-store" };
    if (data !== undefined) {
      options.headers["Content-Type"] = "application/json";
      options.body = JSON.stringify(data);
    }
    let response;
    try { response = await fetch(path, options); }
    catch (_) { throw new Error("Локальный сервер недоступен. Проверьте, запущен ли он, и повторите действие."); }
    let result;
    try { result = await response.json(); }
    catch (_) { throw new Error(`Сервер вернул некорректный ответ (${response.status}).`); }
    if (!response.ok) {
      const detail = result.error || result.detail || result.message;
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail || `Ошибка сервера (${response.status})`));
    }
    return result;
  }

  function toast(message, error = false) {
    const region = $("#toast-region");
    const item = el("div", { class: `toast${error ? " error" : ""}`, text: message });
    region.replaceChildren(item);
    window.setTimeout(() => { if (item.isConnected) item.remove(); }, error ? 9000 : 4500);
  }

  function acceptStudy(value, resetDrafts = true) {
    const study = unwrap(value);
    if (!study || !study.id) throw new Error("Сервер не вернул исследование.");
    state.study = study;
    if (resetDrafts) {
      state.objectDraft = null;
      state.scaleDraft = null;
      state.objectsDirty = false;
      state.scalesDirty = false;
    }
    const index = state.studies.findIndex((item) => item.id === study.id);
    if (index >= 0) state.studies[index] = study;
    else state.studies.unshift(study);
    try { localStorage.setItem("semantic-space-study", study.id); } catch (_) { /* Local selection is optional. */ }
    schedulePoll();
    return study;
  }

  async function refreshStudies() {
    const result = await api("/api/studies");
    state.studies = result.studies || [];
    renderSidebar();
  }

  function inferStep(study) {
    if (isComplete(study)) return 5;
    if (study.status === "running" || study.approvals?.scales?.approved) return 4;
    if (study.scales?.length || study.status === "generating_scales" || study.approvals?.objects?.approved) return 3;
    return 2;
  }

  async function selectStudy(id) {
    const request = ++state.selectedRequest;
    clearTimeout(state.pollTimer);
    state.error = "";
    state.busy = true;
    renderSidebar();
    $("#main").replaceChildren(el("div", { class: "initial-loading" }, el("span", { class: "spinner" }), "Открываем исследование…"));
    try {
      const study = unwrap(await api(studyPath("", id)));
      if (request !== state.selectedRequest) return;
      acceptStudy(study);
      state.step = inferStep(study);
      state.factorCount = "";
      state.clusterCount = "";
    } catch (error) {
      if (request !== state.selectedRequest) return;
      state.study = null;
      state.error = error.message;
      state.step = 1;
    } finally {
      if (request === state.selectedRequest) {
        state.busy = false;
        render();
      }
    }
  }

  function schedulePoll() {
    clearTimeout(state.pollTimer);
    if (!isActive()) return;
    const id = state.study.id;
    state.pollTimer = window.setTimeout(async () => {
      if (state.study?.id !== id) return;
      try {
        const previousStatus = state.study.status;
        const study = unwrap(await api(studyPath("", id)));
        if (state.study?.id !== id) return;
        acceptStudy(study, !state.objectsDirty && !state.scalesDirty);
        if (state.connectionFailed) state.error = "";
        state.connectionFailed = false;
        if (previousStatus === "running" && isComplete(study)) {
          state.step = 5;
          toast("Анализ завершён. Отчёт готов к просмотру.");
        }
        render();
      } catch (error) {
        if (state.study?.id !== id) return;
        state.connectionFailed = true;
        state.error = error.message;
        render();
        // A failed request is not a failed analysis; retry without losing server state.
        clearTimeout(state.pollTimer);
        state.pollTimer = window.setTimeout(schedulePoll, 5000);
      }
    }, 1700);
  }

  async function perform(fn, successMessage = "") {
    if (state.busy) return;
    state.busy = true;
    state.error = "";
    render();
    try {
      await fn();
      if (successMessage) toast(successMessage);
    } catch (error) {
      state.error = error.message;
      toast(error.message, true);
    } finally {
      state.busy = false;
      render();
    }
  }

  function newStudy() {
    ++state.selectedRequest;
    clearTimeout(state.pollTimer);
    state.study = null;
    state.step = 1;
    state.busy = false;
    state.error = "";
    state.newDraft = null;
    state.inputMode = "query";
    state.objectDraft = null;
    state.scaleDraft = null;
    state.objectsDirty = false;
    state.scalesDirty = false;
    render();
    $("#main").focus({ preventScroll: true });
  }

  function renderSidebar() {
    $("#new-study").disabled = state.busy;
    $("#study-count").textContent = state.studies.length;
    const list = $("#study-list");
    if (!state.studies.length) {
      list.replaceChildren(el("p", { class: "sidebar-empty", text: "Здесь появятся ваши исследования и сохранённые результаты." }));
    } else {
      list.replaceChildren(...state.studies.map((study) => {
        const count = countObjects(study);
        const meta = [count ? `${count} объектов` : null, STATUS_NAMES[study.status] || "Исследование"].filter(Boolean).join(" · ");
        return el("button", {
          class: `study-item${state.study?.id === study.id ? " active" : ""}`,
          disabled: state.busy,
          onclick: () => selectStudy(study.id), "aria-current": state.study?.id === study.id ? "page" : null,
          title: study.title || study.scope || "Без названия",
        }, el("span", { class: `status-dot ${study.status || ""}` }),
        el("span", {}, el("span", { class: "study-item-name", text: study.title || study.scope || "Без названия" }), el("span", { class: "study-item-meta", text: meta })));
      }));
    }
    $("#provider-dot").className = `status-dot${state.config.provider_configured ? " connected" : ""}`;
    $("#provider-label").textContent = state.config.provider_configured ? "Модель подключена" : "Модель не подключена";
  }

  function statusBadge() {
    if (!state.study) return el("span", { class: "badge", text: "НОВОЕ ИССЛЕДОВАНИЕ" });
    const status = state.study.status;
    return el("span", { class: `badge${["error", "failed"].includes(status) ? " badge-danger" : isActive() ? " badge-warning" : ""}` },
      el("span", { class: `status-dot ${status}` }), STATUS_NAMES[status] || "Исследование");
  }

  function canVisit(step) {
    if (step === 1) return true;
    if (!state.study) return false;
    if (step === 2) return true;
    if (step === 3) return approved("objects") || !!state.study.scales?.length;
    if (step === 4) return approved("scales") || !!state.study.analysis || state.study.status === "running";
    return isComplete();
  }

  function stepDone(step) {
    if (!state.study) return false;
    if (step === 1) return true;
    if (step === 2) return approved("objects");
    if (step === 3) return approved("scales");
    return step === 4 ? isComplete() : false;
  }

  function steps() {
    return el("nav", { class: "steps", "aria-label": "Этапы исследования" }, STEP_NAMES.map((name, i) => {
      const step = i + 1;
      const done = stepDone(step);
      return el("button", {
        class: `step${state.step === step ? " active" : done ? " done" : ""}`,
        disabled: !canVisit(step) || state.busy,
        "aria-current": state.step === step ? "step" : null,
        onclick: () => { state.step = step; state.error = ""; renderMain(); },
      }, el("span", { class: "step-index", text: done && state.step !== step ? "✓" : step }), el("span", { text: name }));
    }));
  }

  function notice(message, kind = "", action) {
    return el("div", { class: `notice${kind ? ` notice-${kind}` : ""}`, role: kind === "error" ? "alert" : "status" },
      el("span", { class: "notice-icon", "aria-hidden": "true", text: kind === "error" ? "!" : kind === "warning" ? "◷" : "↳" }),
      el("p", { text: message }), action ? el("button", { class: "button-link", type: "button", text: action.label, onclick: action.fn }) : null);
  }

  function button(label, fn, options = {}) {
    return el("button", { class: `button ${options.primary ? "button-primary" : options.quiet ? "button-quiet" : "button-secondary"}${options.small ? " button-small" : ""}`, type: "button", disabled: options.disabled || state.busy, onclick: fn },
      options.loading ? el("span", { class: "spinner" }) : null, label);
  }

  function field(label, value, set, options = {}) {
    const id = options.id || `field-${Math.random().toString(36).slice(2, 9)}`;
    const input = el(options.multiline ? "textarea" : "input", {
      id, value: value ?? "", type: options.multiline ? null : options.type || "text",
      placeholder: options.placeholder, min: options.min, max: options.max, step: options.step,
      required: options.required, readOnly: options.readOnly, rows: options.rows,
      oninput: (event) => set(event.target.value), autocomplete: "off",
    });
    return el("div", { class: `field${options.full ? " full" : ""}` },
      el("label", { class: "field-label", for: id }, label, options.optional ? el("span", { class: "optional", text: "необязательно" }) : null), input,
      options.hint ? el("p", { class: "field-hint", text: options.hint }) : null);
  }

  function render() { renderSidebar(); renderMain(); }

  function renderMain() {
    const study = state.study;
    const main = $("#main");
    const headings = ["Новое исследование", "Проверьте список объектов", "Утвердите семантические шкалы", "Построение пространства", "Результаты исследования"];
    const descriptions = [
      "Задайте область и объекты. Программа предложит выборку, соберёт признаки и построит интерпретируемую модель.",
      "Уточните состав выборки, формулировки и источники. На следующем шаге программа подготовит общий словарь признаков.",
      "Проверьте смысл полюсов и определения. Каждый объект будет оценён по всему утверждённому набору шкал.",
      "Оценки объектов, факторные оси и сравнение группировок объединяются в воспроизводимый исследовательский результат.",
      "Факторные оси, профили групп и положение объектов. Полный отчёт и файлы исследования доступны ниже.",
    ];
    const title = state.step === 1 && study ? "Постановка исследования" : headings[state.step - 1];
    const heading = el("div", { class: "page-heading" },
      el("div", {}, el("div", { class: "eyebrow", text: study ? study.title || "СОХРАНЁННОЕ ИССЛЕДОВАНИЕ" : "ОТ ОБЪЕКТОВ К ЛАТЕНТНЫМ ПРИЗНАКАМ" }),
        el("h1", { text: title }), el("p", { class: "page-description", text: descriptions[state.step - 1] })),
      el("div", { class: "page-heading-actions" }, statusBadge()));
    const children = [heading, steps()];
    if (study?.example) children.push(notice("Пример на прежних данных; не результат исследования 70 российских политиков. Старые оценки 0–10 перенесены через x−5. Новых оценок и пользовательских утверждений нет.", "warning"));
    if (state.error) children.push(notice(state.error, "error"));
    const backendError = study?.error;
    if (backendError && backendError !== state.error) children.push(notice(typeof backendError === "string" ? backendError : JSON.stringify(backendError), "error"));
    if (state.step === 1) children.push(study ? renderScopeSummary() : renderNewForm());
    else if (state.step === 2) children.push(renderObjects());
    else if (state.step === 3) children.push(renderScales());
    else if (state.step === 4) children.push(renderRun());
    else children.push(renderReport());
    main.replaceChildren(...children);
  }

  function createNewDraft() {
    const today = new Date();
    const previous = new Date(today.getFullYear() - 10, today.getMonth(), today.getDate());
    const date = (value) => `${value.getFullYear()}-${String(value.getMonth() + 1).padStart(2, "0")}-${String(value.getDate()).padStart(2, "0")}`;
    return { title: "", scope: "", aspect: "Публичный имидж", query: "", objects_text: "", date_from: date(previous), date_to: date(today), focus: "", target_count: "30" };
  }

  function renderNewForm() {
    const draft = state.newDraft || (state.newDraft = createNewDraft());
    const update = (key) => (value) => { draft[key] = value; };
    const form = el("form", { class: "panel", onsubmit: submitNewStudy });
    form.append(el("section", { class: "panel-section" },
      el("div", { class: "section-heading" }, el("span", { class: "section-number", text: "01" }), "Что исследуем"),
      el("div", { class: "field-grid" },
        field("Название исследования", draft.title, update("title"), { full: true, optional: true, placeholder: "Например: Политическое пространство России" }),
        field("Предметная область", draft.scope, update("scope"), { required: true, placeholder: "Политическое пространство России", hint: "Укажите область и географию сравнения." }),
        field("Измеряемый аспект", draft.aspect, update("aspect"), { required: true, placeholder: "Публичный имидж", hint: "Например, имидж, программные позиции или ценности." }),
        field("Начало периода", draft.date_from, update("date_from"), { type: "date", required: true }),
        field("Конец периода", draft.date_to, update("date_to"), { type: "date", required: true }))));
    const input = el("section", { class: "panel-section" },
      el("div", { class: "section-heading" }, el("span", { class: "section-number", text: "02" }), "Какие объекты сравним"),
      el("div", { class: "segmented-control", role: "group", "aria-label": "Способ задания объектов" },
        el("button", { class: `segment${state.inputMode === "query" ? " active" : ""}`, type: "button", "aria-pressed": state.inputMode === "query", onclick: () => { state.inputMode = "query"; renderMain(); }, text: "Описать критерии поиска" }),
        el("button", { class: `segment${state.inputMode === "list" ? " active" : ""}`, type: "button", "aria-pressed": state.inputMode === "list", onclick: () => { state.inputMode = "list"; renderMain(); }, text: "Вставить готовый список" })));
    input.append(state.inputMode === "query"
      ? field("Запрос для подбора объектов", draft.query, update("query"), { required: true, multiline: true, placeholder: "Известные российские политики последних десяти лет: представители ведущих партий и публичные независимые политики.", hint: "Опишите тип объектов и основания включения. Список можно будет изменить перед анализом." })
      : field("Список объектов", draft.objects_text, update("objects_text"), { required: true, multiline: true, rows: 6, placeholder: "Один объект на строку\nНазвание первого объекта\nНазвание второго объекта", hint: "Имена и названия будут приведены к единому реестру. Вы проверите его перед продолжением." }));
    if (state.inputMode === "query") input.append(el("div", { style: "margin-top:17px;max-width:210px" },
      field("Ориентир по числу объектов", draft.target_count, update("target_count"), { type: "number", min: 12, max: 120, hint: "В MVP — от 12 до 120. Итоговую выборку определяете вы." })));
    form.append(input);
    form.append(el("section", { class: "panel-section" },
      el("div", { class: "section-heading" }, el("span", { class: "section-number", text: "03" }), "Фокус итогового отчёта"),
      field("Объект для подробного разбора", draft.focus, update("focus"), { optional: true, placeholder: "Имя политика или название организации", hint: "Отчёт подробнее покажет его профиль, ближайших соседей и положение среди групп." })));
    const submit = el("button", { class: "button button-primary", type: "submit", disabled: state.busy },
      state.busy ? el("span", { class: "spinner" }) : null, "Создать и подобрать объекты", el("span", { "aria-hidden": "true", text: "→" }));
    form.append(el("div", { class: "form-footer" }, el("p", { class: "form-footer-note", text: "Список объектов и шкалы потребуют вашего утверждения." }), submit));
    const aside = el("aside", {}, workflowAside());
    if (state.config.seed_id) aside.append(el("button", { class: "example-button", type: "button", onclick: () => selectStudy(state.config.seed_id) }, "Открыть подготовленную выборку", el("span", { "aria-hidden": "true", text: "↗" })));
    aside.append(el("div", { class: "aside-below" }, el("p", { text: state.config.provider_configured ? "Модель подключена. Оценивание выполняется на сервере; результаты и источники сохраняются вместе с исследованием." : "Для нового сбора и оценивания настройте модель в серверном .env. Сохранённые исследования доступны без подключения." }),
      !state.config.provider_configured ? el("button", { class: "button-link", style: "margin-top:9px", type: "button", onclick: openSettings, text: "Настройки подключения" }) : null));
    return el("div", { class: "content-columns" }, form, aside);
  }

  function workflowAside() {
    const items = [
      ["Выборка объектов", "Найденные объекты и источники — на проверку вам."],
      ["Общий словарь", "Нормализация признаков, синонимы и полюса шкал."],
      ["Факторные оси", "Ограниченный набор интерпретируемых измерений."],
      ["Группы и отчёт", "Общность объектов, карты и подробная интерпретация."],
    ];
    return el("div", { class: "aside-card" }, el("h3", { text: "Путь исследования" }),
      el("p", { text: "Пространство восстанавливается через свойства его объектов." }),
      el("ol", { class: "process-list" }, items.map(([name, note], i) => el("li", {}, el("span", { class: "process-marker", text: i + 1 }), el("div", {}, el("div", { class: "process-name", text: name }), el("p", { class: "process-note", text: note }))))),
      el("hr", { class: "aside-divider" }), el("div", { class: "aside-label", text: "ДВА ЭТАПА ПРОВЕРКИ" }),
      el("p", { class: "aside-small", text: "Вы утверждаете объекты и шкалы. Затем программа выполняет анализ и собирает результат." }));
  }

  async function submitNewStudy(event) {
    event.preventDefault();
    const draft = state.newDraft;
    if (draft.date_from > draft.date_to) { state.error = "Дата начала должна быть не позднее даты окончания периода."; renderMain(); return; }
    const payload = { ...draft, measurement_model: "independent_poles", title: draft.title.trim() || draft.scope.trim(), target_count: Number(draft.target_count) || 30,
      query: state.inputMode === "query" ? draft.query : "", objects_text: state.inputMode === "list" ? draft.objects_text : "" };
    await perform(async () => {
      acceptStudy(await api("/api/studies", "POST", payload));
      state.step = 2;
      acceptStudy(await api(studyPath("/discover"), "POST", {}));
    });
  }

  function renderScopeSummary() {
    const study = state.study;
    const values = [ ["Предметная область", study.scope], ["Измеряемый аспект", study.aspect], ["Период", study.period_label || `${study.date_from || "—"} — ${study.date_to || "—"}`], ["Объект для подробного разбора", study.focus || "Не задан"], ["Критерии поиска", study.query || "Использован готовый список объектов"], ["Состояние выборки", `${countObjects(study)} объектов · ${study.scales?.length || 0} шкал`] ];
    return el("div", { class: "content-columns" },
      el("div", {}, el("div", { class: "details-grid" }, values.map(([label, value]) => el("div", { class: "metadata-card" }, el("div", { class: "metadata-label", text: label }), el("div", { class: "metadata-value", text: value || "Не указано" })))),
        el("div", { style: "margin-top:22px;display:flex;gap:10px;flex-wrap:wrap" }, button("Перейти к объектам →", () => { state.step = 2; renderMain(); }, { primary: true }), button("Новое исследование", newStudy))),
      el("aside", {}, workflowAside()));
  }

  function getObjectDraft() {
    if (!state.objectDraft) state.objectDraft = clone(state.study.objects || []).map((obj) => ({ ...obj, _include: true }));
    return state.objectDraft;
  }

  function dirtyObjects() { state.objectsDirty = true; updateSelectionCount(); }

  function updateSelectionCount() {
    const node = $("#selection-count");
    if (node && state.objectDraft) node.textContent = `${state.objectDraft.filter((obj) => obj._include !== false).length} объектов выбрано`;
  }

  function renderObjects() {
    const study = state.study;
    if (study.status === "discovering") return el("div", { class: "panel" }, loadingState("Собираем список объектов", study.progress?.message || "Ищем объекты по вашим критериям и сохраняем основания включения и источники."));
    const objects = getObjectDraft();
    if (!objects.length) return el("div", { class: "panel" }, emptyState("Пока нет объектов", "Запустите подбор по заданным критериям или добавьте объекты вручную.", [
      button("Подобрать объекты", () => perform(async () => { acceptStudy(await api(studyPath("/discover"), "POST", {})); }), { primary: true }),
      button("Добавить вручную", addObject),
    ]));
    const box = el("div");
    if (approved("objects")) box.append(notice("Список утверждён. Изменение выборки потребует повторного утверждения и нового анализа.", "", { label: "К шкалам →", fn: () => { state.step = 3; renderMain(); } }));
    box.append(el("div", { class: "editor-toolbar" },
      el("p", {}, `${objects.length} объектов в списке. `, el("span", { text: "Снимите отметку, чтобы исключить объект." })),
      el("div", { class: "toolbar-actions" }, button("＋ Добавить объект", addObject, { small: true }), button("Обновить подбор", () => perform(async () => { acceptStudy(await api(studyPath("/discover"), "POST", {})); }), { small: true, disabled: isActive() }))));
    const table = el("table", { class: "objects-table" },
      el("thead", {}, el("tr", {}, el("th", {}, el("span", { class: "sr-only", text: "Включить" })), el("th", { text: "ОБЪЕКТ / КАТЕГОРИЯ" }), el("th", { text: "РОЛЬ И ОСНОВАНИЕ ВКЛЮЧЕНИЯ" }), el("th", { text: "ИСТОЧНИКИ" }))),
      el("tbody", {}, objects.map(renderObjectRow)));
    box.append(el("div", { class: "panel table-container" }, table));
    box.append(el("div", { class: "sticky-actions" },
      el("div", {}, el("div", { id: "selection-count", class: "selection-count", text: `${objects.filter((obj) => obj._include !== false).length} объектов выбрано` }), el("p", { class: "sticky-note", text: "Далее — общий словарь семантических признаков." })),
      el("div", { class: "editor-actions" }, button("Сохранить список", () => saveObjects(false), { disabled: isActive() }), button("Утвердить и подготовить шкалы →", () => saveObjects(true), { primary: true, disabled: isActive() }))));
    return box;
  }

  function addObject() {
    getObjectDraft().push({ id: `manual-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`, name: "", category: "", role: "", inclusion_reason: "Добавлен пользователем", sources: [], _include: true });
    state.objectsDirty = true;
    renderMain();
    const inputs = document.querySelectorAll(".object-name");
    inputs[inputs.length - 1]?.focus();
  }

  function renderObjectRow(object) {
    const row = el("tr", { class: object._include === false ? "excluded" : "" });
    const update = (key) => (event) => { object[key] = event.target.value; dirtyObjects(); };
    const checkbox = el("input", { type: "checkbox", checked: object._include !== false, "aria-label": `Включить объект ${object.name || "без названия"}`, onchange: (event) => { object._include = event.target.checked; row.classList.toggle("excluded", !event.target.checked); dirtyObjects(); } });
    row.append(el("td", {}, checkbox));
    row.append(el("td", {}, el("input", { class: "table-input object-name", value: object.name || "", placeholder: "Название или имя", "aria-label": "Название объекта", oninput: update("name") }),
      el("input", { class: "table-input subtle", value: object.category || "", placeholder: "Категория", "aria-label": "Категория объекта", oninput: update("category") })));
    row.append(el("td", {}, el("textarea", { class: "table-textarea", rows: 2, value: object.role || "", placeholder: "Публичная роль", "aria-label": "Роль объекта", oninput: update("role") }),
      el("details", { style: "margin-top:6px" }, el("summary", { class: "source-summary", text: "Основание включения" }), el("textarea", { class: "table-textarea", style: "margin-top:7px", rows: 2, value: object.inclusion_reason || "", "aria-label": "Основание включения", oninput: update("inclusion_reason") }))));
    if (!Array.isArray(object.sources)) object.sources = [];
    const sources = el("td");
    object.sources.slice(0, 3).forEach((source) => {
      const href = safeSourceUrl(source.url);
      const title = source.title || source.url || "Источник";
      sources.append(href ? el("a", { class: "source-link", href, target: "_blank", rel: "noopener noreferrer", title, text: title }) : el("span", { class: "source-link muted", text: title }));
    });
    if (!object.sources.length) sources.append(el("p", { class: "field-hint", text: "Источники пока не указаны" }));
    const sourceEditor = el("div", { class: "sources-editor" });
    const refreshSources = () => {
      sourceEditor.replaceChildren(...object.sources.map((source, i) => el("div", { class: "source-edit-row" },
        el("input", { class: "table-input", value: source.title || "", placeholder: "Название источника", "aria-label": "Название источника", oninput: (event) => { source.title = event.target.value; dirtyObjects(); } }),
        el("input", { class: "table-input", value: source.url || "", placeholder: "https://…", type: "url", "aria-label": "Ссылка на источник", oninput: (event) => { source.url = event.target.value; dirtyObjects(); } }),
        el("button", { class: "icon-button", type: "button", text: "×", "aria-label": "Удалить источник", onclick: () => { object.sources.splice(i, 1); dirtyObjects(); refreshSources(); } }))));
    };
    refreshSources();
    sources.append(el("details", {}, el("summary", { class: "source-summary", text: `Редактировать источники (${object.sources.length})` }), sourceEditor,
      el("button", { class: "button-link", style: "margin-top:7px", type: "button", text: "＋ Добавить источник", onclick: () => { object.sources.push({ title: "", url: "" }); dirtyObjects(); refreshSources(); } })));
    row.append(sources);
    return row;
  }

  async function saveObjects(approve) {
    const selected = getObjectDraft().filter((object) => object._include !== false).map(({ _include, ...object }) => ({ ...object, name: object.name.trim() }));
    if (!selected.length || selected.some((object) => !object.name)) { state.error = "Выберите хотя бы один объект и заполните все названия включённых объектов."; renderMain(); return; }
    await perform(async () => {
      acceptStudy(await api(studyPath("/objects"), "PUT", { objects: selected }));
      if (approve) {
        acceptStudy(await api(studyPath("/approve-objects"), "POST", {}));
        state.step = 3;
        if (!state.study.scales?.length) acceptStudy(await api(studyPath("/generate-scales"), "POST", {}));
      }
    }, approve ? "Список объектов утверждён." : "Список объектов сохранён.");
  }

  function getScaleDraft() {
    if (!state.scaleDraft) state.scaleDraft = clone(state.study.scales || []);
    return state.scaleDraft;
  }

  function renderScales() {
    const study = state.study;
    if (study.status === "generating_scales") return el("div", { class: "panel" }, loadingState("Формируем общий словарь", study.progress?.message || "Извлекаем определения, объединяем близкие синонимы и проверяем смысл противоположных полюсов."));
    const scales = getScaleDraft();
    if (!scales.length) return el("div", { class: "panel" }, emptyState("Подготовим шкалы для всей выборки", "Шкалы будут нормализованы к выбранному аспекту. Все объекты оцениваются по одному общему набору признаков.", [
      button("Подготовить шкалы", () => perform(async () => { acceptStudy(await api(studyPath("/generate-scales"), "POST", {})); }), { primary: true, disabled: !approved("objects") }),
      button("Добавить вручную", addScale),
    ]));
    const box = el("div");
    if (approved("scales")) box.append(notice("Шкалы утверждены. Изменение их смысла потребует повторного утверждения и нового анализа.", "", { label: "К анализу →", fn: () => { state.step = 4; renderMain(); } }));
    box.append(el("div", { class: "editor-toolbar" }, el("p", { text: `${scales.length} шкал · алиасы сохраняют исходные формулировки без повторного веса в матрице.` }),
      el("div", { class: "toolbar-actions" }, button("＋ Добавить шкалу", addScale, { small: true }), button("Сформировать заново", () => perform(async () => { acceptStudy(await api(studyPath("/generate-scales"), "POST", {})); }), { small: true, disabled: !approved("objects") || isActive() }))));
    box.append(el("div", { class: "scales-grid" }, scales.map(renderScaleCard)));
    box.append(el("p", { class: "field-hint", style: "margin-top:16px", text: study.example ? "В этом примере импортирована только старая оценка x из диапазона 0–10: координата = x−5. Полюса не оценивались независимо, сумма полюсов неизвестна. Ноль означает прежнюю оценку 5. Это не заново оценённые пары антонимов." : study.measurement_model === "independent_poles" ? "Каждое свойство 0–5 входит в анализ отдельной переменной. Два свойства могут сочетаться; неизвестность одного не отменяет оценку другого. Можно добавить однополюсную шкалу. Полюса факторов определяются по нагрузкам после анализа." : "Полюса оцениваются отдельно от 0 до 5. Разность задаёт координату −5…+5, а исходные оценки позволяют различить невыраженность обоих свойств и их сочетание. Недостаток данных сохраняется отдельно." }));
    box.append(el("div", { class: "sticky-actions" }, el("div", {}, el("div", { class: "selection-count", text: `${scales.length} шкал в общем словаре` }), el("p", { class: "sticky-note", text: "После утверждения можно запускать оценивание и анализ." })),
      el("div", { class: "editor-actions" }, button("Сохранить шкалы", () => saveScales(false), { disabled: isActive() }), button("Утвердить шкалы →", () => saveScales(true), { primary: true, disabled: !approved("objects") || isActive() }))));
    return box;
  }

  function addScale() {
    getScaleDraft().push({ id: `scale-${Date.now()}`, name: "", positive: "", negative: "", definition: "", aliases: [], anchors: {} });
    state.scalesDirty = true;
    renderMain();
    const inputs = document.querySelectorAll(".scale-title-input");
    inputs[inputs.length - 1]?.focus();
  }

  function renderScaleCard(scale, index) {
    const set = (key) => (value) => { scale[key] = value; state.scalesDirty = true; };
    const independent = state.study.measurement_model === "independent_poles";
    const single = scale.kind === "unipolar";
    const title = el("input", { class: "scale-title-input", value: scale.name || "", placeholder: "Название шкалы", "aria-label": `Название шкалы ${index + 1}`, oninput: (event) => set("name")(event.target.value) });
    const card = el("section", { class: "scale-card", "aria-label": `Шкала ${index + 1}` },
      el("div", { class: "scale-card-header" }, el("span", { class: "scale-number", text: String(index + 1).padStart(2, "0") }), title,
        el("button", { class: "icon-button", type: "button", text: "×", "aria-label": `Удалить шкалу ${scale.name || index + 1}`, onclick: () => { getScaleDraft().splice(index, 1); state.scalesDirty = true; renderMain(); } })),
      independent && el("label", {}, "Тип шкалы", el("select", { class: "table-input", "aria-label": "Тип шкалы", onchange: (event) => { set("kind")(event.target.value); renderMain(); } },
        el("option", { value: "paired", selected: !single, text: "Два независимых свойства" }), el("option", { value: "unipolar", selected: single, text: "Одно свойство" }))),
      single ? el("p", { class: "field-hint", text: "Измеряется выраженность свойства, указанного в названии и определении. Противоположный признак не требуется." }) : el("div", { class: "scale-poles" },
        el("div", {}, el("div", { class: "pole-label", text: independent ? "СВОЙСТВО Б · 0–5" : "− ОТРИЦАТЕЛЬНЫЙ ПОЛЮС" }), el("input", { class: "table-input", value: scale.negative || "", placeholder: "Второе свойство", "aria-label": "Второе свойство", oninput: (event) => set("negative")(event.target.value) })),
        el("div", {}, el("div", { class: "pole-label positive", text: independent ? "СВОЙСТВО А · 0–5" : "+ ПОЛОЖИТЕЛЬНЫЙ ПОЛЮС" }), el("input", { class: "table-input", value: scale.positive || "", placeholder: "Выраженное свойство", "aria-label": "Первое свойство", oninput: (event) => set("positive")(event.target.value) }))),
      el("div", { class: "scale-range", "aria-hidden": "true" }, independent ? "0" : "−5", el("span", { class: "scale-range-line" }), independent ? "5 · каждое свойство отдельно" : "+5"),
      field("Определение и границы смысла", scale.definition, set("definition"), { multiline: true, rows: 3, placeholder: "Как понимать признак и по каким проявлениям его оценивать" }),
      field("Исходные формулировки / алиасы", Array.isArray(scale.aliases) ? scale.aliases.join("; ") : String(scale.aliases || ""), (value) => { scale.aliases = value.split(/[;\n]/).map((item) => item.trim()).filter(Boolean); state.scalesDirty = true; }, { placeholder: "Близкие формулировки через точку с запятой" }));
    const anchors = flattenAnchors(scale.anchors);
    if (anchors.length) card.append(el("details", { class: "anchor-details" }, el("summary", { text: "Якоря оценивания" }), anchors.map((text) => el("p", { text }))));
    return card;
  }

  function flattenAnchors(value, prefix = "") {
    if (value === null || value === undefined) return [];
    if (typeof value !== "object") return [`${prefix ? `${prefix}: ` : ""}${value}`];
    return Object.entries(value).flatMap(([key, item]) => flattenAnchors(item, prefix ? `${prefix} · ${key}` : key));
  }

  async function saveScales(approve) {
    const scales = clone(getScaleDraft()).map((scale) => ({ ...scale, name: (scale.name || "").trim() }));
    if (scales.length < 6 || scales.length > 80 || scales.some((scale) => !scale.name || !scale.definition?.trim() || (scale.kind !== "unipolar" && (!scale.positive?.trim() || !scale.negative?.trim())))) {
      state.error = "В MVP нужно от 6 до 80 шкал. Заполните название и определение; для парной шкалы — оба свойства.";
      renderMain(); return;
    }
    await perform(async () => {
      acceptStudy(await api(studyPath("/scales"), "PUT", { scales }));
      if (approve) { acceptStudy(await api(studyPath("/approve-scales"), "POST", {})); state.step = 4; }
    }, approve ? "Шкалы утверждены. Исследование готово к запуску." : "Шкалы сохранены.");
  }

  function loadingState(title, message) {
    return el("div", { class: "empty-state" }, el("div", { class: "empty-icon" }, el("span", { class: "spinner" })), el("h2", { text: title }), el("p", { text: message }),
      el("p", { class: "field-hint", text: "Можно переключиться на другое исследование. Программа продолжит работу на сервере." }));
  }

  function emptyState(title, message, actions = []) {
    return el("div", { class: "empty-state" }, el("div", { class: "empty-icon", "aria-hidden": "true", text: "⌁" }), el("h2", { text: title }), el("p", { text: message }),
      el("div", { style: "display:flex;justify-content:center;gap:9px;flex-wrap:wrap" }, actions));
  }

  function renderRun() {
    const study = state.study;
    const previewOnly = study.delivery_mode === "factor_preview";
    const measurementFailed = study.analysis?.measurement_status === "insufficient_data";
    const running = study.status === "running";
    const complete = isComplete();
    const stopped = ["error", "failed", "interrupted"].includes(study.status);
    const progress = study.progress || {};
    const missing = [];
    if (!approved("objects")) missing.push("список объектов");
    if (!approved("scales")) missing.push("семантические шкалы");
    const current = Number(progress.current) || 0;
    const total = Number(progress.total) || 0;
    const percent = total > 0 ? Math.min(100, Math.max(0, Math.round(current / total * 100))) : 0;
    const main = el("section", { class: "panel run-panel" },
      el("div", { class: "run-eyebrow", text: running ? "ИССЛЕДОВАНИЕ В ПРОЦЕССЕ" : complete ? "ГОТОВЫЙ РЕЗУЛЬТАТ" : stopped ? "ТРЕБУЕТСЯ ВНИМАНИЕ" : "ПАРАМЕТРЫ ПОДГОТОВЛЕНЫ" }),
      el("h2", { text: running ? "Строим семантическое пространство" : complete ? "Анализ завершён" : stopped ? "Выполнение остановлено" : "Всё готово к анализу" }),
      el("p", { class: "run-message", text: running ? progress.message || "Оценивание объектов и анализ выполняются на сервере." : complete ? (previewOnly ? "Сравните факторные решения в коротком отчёте. Расширенный отчёт и презентация появятся после выбора решения." : "Результаты сохранены. Откройте отчёт, сравните оси и профили групп или скачайте материалы исследования.") : stopped ? "Причина остановки указана выше. После её устранения можно запустить этап повторно; сохранённые исходные данные доступны." : previewOnly ? "Программа оценит объекты и сравнит допустимые решения с 2–8 факторами. Первый результат — короткий отчёт для выбора числа факторов." : "Программа оценит объекты по общим шкалам, выделит факторные оси и сравнит варианты группировки." }));
    if (missing.length) main.append(notice(`До запуска нужно утвердить: ${missing.join(" и ")}.`, "warning", { label: "Проверить →", fn: () => { state.step = !approved("objects") ? 2 : 3; renderMain(); } }));
    if (complete && measurementFailed) {
      main.querySelector("h2").textContent = "Аудит данных завершён";
      main.querySelector(".run-message").textContent = "Сопоставимых измерений недостаточно для факторного анализа. Короткий отчёт показывает покрытие и основания исключений; выбор числа факторов пока недоступен.";
    }
    if (running || complete) main.append(el("div", { class: "progress-track", role: "progressbar", "aria-label": "Ход анализа", "aria-valuemin": 0, "aria-valuemax": 100, "aria-valuenow": complete ? 100 : total ? percent : null },
      el("div", { class: `progress-fill${running && !total ? " indeterminate" : ""}`, style: complete ? "width:100%" : total ? `width:${percent}%` : null })),
      el("div", { class: "progress-caption" }, el("span", { text: STAGE_NAMES[progress.stage] || (complete ? "Отчёт подготовлен" : "Выполняется этап") }), el("span", { text: complete ? "Готово" : total ? `${current} / ${total}` : "В процессе" })));
    const stageIndex = stagePosition(progress.stage);
    const stages = measurementFailed && complete
      ? [["Оценивание объектов", "Исходные предложения модели"], ["Проверка оснований", "Согласованность с выписками"], ["Аудит покрытия", "Недостаточно сопоставимых измерений"]]
      : previewOnly
      ? [["Оценивание объектов", "Общий словарь и источники"], ["Проверка измерения", "Покрытие и пропуски"], ["Сравнение факторов", "Решения с 2–8 осями"], ["Короткий отчёт", "Выбор решения пользователем"]]
      : [ ["Оценивание объектов", "Общий словарь и основания оценок"], ["Проверка измерения", "Покрытие, полюса и пропуски"], ["Выделение факторов", "Размерность, нагрузки и баллы"], ["Сравнение группировок", "Устойчивость и интерпретируемость"], ["Интерпретация", "Оси, группы и выбранный объект"], ["Подготовка отчёта", "Карты и файлы для скачивания"] ];
    main.append(el("div", { class: "run-stages" }, stages.map(([name, note], i) => {
      const done = complete || (running && i < stageIndex);
      const active = running && i === stageIndex;
      return el("div", { class: `run-stage${done ? " done" : active ? " active" : ""}` }, el("span", { class: "run-stage-dot", text: done ? "✓" : active ? "" : i + 1 }),
        el("span", { text: name }), el("span", { class: "run-stage-meta", text: done ? "Готово" : active ? "В процессе" : note }));
    })));
    main.append(el("div", { class: "run-start" }, complete
      ? button("Открыть отчёт →", () => { state.step = 5; renderMain(); }, { primary: true })
      : button(running ? "Анализ выполняется" : "Запустить оценивание и анализ →", runAnalysis, { primary: true, disabled: running || missing.length > 0, loading: running || state.busy }),
      !running && complete ? button("Пересчитать", runAnalysis, { quiet: true, disabled: missing.length > 0 }) : null));
    if (study.logs?.length) {
      const lines = study.logs.slice(-30).map((item) => typeof item === "string" ? item : item.message ? `${item.time || item.timestamp || item.at || ""} ${item.message}`.trim() : JSON.stringify(item));
      main.append(el("details", { class: "logs-details" }, el("summary", { text: "Журнал выполнения" }), el("pre", { class: "logs-pre", text: lines.join("\n") })));
    }
    const summary = el("div", { class: "run-side-card" }, el("h3", { text: "Состав исследования" }),
      el("dl", { class: "definition-list" }, [["Объекты", countObjects(study)], ["Шкалы", study.scales?.length || 0], ["Аспект", study.aspect || "Не указан"], ["Подтверждения", `${[approved("objects"), approved("scales")].filter(Boolean).length} из 2`],
        ...(study.budget_limit_usd != null ? [["Бюджет", `$${Number(study.budget_limit_usd).toFixed(2)}`]] : []),
        ...(study.usage?.estimated_usd != null ? [["Расход, оценка", `$${Number(study.usage.estimated_usd).toFixed(2)}`]] : [])]
        .map(([key, value]) => el("div", {}, el("dt", { text: key }), el("dd", { text: value })))));
    const advanced = el("details", { class: "advanced-settings" }, el("summary", { text: "Параметры анализа" }),
      field("Число факторов", state.factorCount, (value) => { state.factorCount = value; }, { type: "number", min: 2, max: 8, placeholder: "Автоматический подбор" }),
      field("Число кластеров", state.clusterCount, (value) => { state.clusterCount = value; }, { type: "number", min: 2, max: 10, placeholder: "Сравнить варианты" }),
      el("p", { class: "advanced-note", text: "Пустые поля оставляют автоматический выбор. Решение оценивается по устойчивости, качеству и содержательной интерпретации." }));
    summary.append(previewOnly ? el("p", { class: "advanced-note", text: "Сейчас сравниваются все допустимые решения с 2–8 факторами. Выбор решения — после короткого отчёта." }) : advanced);
    const aside = el("aside", {}, summary, el("div", { class: "aside-below" }, el("p", { text: "Модельные оценки — разведочный материал. Сходство профилей не доказывает совпадение электоратов, а пустая область карты — наличие общественного спроса." })));
    return el("div", { class: "run-layout" }, main, aside);
  }

  function stagePosition(stage) {
    const value = String(stage || "").toLowerCase();
    if (/report|export|отч[её]т|презентац/.test(value)) return 5;
    if (/interpret|интерпрет/.test(value)) return 4;
    if (/cluster|групп|кластер/.test(value)) return 3;
    if (/factor|фактор|analysis|анализ/.test(value)) return 2;
    if (/valid|диагност|провер/.test(value)) return 1;
    return 0;
  }

  async function runAnalysis() {
    const factor = state.factorCount ? Number(state.factorCount) : null;
    const cluster = state.clusterCount ? Number(state.clusterCount) : null;
    if ((factor !== null && (!Number.isInteger(factor) || factor < 2 || factor > 8)) || (cluster !== null && (!Number.isInteger(cluster) || cluster < 2 || cluster > 10))) {
      state.error = "В MVP число факторов: целое от 2 до 8; число кластеров: от 2 до 10. Допустимый максимум зависит от данных. Для автоматического выбора оставьте поля пустыми.";
      renderMain(); return;
    }
    await perform(async () => {
      acceptStudy(await api(studyPath("/run"), "POST", { factor_count: factor, cluster_count: cluster }));
      state.step = state.study.status === "running" ? 4 : inferStep(state.study);
    });
  }

  function renderReport() {
    const study = state.study;
    const analysis = study.analysis || {};
    if (!study.artifacts?.length) return el("div", { class: "panel" }, emptyState("Отчёт ещё не подготовлен", "После завершения анализа здесь появятся карты, интерпретация пространства и файлы исследования.", [button("К анализу →", () => { state.step = 4; renderMain(); }, { primary: true })]));
    if (study.delivery_mode === "factor_preview") return renderFactorPreview(study, analysis);
    const container = el("div");
    const variance = Number(analysis.variance_total);
    const varianceText = Number.isFinite(variance) ? `${(variance <= 1 ? variance * 100 : variance).toLocaleString("ru-RU", { maximumFractionDigits: 1 })}%` : "—";
    const stats = [["Объектов", countObjects(study), "В общей выборке"], ["Факторных осей", analysis.factor_count ?? "—", "Интерпретируемые измерения"], ["Групп", analysis.cluster_count ?? "—", "Выбранное решение"], ["Объяснённая дисперсия", analysis.variance_total == null ? "—" : varianceText, "В выбранной факторной модели"]];
    container.append(el("div", { class: "stats-grid" }, stats.map(([name, value, note]) => el("div", { class: "stat-card" }, el("div", { class: "stat-label", text: name }), el("div", { class: "stat-value", text: value }), el("div", { class: "stat-note", text: note })))));
    const artifacts = study.artifacts.map((artifact) => ({ ...artifact, safeUrl: safeArtifactUrl(artifact.url) }));
    const report = artifacts.find((artifact) => artifact.format?.toLowerCase() === "html" && artifact.safeUrl && /report/i.test(artifact.url)) || artifacts.find((artifact) => artifact.format?.toLowerCase() === "html" && artifact.safeUrl);
    const downloads = el("div", { class: "downloads", "aria-label": "Скачать материалы исследования" });
    const formatOrder = ["md", "pdf", "pptx", "docx", "html", "zip"];
    const primaryDownloadUrls = new Set();
    formatOrder.forEach((format) => {
      const artifact = artifacts.find((item) => (item.format?.toLowerCase() === format || (format === "md" && item.format === "markdown")) && item.safeUrl);
      if (artifact) {
        primaryDownloadUrls.add(artifact.safeUrl);
        downloads.append(el("a", { class: "button download-button", href: artifact.safeUrl, download: "", title: artifact.label || `Скачать ${FORMAT_NAMES[format]}`, text: `↓ ${FORMAT_NAMES[format]}` }));
      }
      else downloads.append(el("button", { class: "button download-button", type: "button", disabled: true, title: "Файл пока не создан", text: FORMAT_NAMES[format] }));
    });
    const toolbar = el("div", { class: "report-toolbar" }, el("div", {}, el("h3", { text: "Отчёт о семантическом пространстве" }), el("p", { text: study.focus ? `Подробный разбор: ${study.focus}` : "Карты, факторы, профили групп и ограничения" })), downloads);
    const panel = el("section", { class: "panel" }, toolbar);
    if (report) panel.append(el("iframe", { class: "report-frame", src: report.safeUrl, title: "Полный отчёт с картами и интерпретацией", sandbox: "allow-same-origin", loading: "lazy", referrerpolicy: "no-referrer" }));
    else panel.append(emptyState("Отчёт доступен в файлах", "Скачайте подготовленные материалы. Предпросмотр HTML для этого исследования не создан."));
    panel.append(el("div", { class: "report-warning" }, "Модель описывает объекты выбранной выборки и периода. Положения на картах и названия групп следует читать вместе с исходными шкалами и диагностикой устойчивости."));
    if (state.config.markdown_note) panel.append(el("div", { class: "report-warning", text: state.config.markdown_note }));
    container.append(panel);
    const extras = artifacts.filter((artifact) => artifact.safeUrl && !primaryDownloadUrls.has(artifact.safeUrl));
    const footer = el("div", { style: "margin-top:18px;display:flex;gap:9px;align-items:center;flex-wrap:wrap" });
    if (report) footer.append(el("a", { class: "button button-small", href: report.safeUrl, target: "_blank", rel: "noopener noreferrer", text: "Открыть отчёт отдельно ↗" }));
    extras.forEach((artifact) => footer.append(el("a", { class: "button button-small", href: artifact.safeUrl, download: "", text: `↓ ${artifact.label || FORMAT_NAMES[artifact.format] || artifact.format || "Файл"}` })));
    footer.append(button("Параметры и пересчёт", () => { state.step = 4; renderMain(); }, { small: true, quiet: true }));
    container.append(footer);
    return container;
  }

  function renderFactorPreview(study, analysis) {
    const insufficient = analysis.measurement_status === "insufficient_data";
    const container = el("div", { class: "panel" },
      el("h2", { text: insufficient ? "Аудит сопоставимости данных" : "Сравнение факторных решений" }),
      el("p", { text: insufficient ? "Факторные оси не оценивались: матрица пока не прошла заранее заданные требования к покрытию. Короткий отчёт показывает, каких измерений не хватает." : "Это первый этап. Сравните смысл осей и доли дисперсии; выбранное решение станет основой расширенного отчёта и презентации." }));
    const links = el("div", { class: "downloads" });
    for (const artifact of study.artifacts) {
      const url = safeArtifactUrl(artifact.url);
      if (url) links.append(el("a", { class: "button download-button", href: url, download: "", text: `↓ ${artifact.label || artifact.format}` }));
    }
    container.append(links);
    const percent = (value) => Number.isFinite(Number(value)) ? `${(100 * Number(value)).toLocaleString("ru-RU", { maximumFractionDigits: 2 })}%` : "—";
    const theme = analysis.russia_review || {};
    const themeNames = {
      insufficient_measurement: "Недостаточно измерений для вывода об отдельном факторе",
      numeric_candidates_pending_audit: "Есть численные кандидаты; требуется проверка содержания и устойчивости",
      conditional_two_indicator_candidates: "Есть условные кандидаты с двумя индикаторами",
      no_numeric_candidate: "В просмотренных решениях подходящий численный кандидат не найден",
    };
    if (theme.configured_ids?.length) container.append(el("div", { class: "report-warning" },
      el("strong", { text: "Отношение к России. " }),
      `${themeNames[theme.status] || theme.status}. В словаре ${theme.configured_ids.length} шкал; в пригодной матрице ${theme.retained_ids?.length || 0}. Порог показа кандидата — 2% общей дисперсии.`));
    if (insufficient) for (const chart of analysis.charts || []) {
      const url = safeArtifactUrl(`${studyPath("/files/")}${chart.file}`);
      if (url) container.append(el("img", { src: url, alt: chart.title, loading: "lazy", style: "width:100%;height:auto;margin-top:16px" }));
    }
    for (const model of analysis.models || []) {
      const card = el("section", { class: "run-side-card", style: "margin-top:20px" },
        el("h3", { text: `${model.q} фактора: ${percent(model.variance_total)} общей дисперсии` }));
      for (const factor of model.factors || []) card.append(el("p", {},
        el("strong", { text: `${factor.id}: ${factor.name || factor.label}. ` }),
        `${percent(factor.variance_share)}. ${factor.interpretation || ""}`));
      const chart = (analysis.charts || []).find((item) => item.q === model.q);
      const chartUrl = chart && safeArtifactUrl(`${studyPath("/files/")}${chart.file}`);
      if (chartUrl) card.append(el("img", { src: chartUrl, alt: chart.title, loading: "lazy", style: "width:100%;height:auto;margin-top:12px" }));
      container.append(card);
    }
    for (const limitation of analysis.limitations || []) container.append(el("p", { class: "report-warning", text: limitation }));
    container.append(button("Вернуться к анализу", () => { state.step = 4; renderMain(); }, { quiet: true }));
    return container;
  }

  function openSettings() {
    const configured = !!state.config.provider_configured;
    const body = $("#settings-content");
    body.replaceChildren(el("p", { text: "Ключи и параметры модели задаются на локальном сервере. Интерфейс не запрашивает и не хранит API-ключи в браузере." }),
      notice(configured ? "Модель настроена на сервере. Можно собирать и оценивать новые объекты." : "Модель ещё не подключена. Добавьте настройки провайдера в серверный файл .env и перезапустите сервер.", configured ? "" : "warning"));
    const details = [];
    if (state.config.provider) details.push(["Провайдер", state.config.provider]);
    if (state.config.model) details.push(["Модель", state.config.model]);
    if (state.config.search_configured !== undefined) details.push(["Поиск источников", state.config.search_configured ? "Подключён" : "Не подключён"]);
    if (details.length) body.append(el("dl", { class: "definition-list", style: "margin-top:17px" }, details.map(([key, value]) => el("div", {}, el("dt", { text: key }), el("dd", { text: value })))));
    body.append(el("p", { style: "margin-top:17px;font-size:11px", text: "Просмотр исходного исследования и скачивание уже созданных результатов доступны без подключения модели." }));
    $("#settings-dialog").showModal();
  }

  async function init() {
    $("#new-study").addEventListener("click", newStudy);
    $("#settings-button").addEventListener("click", openSettings);
    $("#top-settings").addEventListener("click", openSettings);
    $("#close-settings").addEventListener("click", () => $("#settings-dialog").close());
    $("#settings-dialog").addEventListener("click", (event) => { if (event.target === $("#settings-dialog")) $("#settings-dialog").close(); });
    const [configResult, studyResult] = await Promise.allSettled([api("/api/config"), api("/api/studies")]);
    if (configResult.status === "fulfilled") state.config = configResult.value;
    else state.error = configResult.reason.message;
    if (studyResult.status === "fulfilled") state.studies = studyResult.value.studies || [];
    else state.error = studyResult.reason.message;
    render();
    let previous;
    try { previous = localStorage.getItem("semantic-space-study"); } catch (_) { /* No persistence required for browser selection. */ }
    if (previous && state.studies.some((study) => study.id === previous)) await selectStudy(previous);
    // Start on a clear new-study screen; seed research stays visible in the sidebar.
    document.addEventListener("visibilitychange", () => { if (!document.hidden && isActive()) schedulePoll(); });
  }

  init().catch((error) => { state.error = error.message; state.busy = false; render(); });
})();
