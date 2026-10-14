/** Source-backed editable presentation export using the first-party Artifact Tool.
 * CLI: bundled-node presentation.mjs ABS_CONTENT_JSON ABS_OUTPUT_DIR
 * Artifact operation marking belongs to the calling service, before first run.
 */
import fs from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import { randomUUID } from "node:crypto";

const BUNDLED = "C:/Users/USER/.cache/codex-runtimes/codex-primary-runtime/dependencies";
const NODE_MODULES = process.env.SEMANTIC_NODE_MODULES || `${BUNDLED}/node/node_modules`;
const PYTHON = process.env.SEMANTIC_ARTIFACT_PYTHON || `${BUNDLED}/python/python.exe`;
const SKILL = process.env.SEMANTIC_PRESENTATION_SKILL ||
  "C:/Users/USER/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations";
const FONT = "Arial";
const INK = "#132126";
const TEAL = "#166F73";
const MUTED = "#56646D";
const LIGHT = "#E0E7E9";
const WIDTH = 1280;
const HEIGHT = 720;

const clean = value => String(value ?? "").replace(/\s+/g, " ").trim();
const short = (value, limit = 130) => {
  const text = clean(value);
  if (text.length <= limit) return text;
  let prefix = text.slice(0, limit - 1).trimEnd();
  const boundary = prefix.lastIndexOf(" ");
  if (boundary >= Math.floor(limit * .65)) prefix = prefix.slice(0, boundary);
  return `${prefix}…`;
};
const pct = value => `${(100 * Number(value || 0)).toFixed(1).replace(".", ",")}%`;
const num = (value, digits = 2) => Number(value || 0).toFixed(digits).replace(".", ",");
const signed = value => `${Number(value) >= 0 ? "+" : ""}${num(value)}`;
const chunks = (values, size) => Array.from({ length: Math.ceil(values.length / size) },
  (_, index) => values.slice(index * size, (index + 1) * size));
const summary = (value, limit = 285) => {
  const sentences = clean(value).split(/(?<=[.!?])\s+(?=[А-ЯЁA-Z])/);
  let result = "";
  for (const sentence of sentences) {
    if (result && result.length + sentence.length + 1 > limit) break;
    result = result ? `${result} ${sentence}` : sentence;
    if (result.length >= limit) break;
  }
  return short(result, limit);
};
const wrapText = (value, width) => {
  const words = clean(value).split(/\s+/).filter(Boolean);
  const lines = [];
  let line = "";
  for (const word of words) {
    if (line && line.length + word.length + 1 > width) {
      lines.push(line);
      line = "";
    }
    if (word.length > width) {
      if (line) { lines.push(line); line = ""; }
      for (let offset = 0; offset < word.length; offset += width) lines.push(word.slice(offset, offset + width));
    } else line = line ? `${line} ${word}` : word;
  }
  if (line) lines.push(line);
  return lines;
};

async function main() {
  const [contentArgument, outputArgument] = process.argv.slice(2);
  if (!contentArgument || !outputArgument) throw new Error("Usage: presentation.mjs CONTENT_JSON OUTPUT_DIR");
  const contentPath = path.resolve(contentArgument);
  const sourceDir = path.dirname(contentPath);
  const output = path.resolve(outputArgument);
  const content = JSON.parse(await fs.readFile(contentPath, "utf8"));
  const study = content.study;
  const analysis = content.analysis;
  if (!study || !analysis?.factors?.length || !analysis?.clusters?.length)
    throw new Error("The report content must contain study, analysis.factors and analysis.clusters");
  await fs.mkdir(output, { recursive: true });
  const build = path.join(output, ".privatebuild", `presentation-${randomUUID()}`);
  const previews = path.join(build, "previews");
  const draftDir = path.join(build, "draft");
  const validatedDir = path.join(build, "validated");
  await Promise.all([previews, draftDir, validatedDir].map(directory => fs.mkdir(directory, { recursive: true })));
  process.env.RUNTIME_NODE_MODULES = NODE_MODULES;
  let artifactTool;
  try {
    const runtimeRequire = createRequire(path.join(NODE_MODULES, ".semantic-resolver.cjs"));
    artifactTool = await import(pathToFileURL(runtimeRequire.resolve("@oai/artifact-tool")).href);
  } catch (error) {
    throw new Error(`Cannot load bundled @oai/artifact-tool from ${NODE_MODULES}. Set SEMANTIC_NODE_MODULES to the bundled package directory. ${error.message}`);
  }
  const { Presentation, PresentationFile, FileBlob } = artifactTool;
  const { finalizePresentation } = await import(pathToFileURL(path.join(SKILL, "container_tools", "artifact_tool_utils.mjs")).href);
  const presentation = Presentation.create({ slideSize: { width: WIDTH, height: HEIGHT } });
  const nativeTableSlides = [];
  const sources = [...new Set([
    ...(content.blocks || []).filter(block => block.type === "link").map(block => block.url),
    ...(study.objects || []).flatMap(object => (object.sources || []).map(source => typeof source === "string" ? source : source.url)),
  ].filter(value => typeof value === "string" && /^https?:\/\//.test(value)))];
  const commonNotes = `Числа: metadata.json, scores.csv, loadings.csv и утверждённая матрица оценок.\n` +
    `Изучаемое пространство: ${clean(study.scope || study.title)}.\n` +
    `Оценщик: ${clean(study.measurement_mode || "llm_exploratory")}.\n` +
    (study.example ? `Демонстрационный пример: ${clean(study.example_note || study.note || "ранее собранная матрица, а не новое исследование политиков")}.\n` : "") +
    `Исходные URL:\n${sources.join("\n")}`;

  function text(slide, value, left, top, width, height, options = {}) {
    const shape = slide.shapes.add({ geometry: "textbox", name: options.name || "text",
      position: { left, top, width, height }, fill: "none",
      line: { style: "solid", fill: "none", width: 0 } });
    shape.text = String(value ?? "");
    shape.text.style = { typeface: FONT, fontSize: options.size || 25, color: options.color || INK,
      bold: Boolean(options.bold), alignment: options.align || "left", verticalAlignment: "top",
      autoFit: "none", wrap: "square", insets: 0 };
    return shape;
  }

  function slide(title, note = "") {
    const current = presentation.slides.add();
    current.background.fill = "#FFFFFF";
    const titleSize = [38, 33, 30, 28, 26].find(size =>
      wrapText(title, Math.floor(1144 / (size * .56))).length * size * 1.22 <= 82) || 26;
    text(current, title, 68, 45, 1144, 82, { size: titleSize, bold: true });
    text(current, String(presentation.slides.items.length).padStart(2, "0"), 1170, 672, 42, 21,
      { size: 13, color: MUTED, align: "right" });
    current.speakerNotes.text = `${note}\n\n${commonNotes}`;
    return current;
  }

  function lines(current, values, left, top, width, size = 25, gap = 27) {
    let cursor = top;
    for (const value of values) {
      const count = Math.max(1, Math.ceil(clean(value).length / Math.max(18, width / (size * .52))));
      const height = count * size * 1.23;
      text(current, value, left, cursor, width, height, { size });
      cursor += height + gap;
    }
    return cursor;
  }

  function narrativeSlides(title, narrative) {
    const source = String(narrative || "").trim();
    const paragraphs = source.split(/\r?\n+/).map(part => part.trim()).filter(Boolean);
    const units = [];
    for (const [paragraphIndex, paragraph] of paragraphs.entries()) {
      const wrapped = wrapText(paragraph, 78);
      if (wrapped.length <= 13) {
        units.push({ lines: wrapped, paragraphBreak: paragraphIndex > 0 });
        continue;
      }
      // Keep normal paragraphs and sentences intact. Only a single sentence
      // longer than one slide falls back to line-wise continuation.
      const sentences = paragraph.split(/(?<=[.!?])\s+(?=[А-ЯЁA-Z])/);
      for (const [sentenceIndex, sentence] of sentences.entries()) {
        const pieces = chunks(wrapText(sentence, 78), 13);
        pieces.forEach((piece, pieceIndex) => units.push({ lines: piece,
          paragraphBreak: paragraphIndex > 0 && sentenceIndex === 0 && pieceIndex === 0 }));
      }
    }
    const pages = [];
    let pending = [];
    for (const unit of units) {
      const gap = pending.length && unit.paragraphBreak ? 1 : 0;
      if (pending.length && pending.length + gap + unit.lines.length > 13) {
        pages.push(pending);
        pending = [];
      }
      if (pending.length && unit.paragraphBreak) pending.push("");
      pending.push(...unit.lines);
    }
    if (pending.length) pages.push(pending);
    pages.forEach((page, index) => {
      const current = slide(index ? `${title} (продолжение)` : title, source);
      text(current, page.join("\n"), 72, 162, 1136, 430, { size: 25 });
    });
  }

  function orientedFactorProperties(factor, high) {
    const approved = new Map((study.scales || []).map(scale => [String(scale.id), scale]));
    const entries = [];
    for (const [key, positiveLoading] of [["positive_traits", true], ["negative_traits", false]]) {
      for (const trait of factor[key] || []) {
        const scale = approved.get(String(trait.id)) || {};
        const propertyKey = high === positiveLoading ? "positive" : "negative";
        const property = clean(trait[propertyKey] || scale[propertyKey] || (propertyKey === "positive" ? trait.name : ""));
        if (property) entries.push({ property, strength: Math.abs(Number(trait.loading)) });
      }
    }
    entries.sort((a, b) => b.strength - a.strength);
    const seen = new Set();
    return entries.filter(item => {
      const key = item.property.toLocaleLowerCase("ru");
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    }).slice(0, 3);
  }

  function table(current, headers, rows, geometry, widths) {
    const values = [headers, ...rows].map(row => row.map(value => String(value ?? "")));
    const result = current.tables.add({ rows: values.length, columns: headers.length,
      left: geometry.left, top: geometry.top, width: geometry.width, height: geometry.height,
      values, columnWidths: widths });
    result.borders.assign({ style: "solid", fill: LIGHT, width: 1 });
    for (let row = 0; row < values.length; row++) {
      for (let column = 0; column < headers.length; column++) {
        const cell = result.getCell(row, column);
        cell.fill = row === 0 ? TEAL : row % 2 ? "#FFFFFF" : "#F2F6F7";
        cell.text.style = { typeface: FONT, fontSize: geometry.fontSize || 21,
          color: row === 0 ? "#FFFFFF" : INK, bold: row === 0, verticalAlignment: "middle",
          alignment: column === 0 ? "left" : "center", insets: geometry.padding ?? 7, autoFit: "none", wrap: "square" };
      }
    }
    nativeTableSlides.push(presentation.slides.items.length);
    return result;
  }

  async function image(current, filename, geometry) {
    const chart = (analysis.charts || []).find(item => path.basename(item.file) === filename);
    if (!chart) throw new Error(`Required analysis chart is missing: ${filename}`);
    const filenameAbsolute = path.resolve(sourceDir, chart.file);
    if (!filenameAbsolute.startsWith(`${sourceDir}${path.sep}`)) throw new Error("Chart path is outside the report directory");
    const bytes = await fs.readFile(filenameAbsolute);
    current.images.add({ blob: bytes, contentType: "image/png", alt: chart.title || filename,
      fit: "contain", position: geometry });
    return chart;
  }

  const focus = clean(study.focus || study.focus_object_id).toLocaleLowerCase("ru");
  const focusIds = new Set((study.focus_object_ids || []).map(String));
  const focused = (analysis.object_profiles || []).filter(object => focusIds.has(object.id) ||
    (focus && (object.id.toLocaleLowerCase("ru") === focus || object.name.toLocaleLowerCase("ru").includes(focus))));
  const showScalesSlide = true;

  {
    const current = slide(short(study.title || "Структура семантического пространства", 135));
    lines(current, [short(study.scope || "", 220),
      `Период: ${clean(study.period_label) || `${clean(study.date_from)} – ${clean(study.date_to)}`}`], 72, 180, 1075, 29, 20);
    text(current, `${analysis.n_objects} объектов`, 72, 440, 345, 58, { size: 40, bold: true, color: TEAL });
    text(current, `${analysis.n_scales} шкал`, 455, 440, 345, 58, { size: 40, bold: true, color: TEAL });
    text(current, `${analysis.factor_count} факторов`, 838, 440, 345, 58, { size: 40, bold: true, color: TEAL });
    text(current, study.example ?
      "Технический пример: повторный анализ прежних оценок партий. Эти данные не являются новым исследованием политиков." :
      "Разведочное исследование оценённых публичных образов", 72, 570, 1100, 75, { size: 23, color: MUTED });
  }
  {
    const current = slide("Выборка и способ измерения", JSON.stringify(analysis.diagnostics?.dimension_selection || {}, null, 2));
    const coverage = analysis.diagnostics?.retained_coverage;
    const selectionMethod = typeof study.selection_method === "string" ? study.selection_method : study.selection_method?.description;
    lines(current, [short(selectionMethod || "Целевая выборка по утверждённым пользователем критериям", 230),
      study.legacy_import ? "Прежние оценки 0–10 преобразованы как x − 5. Два полюса отдельно не измерялись. Пропуски сохранены." :
        "Каждый полюс: от 0 до 5. Разность: от −5 до +5. Неизвестное остаётся пропуском.",
      "Gaussian FA и согласованное вращение varimax. Группы определяются по всем выбранным факторам.",
      `${analysis.cluster_count} групп. Описанная общая дисперсия: ${pct(analysis.variance_total)}.` +
        (coverage == null ? "" : ` Полнота матрицы: ${pct(coverage)}.`)], 72, 155, 1120, 26, 27);
  }
  if (showScalesSlide) {
    const current = slide("Измерительные шкалы", JSON.stringify(study.scales || [], null, 2));
    const examples = (study.scales || []).slice(0, 6);
    table(current, ["Шкала", "Положительный полюс", "Отрицательный полюс"],
      examples.map(item => [short(item.name, 32), short(item.positive, 32), short(item.negative, 32)]),
      { left: 72, top: 155, width: 1136, height: 410, fontSize: 20 }, [360, 388, 388]);
    text(current, `Примеры ${examples.length} из ${(study.scales || []).length} утверждённых шкал. Полный словарь и определения включены в отчёт.`,
      72, 600, 1130, 53, { size: 21, color: MUTED });
  }
  {
    const current = slide("Описанная дисперсия факторов");
    await image(current, "variance.png", { left: 65, top: 145, width: 810, height: 470 });
    text(current, pct(analysis.variance_total), 925, 225, 270, 78, { size: 57, bold: true, color: TEAL });
    text(current, "Общая описанная дисперсия выбранного решения", 925, 325, 265, 110, { size: 26 });
    text(current, "Доли FA описывают модель оценок. Их нельзя трактовать как доли населения.", 72, 625, 1136, 40, { size: 20, color: MUTED });
  }
  for (const factor of analysis.factors) {
    const current = slide(`${factor.id} ${short(factor.label || factor.name, 150)}`, JSON.stringify(factor, null, 2));
    text(current, `Доля наблюдаемой дисперсии: ${pct(factor.variance_share)}`, 72, 144, 1136, 37,
      { size: 25, bold: true, color: TEAL });
    text(current, summary(factor.interpretation, 280), 72, 196, 1136, 115, { size: 23 });
    for (const [high, left] of [[true, 72], [false, 667]]) {
      const properties = orientedFactorProperties(factor, high);
      text(current, high ? "Высокая координата" : "Низкая координата", left, 337, 535, 42,
        { size: 26, bold: true, color: TEAL });
      text(current, properties.length ? properties.map(item => short(item.property, 72)).join("\n\n") :
        "Нет достаточных заданных свойств для отдельного названия полюса", left, 395, 535, 213, { size: 22 });
    }
    const bipolar = (factor.positive_traits || []).length && (factor.negative_traits || []).length;
    text(current, bipolar ?
      "Свойства ориентированы по знакам нагрузок и заданным полюсам шкал. Знак факторной оси условен." :
      "Нагрузки преимущественно одного знака. Противоположные свойства взяты только из заданных шкал. Биполярность требует проверки.",
      72, 630, 1136, 45, { size: 18, color: MUTED });
  }
  for (const [filename, title, caption] of [
    ["objects_f1_f2.png", "Объекты в факторном пространстве", "F1/F2 показывают проекцию. Группы рассчитаны по всем выбранным факторам."],
    ["traits_f1_f2.png", "Признаки в факторном пространстве", "Точки обозначают шкалы. Их координаты — нагрузки, а не факторные баллы объектов."],
  ]) {
    const current = slide(title);
    await image(current, filename, { left: 105, top: 135, width: 1070, height: 485 });
    text(current, caption, 72, 632, 1136, 40, { size: 21, color: MUTED });
  }
  {
    const current = slide("Локальное соседство на картах t-SNE", JSON.stringify(analysis.diagnostics?.tsne || {}));
    await image(current, "tsne_objects.png", { left: 60, top: 175, width: 565, height: 390 });
    await image(current, "tsne_traits.png", { left: 655, top: 175, width: 565, height: 390 });
    text(current, "Объекты", 72, 136, 500, 34, { size: 25, bold: true, color: TEAL });
    text(current, "Семантические шкалы", 670, 136, 500, 34, { size: 25, bold: true, color: TEAL });
    text(current, "Оси t-SNE не являются факторами. Пустоты и глобальные расстояния на этой проекции требуют осторожной интерпретации.",
      72, 605, 1136, 64, { size: 22, color: MUTED });
  }
  {
    const current = slide("Сравнение числа групп", JSON.stringify(analysis.diagnostics?.cluster_selection || {}, null, 2));
    const rows = (analysis.cluster_candidates || []).map(item => [item.k,
      num(item.silhouette, 3), `${item.min_size}–${item.max_size}`,
      num(item.stability_mean_ari ?? item.stability?.mean_ari, 3)]);
    table(current, ["Групп k", "Силуэт", "Размеры групп", "Устойчивость ARI"], rows,
      { left: 72, top: 148, width: 1136, height: Math.min(430, 45 * (rows.length + 1)), fontSize: 22 },
      [200, 280, 328, 328]);
    text(current, `Рабочее решение: ${analysis.cluster_count} групп. Устойчивость оценена на подвыборках при фиксированных факторных координатах.`,
      72, 610, 1136, 57, { size: 22, color: MUTED });
  }
  narrativeSlides(`Почему выбрано решение с k=${analysis.cluster_count}`,
    analysis.cluster_selection_reason || analysis.diagnostics?.cluster_selection?.reason ||
    "Число групп зафиксировано в настройках этого расчёта. Это рабочая типология, которую следует сопоставить с содержанием альтернативных решений. Численные критерии приведены на предыдущем слайде.");
  for (const cluster of analysis.clusters) {
    const current = slide(`${cluster.id} ${short(cluster.label || cluster.name || "Группа", 150)}`,
      JSON.stringify(cluster, null, 2));
    text(current, `${cluster.count} объектов (${pct(cluster.count / analysis.n_objects)} выборки)`,
      72, 145, 1136, 40, { size: 26, bold: true, color: TEAL });
    text(current, summary(cluster.interpretation, 310), 72, 200, 1136, 135, { size: 24 });
    const members = [];
    const memberIds = new Set();
    for (const member of [...(cluster.representatives || []), ...(cluster.members || [])]) {
      const id = typeof member === "object" ? String(member.id || member.name) : String(member);
      if (memberIds.has(id)) continue;
      memberIds.add(id);
      members.push(typeof member === "object" ? member.name : member);
    }
    const shown = members.slice(0, 8);
    text(current, "Состав группы", 72, 361, 555, 37, { size: 25, bold: true });
    text(current, shown.map(name => short(name, 42)).join("\n"), 72, 411, 555, 212, { size: 21 });
    text(current, "Отличия от среднего", 680, 361, 525, 37, { size: 25, bold: true });
    text(current, (cluster.distinguishing_scales || []).slice(0, 3).map(item =>
      `${short(item.positive || item.name, 37)}\n${signed(item.difference)} σ`).join("\n\n"),
      680, 411, 525, 212, { size: 21 });
    text(current, members.length > shown.length ?
      `Показаны ${shown.length} участников, представители центра идут первыми. Ещё ${members.length - shown.length} объектов приведены в полном отчёте.` :
      "Показан полный состав группы, представители центра идут первыми. Основания и пограничные профили приведены в отчёте.",
      72, 628, 1136, 42, { size: 17, color: MUTED });
  }
  if (focused.length) {
    const object = focused[0];
    const current = slide(short(object.name, 115), focused.map(item => JSON.stringify(item, null, 2)).join("\n\n"));
    text(current, object.cluster_id, 72, 160, 180, 75, { size: 57, bold: true, color: TEAL });
    text(current, short(object.profile, 390), 300, 160, 880, 160, { size: 26 });
    const rows = Object.entries(object.factor_scores || {}).map(([id, value]) => [id, signed(value)]);
    table(current, ["Фактор", "Координата"], rows, { left: 72, top: 345, width: 480,
      height: Math.min(264, 32 * (rows.length + 1)), fontSize: 18, padding: 3 }, [240, 240]);
    text(current, "Ближайшие объекты", 640, 345, 540, 40, { size: 25, bold: true });
    text(current, (object.nearest_objects || []).slice(0, 4).map(item => short(item.name, 42)).join("\n"),
      640, 400, 540, 200, { size: 23 });
    text(current, `Измерено шкал: ${object.observed_scales}. Пропусков: ${object.missing_scales}.`,
      72, 632, 1136, 38, { size: 21, color: MUTED });
  }
  {
    const boundary = (analysis.object_profiles || []).filter(item => item.mixed_or_boundary).length;
    narrativeSlides("Интерпретация изученного пространства", [analysis.space_interpretation ||
      "Факторы описывают совместное изменение семантических шкал. Группы объединяют объекты с похожими сочетаниями этих свойств.",
      `Пограничных профилей: ${boundary} из ${analysis.n_objects}. Их свойства могут сочетать признаки нескольких групп.`,
      "Незаполненный сектор означает отсутствие такого профиля в выбранной выборке и корпусе. Международное сравнение требует единого словаря и сопоставимого отбора."].join("\n\n"));
  }
  {
    const current = slide("Ограничения и следующая проверка", (analysis.warnings || []).join("\n"));
    lines(current, ["Число факторов и групп остаётся исследовательским решением. Содержательная интерпретация дополняет численные критерии.",
      "Оценки LLM отражают модель публичного образа в источниках. Они требуют сравнения с независимыми человеческими оценками.",
      "Сходство партий или политиков не устанавливает общность реального электората. Частоты объектов в выборке не являются долями населения.",
      "Следующая проверка: новые объекты, иной корпус и независимые оценщики при фиксированных шкалах и параметрах проекции."],
      72, 154, 1120, 26, 27);
  }

  const candidatePath = path.join(draftDir, "candidate.pptx");
  await (await PresentationFile.exportPptx(presentation)).save(candidatePath);
  const validatedPath = path.join(validatedDir, "slides.pptx");
  await finalizePresentation({ workspaceDir: output, candidatePath, finalPath: validatedPath,
    pythonExecutable: PYTHON,
    integrityValidatorPath: path.join(SKILL, "container_tools", "inspect_presentation_package_integrity.py"),
    layoutValidatorPath: path.join(SKILL, "container_tools", "inspect_presentation_layout_geometry.py"),
    layoutArgs: ["--expected-slide-size-emu", "12192000,6858000", "--validate-bullet-geometry", "--validate-heading-fit",
      ...nativeTableSlides.flatMap(number => ["--require-native-table-slide", String(number)])],
    requiredNativeTableOwnerSlides: nativeTableSlides,
    requiredNativeChartOwnerSlides: [],
    fontPolicy: { basis: "design", families: [FONT] },
    verifyArtifactToolImport: true,
    receiptPath: path.join(build, "validation.json") });
  const imported = await PresentationFile.importPptx(await FileBlob.load(validatedPath));
  for (const [index, current] of imported.slides.items.entries()) {
    const blob = await imported.export({ slide: current, format: "png", scale: 1 });
    await fs.writeFile(path.join(previews, `slide-${String(index + 1).padStart(2, "0")}.png`),
      new Uint8Array(await blob.arrayBuffer()));
  }
  await fs.copyFile(validatedPath, path.join(output, "slides.pptx"));
  await fs.writeFile(path.join(build, "inventory.json"), JSON.stringify({ slides: imported.slides.items.length,
    previews, candidatePath, validatedPath, source: contentPath }, null, 2));
  console.log(JSON.stringify({ pptx: path.join(output, "slides.pptx"), slide_count: imported.slides.items.length,
    previews, validation: path.join(build, "validation.json") }));
}

main().catch(error => {
  console.error(`Presentation export failed: ${error.stack || error.message}`);
  process.exitCode = 1;
});
