/** Reader-facing factor/cluster narrative, using one saved model only. */
import fs from "node:fs/promises";
import path from "node:path";
import { createHash } from "node:crypto";

const TEAL = "#166F73";
const MUTED = "#56646D";
const chunks = (items, size) => Array.from({ length: Math.ceil(items.length / size) },
  (_, index) => items.slice(index * size, (index + 1) * size));
const surname = name => String(name).trim().split(/\s+/).slice(1).join(" ") || String(name);

export async function buildInterpretiveSlides({ sourceDir, study, slide, text, table, lines,
  narrativeSlides, addImage }) {
  const config = study.presentation_config;
  const local = filename => {
    const absolute = path.resolve(sourceDir, filename);
    if (!absolute.startsWith(`${sourceDir}${path.sep}`)) throw new Error("Presentation input is outside output");
    return absolute;
  };
  const load = async filename => JSON.parse(await fs.readFile(local(filename), "utf8"));
  if (config.factor_count !== 4 || config.cluster_count !== 4) throw new Error("This presentation profile requires four factors and four groups");
  for (const field of ["objects", "scales"]) {
    const receipt = study.approvals?.[field];
    if (!receipt?.approved || receipt.fingerprint !== config.input_receipts?.[field])
      throw new Error("The approved research inputs changed. Rebuild the presentation profile.");
  }
  if (["scope", "aspect", "date_from", "date_to"].some(key => study[key] !== config.input_scope?.[key]))
    throw new Error("The research scope changed. Rebuild the presentation profile.");
  const matrixHash = createHash("sha256").update(await fs.readFile(local("matrix.csv"))).digest("hex");
  if (matrixHash !== config.matrix_sha256) throw new Error("The research matrix changed. Rebuild the presentation profile.");
  const model = await load(config.model_file);
  const narrativeBytes = await fs.readFile(local(config.narrative_file));
  if (createHash("sha256").update(narrativeBytes).digest("hex") !== config.narrative_sha256)
    throw new Error("The presentation narrative changed. Rebuild the presentation profile.");
  const narrative = JSON.parse(narrativeBytes.toString("utf8"));
  const manifest = await load(config.chart_manifest);
  if (model.factor_count !== 4 || model.cluster_count !== 4 || manifest.factor_count !== 4 || manifest.cluster_count !== 4)
    throw new Error("Presentation sources disagree with the requested four-factor model");
  const factorText = new Map(narrative.factors.map(item => [item.id, item]));
  const clusterText = new Map(narrative.clusters.map(item => [item.id, item]));
  if (factorText.size !== 4 || clusterText.size !== 4) throw new Error("All four factors and groups need interpretations");
  for (const cluster of model.clusters) {
    if (clusterText.get(cluster.id)?.count !== cluster.count) throw new Error("A presentation group size differs from the numerical result");
  }
  for (const [filename, expected] of Object.entries(manifest.source_sha256 || {})) {
    const actual = createHash("sha256").update(await fs.readFile(local(filename))).digest("hex");
    if (actual !== expected) throw new Error(`The plotted source changed: ${filename}`);
  }
  const charts = new Map(manifest.charts.map(item => [path.basename(item.file), item]));
  async function chartSlide(filename, title, caption) {
    const chart = charts.get(filename);
    if (!chart) throw new Error(`Missing four-factor chart: ${filename}`);
    const current = slide(title);
    await addImage(current, chart, { left: 58, top: 135, width: 1164, height: 474 });
    text(current, caption, 72, 619, 1136, 48, { size: 20, color: MUTED });
    return current;
  }
  {
    const current = slide("Политическое пространство России");
    text(current, "Публичные образы политиков, 2016–2026", 72, 180, 1120, 95, { size: 36 });
    text(current, "Четыре смысловые оси и четыре группы", 72, 312, 1120, 90,
      { size: 36, bold: true, color: TEAL });
    text(current, `${model.n_objects} участника исследования`, 72, 460, 1120, 68, { size: 31 });
    text(current, "Разведочное исследование оценённых публичных образов", 72, 601, 1120, 44,
      { size: 23, color: MUTED });
  }
  {
    const current = slide("Четыре измерения политического образа");
    table(current, ["Ось", "Одно направление", "Противоположное направление"],
      ["F2", "F3", "F1", "F4"].map(id => {
        const factor = factorText.get(id);
        return [id, factor.high, factor.low];
      }), { left: 72, top: 158, width: 1136, height: 400, fontSize: 24, padding: 9 }, [90, 523, 523]);
    text(current, "Каждая ось описывает отдельное различие. У одного политика могут сочетаться разные направления этих осей.",
      72, 604, 1136, 60, { size: 23, color: MUTED });
  }
  {
    const current = slide("Четыре группы публичных образов");
    table(current, ["Группа", "Число", "Представители"], model.clusters.map(cluster => {
      const n = clusterText.get(cluster.id);
      return [`${cluster.id} ${n.label}`, cluster.count, n.representatives.slice(0, 2).map(surname).join(", ")];
    }), { left: 72, top: 158, width: 1136, height: 395, fontSize: 24, padding: 8 }, [655, 145, 336]);
    text(current, "Цвета этих четырёх групп используются на всех картах политиков. Общность публичного стиля может сочетаться с различием политических позиций.",
      72, 598, 1136, 66, { size: 23, color: MUTED });
  }
  // Keep the complete approved dictionary on slides 4 and 5.
  const retained = new Set(model.loading_table.map(item => item.id));
  chunks(study.scales, 13).forEach((page, index) => {
    const current = slide(`Все шкалы публичного образа (${index + 1}/2)`);
    page.forEach((item, row) => {
      const top = 155 + row * 35;
      text(current, String(index * 13 + row + 1), 72, top, 54, 32,
        { size: 24, color: TEAL, bold: true });
      text(current, item.name + (retained.has(item.id) ? "" : " *"),
        136, top, 1072, 32, { size: 24 });
    });
    text(current, "* По этим девяти шкалам пока недостаточно оценок. На картах показаны остальные 17 шкал.",
      72, 624, 1136, 41, { size: 21, color: MUTED });
  });
  for (const id of ["F2", "F3", "F1", "F4"]) {
    const factor = factorText.get(id);
    const current = slide(`${id}. ${factor.label}`);
    text(current, factor.interpretation, 72, 158, 1136, 160, { size: 26 });
    text(current, "На одном полюсе", 72, 356, 520, 42, { size: 26, bold: true });
    text(current, "На другом полюсе", 668, 356, 535, 42, { size: 26, bold: true });
    text(current, factor.high, 72, 424, 520, 163, { size: 32, color: TEAL, bold: true });
    text(current, factor.low, 668, 424, 535, 163, { size: 32, color: TEAL, bold: true });
    if (factor.note) text(current, factor.note, 72, 625, 1136, 41, { size: 21, color: MUTED });
  }
  await chartSlide("traits_f2_f3.png", "Признаки в факторном пространстве",
    "Точки обозначают признаки, а не политиков. Показаны первые полюса шкал. Направления осей подписаны словами.");
  narrativeSlides(narrative.map_explanation.title, narrative.map_explanation.paragraphs.join("\n\n"));
  {
    const current = slide("Три примера чтения карты признаков");
    narrative.map_explanation.examples.forEach((example, index) => {
      const top = 152 + 153 * index;
      text(current, example.title, 72, top, 1136, 45, { size: 25, bold: true, color: TEAL });
      text(current, example.text, 72, top + 48, 1136, 99, { size: 23 });
    });
  }
  await chartSlide("traits_f1_f4.png", "Признаки: самостоятельность и управленческие приоритеты",
    "Другая плоскость раскрывает скрытое различие: практические результаты и социальное перераспределение расходятся по вертикали.");
  await chartSlide("objects_f2_f3.png", "Политики: внешние акценты и способ действия", narrative.object_map_caption);
  await chartSlide("objects_f1_f4.png", "Политики: распределение власти и обоснование программы",
    "Справа сильнее акцент на самостоятельности, слева — на вертикали. Сверху — на результатах, снизу — на идеологической и социальной программе.");
  await chartSlide("objects_tsne.png", "Карта соседства политиков",
    "Карта помогает увидеть ближайшее соседство профилей по всем четырём осям. Направления по горизонтали и вертикали самостоятельного политического смысла не имеют.");
  for (const cluster of model.clusters) {
    const n = clusterText.get(cluster.id);
    const current = slide(`${cluster.id}. ${n.label}`);
    text(current, `${cluster.count} участников`, 72, 144, 530, 43, { size: 29, color: TEAL, bold: true });
    text(current, n.interpretation, 72, 210, 530, 224, { size: 25 });
    text(current, "Близкие к центру группы", 72, 471, 530, 39, { size: 25, bold: true });
    text(current, n.representatives.join("\n"), 72, 527, 530, 111, { size: 24 });
    text(current, "Полный состав", 665, 145, 540, 40, { size: 25, bold: true });
    const members = cluster.members.map(item => item.name);
    if (members.length <= 16) {
      text(current, members.join("\n"), 665, 205, 540, 434, { size: 25 });
    } else {
      const halves = chunks(members, Math.ceil(members.length / 2));
      halves.forEach((half, index) => text(current, half.map(surname).join("\n"),
        665 + index * 280, 205, 260, 434, { size: 22 }));
    }
  }
  narrativeSlides("Содержание политического пространства", [narrative.short_conclusion,
    "Похожий способ действия может сочетаться с противоположными внешнеполитическими акцентами. Институциональная конкуренция и персональная конфронтация образуют разные сочетания публичных свойств.",
    "Общность профиля показывает сходство публичных акцентов. Политические программы и отношение к действующей власти внутри группы могут различаться."].join("\n\n"));
}
