/** Editable, reader-facing Rwanda deck from an audited slide specification. */
import fs from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import { createHash, randomUUID } from "node:crypto";

const BUNDLE = "C:/Users/USER/.cache/codex-runtimes/codex-primary-runtime/dependencies";
const MODULES = process.env.SEMANTIC_NODE_MODULES || `${BUNDLE}/node/node_modules`;
const PYTHON = process.env.SEMANTIC_ARTIFACT_PYTHON || `${BUNDLE}/python/python.exe`;
const SKILL = process.env.SEMANTIC_PRESENTATION_SKILL ||
  "C:/Users/USER/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations";
const FONT = "Arial", INK = "#142B35", TEAL = "#176D77", GREY = "#56656E";
const W = 1280, H = 720;
const digest = bytes => createHash("sha256").update(bytes).digest("hex");

async function main() {
  const [inputArg, outputArg] = process.argv.slice(2);
  if (!inputArg || !outputArg) throw new Error("Usage: presentation_rwanda.mjs SLIDE_CONTENT_JSON OUTPUT_DIR");
  const input = path.resolve(inputArg), output = path.resolve(outputArg);
  const spec = JSON.parse(await fs.readFile(input, "utf8"));
  if (spec.factor_count !== 3 || !spec.slides?.length) throw new Error("A selected three-axis narrative is required");
  for (const [filename, hash] of Object.entries(spec.source_sha256 || {})) {
    const file = path.resolve(path.dirname(input), filename);
    if (!file.startsWith(path.dirname(input) + path.sep) || digest(await fs.readFile(file)) !== hash)
      throw new Error(`The plotted numerical source changed: ${filename}`);
  }
  const build = path.join(output, ".privatebuild", `rwanda-${randomUUID()}`);
  const previews = path.join(build, "previews"), valid = path.join(build, "validated");
  await fs.mkdir(previews, { recursive: true });
  await fs.mkdir(valid, { recursive: true });
  process.env.RUNTIME_NODE_MODULES = MODULES;
  const resolver = createRequire(path.join(MODULES, ".rwanda-resolver.cjs"));
  const { Presentation, PresentationFile, FileBlob } = await import(pathToFileURL(resolver.resolve("@oai/artifact-tool")).href);
  const { finalizePresentation, applyPresentationChartFont } = await import(pathToFileURL(
    path.join(SKILL, "container_tools/artifact_tool_utils.mjs")).href);
  const p = Presentation.create({ slideSize: { width: W, height: H } });
  const tables = [], charts = [];

  function text(s, value, box, size = 27, options = {}) {
    const shape = s.shapes.add({ geometry: "textbox", name: options.name || "Narrative text",
      position: box, fill: "none", line: { fill: "none", width: 0 } });
    shape.text = String(value);
    shape.text.style = { typeface: FONT, fontSize: size, bold: Boolean(options.bold),
      color: options.color || INK, alignment: options.align || "left", verticalAlignment: "top",
      autoFit: "none", wrap: "square", insets: 0 };
    return shape;
  }
  function table(s, page) {
    const values = [page.headers, ...page.rows].map(row => row.map(v => String(v ?? "")));
    const t = s.tables.add({ rows: values.length, columns: values[0].length,
      left: 66, top: 145, width: 1148, height: page.height || 462,
      columnWidths: page.widths, values });
    t.borders.assign({ style: "solid", fill: "#D6E0E3", width: .8 });
    for (let r = 0; r < values.length; r++) {
      for (let c = 0; c < values[r].length; c++) {
        const cell = t.getCell(r, c);
        cell.fill = r === 0 ? TEAL : r % 2 ? "#FFFFFF" : "#F3F6F7";
        cell.text.style = { typeface: FONT, fontSize: page.fontSize || 23,
          color: r === 0 ? "#FFFFFF" : INK, bold: r === 0,
          alignment: "left", verticalAlignment: "middle", insets: page.padding ?? 9,
          autoFit: "none", wrap: "square" };
      }
    }
    tables.push(p.slides.items.length);
  }
  for (const [index, page] of spec.slides.entries()) {
    const s = p.slides.add();
    s.background.fill = "#FFFFFF";
    s.speakerNotes.text = [page.notes || "", ...(page.sources || []).map(x => `Источник: ${x}`)].join("\n\n");
    const titleSize = page.title.length > 82 ? 33 : page.title.length > 57 ? 37 : 42;
    text(s, page.title, { left: 66, top: 40, width: 1148, height: 91 }, titleSize, { bold: true });
    text(s, String(index + 1), { left: 1170, top: 682, width: 44, height: 18 }, 13, { color: GREY, align: "right" });
    if (page.kind === "cover") {
      text(s, page.subtitle, { left: 66, top: 210, width: 1100, height: 130 }, 44, { color: TEAL, bold: true });
      text(s, page.body, { left: 66, top: 405, width: 1110, height: 125 }, 30);
      text(s, "Публичные политические акценты 2021–2026", { left: 66, top: 609, width: 1110, height: 38 }, 24, { color: GREY });
    } else if (page.kind === "table") {
      table(s, page);
    } else if (page.kind === "chart") {
      const chartConfig = structuredClone(page.chart.config);
      const overrides = (chartConfig.series || []).map(series => {
        const labels = series.dataLabelOverrides || [];
        delete series.dataLabelOverrides;
        return labels;
      });
      const chart = s.charts.add(page.chart.type, {
        position: { left: 58, top: 138, width: 1162, height: 473 },
        chartFill: "#FFFFFF", plotAreaFill: "#FFFFFF", ...chartConfig,
      });
      applyPresentationChartFont(chart, { fontFamily: FONT });
      for (let i = 0; i < overrides.length; i++) {
        const series = chart.series.getItemAt(i);
        series.dataLabels.textStyle.typeface = FONT;
        for (const label of overrides[i]) {
          const override = series.dataLabelOverrides.add(label.idx);
          override.text = label.text;
          override.position = label.position;
          override.showCategoryName = true;
          override.showValue = false;
          override.textStyle.typeface = FONT;
          override.textStyle.fontSize = chartConfig.series[i].dataLabels?.textStyle?.fontSize || 14;
          override.textStyle.fill = INK;
        }
      }
      charts.push(index + 1);
    } else if (page.kind === "quote") {
      text(s, `«${page.quote}»`, { left: 84, top: 170, width: 1100, height: 163 }, 39, { color: TEAL, bold: true });
      text(s, page.attribution, { left: 84, top: 340, width: 1090, height: 76 }, 24, { color: GREY });
      text(s, page.body, { left: 84, top: 446, width: 1090, height: 164 }, 27);
    } else {
      const paragraphs = page.paragraphs || [];
      const gap = page.gap || 25;
      let top = 160;
      for (const para of paragraphs) {
        const t = typeof para === "string" ? { text: para } : para;
        const size = t.size || page.fontSize || 28;
        const lines = t.lines || Math.max(1, Math.ceil(t.text.length / (1145 / (size * .53))));
        const height = lines * size * 1.27 + 8;
        text(s, t.text, { left: 72, top, width: 1136, height }, size, { bold: t.bold, color: t.color });
        top += height + gap;
      }
      if (top > 656) throw new Error(`Narrative exceeds the intended canvas on slide ${index + 1}: ${top}`);
    }
    if (page.caption) text(s, page.caption, { left: 66, top: 620, width: 1148, height: 49 }, 20, { color: GREY });
  }
  const candidate = path.join(build, "candidate.pptx"), final = path.join(valid, "slides.pptx");
  await (await PresentationFile.exportPptx(p)).save(candidate);
  await finalizePresentation({ workspaceDir: output, candidatePath: candidate, finalPath: final,
    pythonExecutable: PYTHON,
    integrityValidatorPath: path.join(SKILL, "container_tools/inspect_presentation_package_integrity.py"),
    layoutValidatorPath: path.join(SKILL, "container_tools/inspect_presentation_layout_geometry.py"),
    layoutArgs: ["--expected-slide-size-emu", "12192000,6858000", "--validate-bullet-geometry", "--validate-heading-fit",
      ...tables.flatMap(n => ["--require-native-table-slide", String(n)])],
    requiredNativeTableOwnerSlides: tables, requiredNativeChartOwnerSlides: charts,
    materializeLiteralChartWorkbooks: true,
    fontPolicy: { basis: "design", families: [FONT] }, verifyArtifactToolImport: true,
    receiptPath: path.join(build, "validation.json") });
  const finalDeck = await PresentationFile.importPptx(await FileBlob.load(final));
  for (const [index, s] of finalDeck.slides.items.entries()) {
    const blob = await finalDeck.export({ slide: s, format: "png", scale: 1 });
    await fs.writeFile(path.join(previews, `slide-${String(index + 1).padStart(2, "0")}.png`), new Uint8Array(await blob.arrayBuffer()));
  }
  await fs.copyFile(final, path.join(output, "slides.pptx"));
  const receipt = { slide_count: spec.slides.length, native_table_slides: tables, native_chart_slides: charts,
    input_sha256: digest(await fs.readFile(input)), final_sha256: digest(await fs.readFile(final)),
    previews, validation: path.join(build, "validation.json") };
  await fs.writeFile(path.join(build, "inventory.json"), JSON.stringify(receipt, null, 2));
  console.log(JSON.stringify(receipt));
}
main().catch(error => { console.error(error.stack || error.message); process.exitCode = 1; });
