/** Editable source-based Rwanda update. Offline authoring and rendering only. */
import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL, fileURLToPath} from 'node:url';
import {createHash, randomUUID} from 'node:crypto';

const BUNDLE = 'C:/Users/USER/.cache/codex-runtimes/codex-primary-runtime/dependencies';
const MODULES = process.env.SEMANTIC_NODE_MODULES || `${BUNDLE}/node/node_modules`;
const PYTHON = process.env.SEMANTIC_ARTIFACT_PYTHON || `${BUNDLE}/python/python.exe`;
const SKILL = process.env.SEMANTIC_PRESENTATION_SKILL ||
  'C:/Users/USER/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations';
const FONT = 'Arial', INK = '#142B35', TEAL = '#176D77', GREY = '#56656E';
const digest = bytes => createHash('sha256').update(bytes).digest('hex');

async function main() {
  const [inputArg, outputArg] = process.argv.slice(2);
  if (!inputArg || !outputArg) throw new Error('Usage: presentation_rwanda_source_v2.mjs SLIDE_CONTENT_JSON OUTPUT_DIR');
  const input = path.resolve(inputArg), output = path.resolve(outputArg);
  const spec = JSON.parse(await fs.readFile(input, 'utf8'));
  if (!spec.slides?.length || spec.actor_count !== 42 || spec.scale_count !== 38)
    throw new Error('The complete approved roster and dictionary are required');
  if (spec.analysis_status !== 'not_admitted') throw new Error('This narrative requires a non-admitted source matrix');
  const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
  const source=path.resolve(root,spec.source_json);
  if (!source.startsWith(root+path.sep) || digest(await fs.readFile(source)) !== spec.source_sha256)
    throw new Error('The reviewed numerical source changed after narrative authoring');
  const build = path.join(output, '.privatebuild', `source-v2-${randomUUID()}`);
  const previews = path.join(build, 'previews'), valid = path.join(build, 'validated');
  await fs.mkdir(previews, {recursive:true});
  await fs.mkdir(valid, {recursive:true});
  process.env.RUNTIME_NODE_MODULES = MODULES;
  const resolver = createRequire(path.join(MODULES, '.source-v2-resolver.cjs'));
  const {Presentation, PresentationFile, FileBlob} = await import(pathToFileURL(resolver.resolve('@oai/artifact-tool')).href);
  const {finalizePresentation, applyPresentationChartFont} = await import(pathToFileURL(
    path.join(SKILL, 'container_tools/artifact_tool_utils.mjs')).href);
  const presentation = Presentation.create({slideSize:{width:1280,height:720}});
  const chartOwners = [], tableOwners=[];

  function text(slide, value, position, size=28, options={}) {
    const shape = slide.shapes.add({geometry:'textbox',name:options.name || String(value).slice(0,40),
      position,fill:'none',line:{fill:'none',width:0}});
    shape.text = String(value);
    shape.text.style = {typeface:FONT,fontSize:size,bold:Boolean(options.bold),color:options.color || INK,
      alignment:options.align || 'left',verticalAlignment:'top',autoFit:'none',wrap:'square',insets:0};
    return shape;
  }
  for (const [index,page] of spec.slides.entries()) {
    const slide = presentation.slides.add();
    slide.background.fill = '#FFFFFF';
    slide.speakerNotes.text = [page.notes || '', ...(page.sources || []).map(url=>`Источник: ${url}`)].join('\n\n');
    const titleSize = page.title.length > 75 ? 36 : page.title.length > 58 ? 39 : 42;
    text(slide,page.title,{left:66,top:40,width:1148,height:94},titleSize,{bold:true});
    text(slide,String(index+1),{left:1170,top:682,width:44,height:18},13,{color:GREY,align:'right'});
    if (page.kind === 'cover') {
      text(slide,page.subtitle,{left:66,top:206,width:1110,height:138},44,{color:TEAL,bold:true});
      text(slide,page.body,{left:66,top:414,width:1110,height:156},30);
    } else if (page.kind === 'chart') {
      const chart = slide.charts.add(page.chart.type,{position:{left:78,top:170,width:1108,height:410},
        chartFill:'#FFFFFF',plotAreaFill:'#FFFFFF',...structuredClone(page.chart.config)});
      chart.xAxis.textStyle.fontSize=24;
      chart.yAxis.textStyle.fontSize=24;
      chart.dataLabels.textStyle.fontSize=26;
      applyPresentationChartFont(chart,{fontFamily:FONT});
      chartOwners.push(index+1);
    } else if (page.kind === 'table') {
      const values=[page.headers,...page.rows];
      const table=slide.tables.add({rows:values.length,columns:3,left:66,top:149,width:1148,height:437,
        columnWidths:page.widths,values});
      table.borders.assign({style:'solid',fill:'#D6E0E3',width:.8});
      for(let row=0;row<values.length;row++)for(let column=0;column<3;column++){
        const cell=table.getCell(row,column);
        cell.fill=row===0?TEAL:row%2?'#FFFFFF':'#F3F6F7';
        cell.text.style={typeface:FONT,fontSize:24,color:row===0?'#FFFFFF':INK,bold:row===0,
          alignment:'left',verticalAlignment:'middle',insets:9,autoFit:'none',wrap:'square'};
      }
      tableOwners.push(index+1);
    } else if (page.kind === 'scale_list') {
      const rowHeight = page.items.length > 6 ? 62 : 73;
      for (const [i,item] of page.items.entries()) {
        text(slide,item.name,{left:80,top:148+i*rowHeight,width:1120,height:rowHeight-8},
          item.name.length > 78 ? 25 : 27);
      }
      slide.speakerNotes.text += '\n\nВсе утвержденные признаки этой страницы:\n' + page.items.map(item=>`${item.id}: ${item.name}`).join('\n');
    } else if (page.kind === 'profiles') {
      for (const [i,person] of page.profiles.entries()) {
        const top = 145+i*152;
        text(slide,person.name,{left:78,top,width:1124,height:39},29,{bold:true,color:TEAL});
        text(slide,person.role,{left:78,top:top+42,width:1124,height:70},24);
        text(slide,person.coverage,{left:78,top:top+108,width:1124,height:29},22,{color:GREY});
        slide.speakerNotes.text += '\n\n' + person.name + '\n' + (person.sources||[]).map(url=>`Биографический источник: ${url}`).join('\n');
      }
    } else {
      let top=153;
      const gap=page.gap ?? 25;
      for (const paragraph of page.paragraphs || []) {
        const part=typeof paragraph === 'string'?{text:paragraph}:paragraph;
        const size=part.size || page.fontSize || 28;
        const lines=Math.max(1,Math.ceil(part.text.length/(1136/(size*.53))));
        const height=lines*size*1.27+8;
        text(slide,part.text,{left:78,top,width:1124,height},size,{bold:part.bold,color:part.color});
        top+=height+gap;
      }
      if (top > 658) throw new Error(`Text exceeds available slide area: slide ${index+1}, ${top}`);
    }
    if (page.caption) text(slide,page.caption,{left:66,top:624,width:1148,height:45},20,{color:GREY});
  }
  const candidate=path.join(build,'candidate.pptx'), final=path.join(valid,'slides.pptx');
  await (await PresentationFile.exportPptx(presentation)).save(candidate);
  await finalizePresentation({workspaceDir:output,candidatePath:candidate,finalPath:final,
    pythonExecutable:PYTHON, integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
    layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
    layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',
      ...tableOwners.flatMap(number=>['--require-native-table-slide',String(number)])],
    requiredNativeTableOwnerSlides:tableOwners,requiredNativeChartOwnerSlides:chartOwners,
    materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:[FONT]},verifyArtifactToolImport:true,
    receiptPath:path.join(build,'validation.json')});
  const reopened=await PresentationFile.importPptx(await FileBlob.load(final));
  for (const [index,slide] of reopened.slides.items.entries()) {
    const png=await reopened.export({slide,format:'png',scale:1});
    await fs.writeFile(path.join(previews,`slide-${String(index+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
  }
  await fs.copyFile(final,path.join(output,'slides.pptx'));
  const receipt={slide_count:spec.slides.length,native_chart_slides:chartOwners,native_table_slides:tableOwners,previews,
    input_sha256:digest(await fs.readFile(input)),final_sha256:digest(await fs.readFile(final)),validation:path.join(build,'validation.json')};
  await fs.writeFile(path.join(build,'inventory.json'),JSON.stringify(receipt,null,2));
  console.log(JSON.stringify(receipt));
}
main().catch(error=>{console.error(error.stack || error.message);process.exitCode=1;});
