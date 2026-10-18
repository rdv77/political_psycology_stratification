"""Render the political report's shared content to a Cyrillic, linked PDF.

Use a Python environment with reportlab, Pillow and pypdf. No network or API.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import re

from PIL import Image as PILImage
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, NextPageTemplate,
                               PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'research/rwanda_politicians_2021_2026/political_report_q4_2026_10_18'


class PoliticalReportDoc(BaseDocTemplate):
    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name == 'Heading':
            key = 'section-' + str(self.seq.nextf('section'))
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(flowable.getPlainText(), key, level=0, closed=False)


def register_fonts():
    windows = Path('C:/Windows/Fonts')
    if (windows/'arial.ttf').exists():
        paths = {'Body':windows/'arial.ttf', 'Body-Bold':windows/'arialbd.ttf', 'Body-Italic':windows/'ariali.ttf'}
    else:
        fonts = Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/share/fonts'
        paths = {'Body':fonts/'Ubuntu-R.ttf', 'Body-Bold':fonts/'Ubuntu-B.ttf', 'Body-Italic':fonts/'Ubuntu-RI.ttf'}
    for name, path in paths.items():
        pdfmetrics.registerFont(TTFont(name, str(path)))
    pdfmetrics.registerFontFamily('Body', normal='Body', bold='Body-Bold', italic='Body-Italic', boldItalic='Body-Bold')


def inline(value):
    s = html.escape(value)
    s = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<link href="\2" color="#167d9a">\1</link>', s)
    s = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', s)
    return s.replace('\n', '<br/>')


def build():
    register_fonts()
    body = ParagraphStyle('Body', fontName='Body', fontSize=10.8, leading=15.4,
                          textColor=colors.HexColor('#203743'), spaceAfter=8.5,
                          allowWidows=0, allowOrphans=0, splitLongWords=1)
    styles = {
        'body':body,
        'title':ParagraphStyle('Title', parent=body, fontName='Body-Bold', fontSize=29, leading=34, spaceAfter=22),
        'subtitle':ParagraphStyle('Subtitle', parent=body, fontSize=13, leading=19, textColor=colors.HexColor('#526672'), spaceAfter=15),
        'heading':ParagraphStyle('Heading', parent=body, fontName='Body-Bold', fontSize=20, leading=25, spaceBefore=14, spaceAfter=15, keepWithNext=True),
        'subheading':ParagraphStyle('Subheading', parent=body, fontName='Body-Bold', fontSize=13.3, leading=18, spaceBefore=14, spaceAfter=9, keepWithNext=True),
        'bioheading':ParagraphStyle('Bioheading', parent=body, fontName='Body-Bold', fontSize=12, leading=16, spaceBefore=12, spaceAfter=6, keepWithNext=True),
        'small':ParagraphStyle('Small', parent=body, fontSize=9, leading=12, textColor=colors.HexColor('#526672'), spaceAfter=6),
        'quote':ParagraphStyle('Quote', parent=body, fontSize=13, leading=19, leftIndent=12, rightIndent=10, textColor=colors.HexColor('#167d9a'), spaceAfter=8),
        'note':ParagraphStyle('Note', parent=body, fontSize=10.1, leading=14.2, borderColor=colors.HexColor('#dbe3e7'), borderWidth=.6, borderPadding=12, backColor=colors.HexColor('#f2f6f8'), spaceBefore=10, spaceAfter=16),
        'bullet':ParagraphStyle('Bullet', parent=body, leftIndent=12, firstLineIndent=-10, spaceAfter=11),
        'cell':ParagraphStyle('Cell', parent=body, fontSize=9.7, leading=13.5, spaceAfter=0),
        'figtitle':ParagraphStyle('FigTitle', parent=body, fontName='Body-Bold', fontSize=16, leading=20, spaceAfter=7, keepWithNext=True),
    }
    content = json.loads((OUT/'report_content.json').read_text(encoding='utf-8'))
    output = OUT/'REPORT.pdf'
    width, height = A4
    margin_x, margin_y = 21*mm, 19*mm
    body_width = width-2*margin_x

    def page(canvas, doc):
        w, h = canvas._pagesize
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor('#e0e7eb'))
        canvas.line(17*mm, 13.5*mm, w-17*mm, 13.5*mm)
        canvas.setFont('Body', 8)
        canvas.setFillColor(colors.HexColor('#526672'))
        canvas.drawString(17*mm, 9.5*mm, 'РУАНДА · ПОЛИТИЧЕСКОЕ ПРОСТРАНСТВО · 2021–2026')
        canvas.drawRightString(w-17*mm, 9.5*mm, str(doc.page))
        canvas.restoreState()

    doc = PoliticalReportDoc(str(output), pagesize=A4, title='Руанда: власть, развитие и язык политической легитимности',
                          author='Political psychology stratification research',
                          leftMargin=margin_x, rightMargin=margin_x, topMargin=margin_y, bottomMargin=margin_y)
    portrait = Frame(margin_x, margin_y, body_width, height-2*margin_y,
                     leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id='body')
    lw, lh = landscape(A4)
    landscape_frame = Frame(13*mm, 18*mm, lw-26*mm, lh-29*mm,
                            leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id='chart')
    doc.addPageTemplates([PageTemplate(id='portrait', frames=[portrait], onPage=page, pagesize=A4),
                          PageTemplate(id='landscape', frames=[landscape_frame], onPage=page, pagesize=landscape(A4))])
    story = [Spacer(1, 16*mm)]
    para = lambda text, style='body': Paragraph(inline(text), styles[style])
    # Put each section's wide figures after its prose, so a short introduction or
    # a paragraph between two maps does not consume an otherwise empty page.
    ordered, pending = [], []
    for block in content['blocks']:
        if block['kind'] == 'heading':
            ordered.extend(pending); pending = []
        if block['kind'] == 'figure':
            pending.append(block)
        else:
            ordered.append(block)
    ordered.extend(pending)
    after_figure = False
    biography_prefix = []
    for index, x in enumerate(ordered):
        k, t = x['kind'], x['text']
        if k in ('title', 'subtitle'):
            story.append(para(t, k))
        elif k == 'heading':
            if t == 'Основные выводы' or t.startswith('9. '):
                if not after_figure:
                    story.append(PageBreak())
            story.append(para(t, 'heading'))
        elif k == 'subheading':
            if index+1 < len(ordered) and ordered[index+1]['kind'] == 'biography':
                biography_prefix = [para(t, 'subheading')]
            else:
                story.append(para(t, 'subheading'))
        elif k == 'paragraph':
            story.append(para(t))
        elif k == 'bullet':
            story.append(para('• '+t, 'bullet'))
        elif k == 'note':
            story.append(para(t, 'note'))
        elif k == 'table':
            rows = [[para(v, 'cell') for v in x['headers']]]
            rows += [[para(v, 'cell') for v in row] for row in x['rows']]
            table = Table(rows, colWidths=[body_width*.29, body_width*.71], repeatRows=1, hAlign='LEFT')
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#eaf1f4')),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('BOX', (0,0), (-1,-1), .5, colors.HexColor('#cbd8df')),
                ('INNERGRID', (0,0), (-1,-1), .35, colors.HexColor('#dbe3e7')),
                ('LEFTPADDING', (0,0), (-1,-1), 9), ('RIGHTPADDING',(0,0),(-1,-1),9),
                ('TOPPADDING', (0,0), (-1,-1), 9), ('BOTTOMPADDING',(0,0),(-1,-1),9),
            ]))
            story.extend([table, Spacer(1, 10)])
        elif k == 'quote':
            nested = t.replace('«', '“').replace('»', '”')
            items = [para('«'+nested+'»', 'quote'),
                     para(x['translation']+'. Оригинал: “'+x['original']+'”', 'small'),
                     para(x['attribution']+'. ['+x['source_title']+']('+x['source_url']+').', 'small'), Spacer(1, 8)]
            story.append(KeepTogether(items))
        elif k == 'figure':
            if not after_figure:
                story.extend([NextPageTemplate('landscape'), PageBreak()])
            chart_width = lw-29*mm
            iw, ih = PILImage.open(OUT/x['path']).size
            image_height = chart_width*ih/iw
            # Leave comfortable space for title/caption even for the taller maps.
            max_height = 414
            if image_height > max_height:
                chart_width *= max_height/image_height
                image_height = max_height
            image = Image(str(OUT/x['path']), width=chart_width, height=image_height)
            image.hAlign = 'CENTER'
            next_is_figure = index+1 < len(ordered) and ordered[index+1]['kind'] == 'figure'
            story.extend([para(t, 'figtitle'), image, Spacer(1, 7), para(x['caption'], 'small'),
                          NextPageTemplate('landscape' if next_is_figure else 'portrait'), PageBreak()])
        elif k == 'biography':
            refs = '; '.join('['+s['title']+']('+s['url']+')' for s in x['sources'])
            # Keep each short entry intact; no repeated role line.
            story.append(KeepTogether(biography_prefix+[
                para(t, 'bioheading'), para(x['body']),
                para('Период/дата: '+x['period']+'. '+x['group'], 'small'),
                para('Источники: '+refs+'.', 'small'),
            ]))
            biography_prefix = []
        after_figure = k == 'figure'
    doc.build(story)
    reader = PdfReader(str(output))
    texts = [page.extract_text() or '' for page in reader.pages]
    all_text = '\n'.join(texts)
    bio_checks = {p['name_original']: p['name_original'] in all_text for p in json.loads((OUT/'biographies.json').read_text(encoding='utf-8'))['profiles']}
    result = {'pages':len(reader.pages), 'landscape_pages':[i+1 for i,p in enumerate(reader.pages) if float(p.mediabox.width)>float(p.mediabox.height)],
              'empty_pages':[i+1 for i,t in enumerate(texts) if len(t.strip())<80],
              'directory_names_found':sum(bio_checks.values()), 'directory_names':bio_checks,
              'links':sum(len(p.get('/Annots', [])) for p in reader.pages), 'outline_sections':len(reader.outline), 'file_bytes':output.stat().st_size,
              'paid_api_calls':0}
    assert result['directory_names_found'] == 42, result
    assert len(result['landscape_pages']) == 4, result
    assert not result['empty_pages'], result
    assert result['outline_sections'] == 11, result
    (OUT/'pdf_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='directory_names'}, ensure_ascii=False))


if __name__ == '__main__':
    build()
