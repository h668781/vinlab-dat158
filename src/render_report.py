"""Render docs/rapport.md as a readable PDF. Optional dependency: reportlab."""
import re
from html import escape
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

ROOT = Path(__file__).resolve().parents[1]


def inline(text):
    text = escape(text)
    text = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)',
                  r'<link href="\2" color="#25476a"><u>\1</u></link>', text)
    return text


def main():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='ReportBody', fontName='Helvetica', fontSize=10,
                             leading=14.1, spaceAfter=9))
    styles.add(ParagraphStyle(name='ReportTitle', fontName='Helvetica-Bold', fontSize=24,
                             leading=29, spaceAfter=16, textColor=colors.black))
    styles.add(ParagraphStyle(name='ReportHeading', fontName='Helvetica-Bold', fontSize=15,
                             leading=19, spaceBefore=12, spaceAfter=10, keepWithNext=True))
    styles.add(ParagraphStyle(name='ReportSubheading', fontName='Helvetica-Bold', fontSize=11,
                             leading=15, spaceBefore=10, spaceAfter=8, keepWithNext=True))
    styles.add(ParagraphStyle(name='Cell', fontName='Helvetica', fontSize=8.1, leading=11))
    styles.add(ParagraphStyle(name='CellHeader', fontName='Helvetica-Bold', fontSize=8.1,
                             leading=11, textColor=colors.white))
    story = []
    blocks = (ROOT / 'docs/rapport.md').read_text(encoding='utf-8').split('\n\n')
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if block == '<!-- PAGEBREAK -->':
            story.append(PageBreak())
        elif block.startswith('### '):
            story.append(Paragraph(inline(block[4:]), styles['ReportSubheading']))
        elif block.startswith('## '):
            story.append(Paragraph(inline(block[3:]), styles['ReportHeading']))
        elif block.startswith('# '):
            story.append(Paragraph(inline(block[2:]), styles['ReportTitle']))
        elif block.startswith('|'):
            rows = []
            for line in block.splitlines():
                if re.fullmatch(r'[|:\- ]+', line):
                    continue
                values = [x.strip() for x in line.strip('|').split('|')]
                rows.append([Paragraph(inline(x), styles['CellHeader' if not rows else 'Cell']) for x in values])
            n = len(rows[0])
            widths = {6: [145, 64, 60, 65, 58, 63], 5: [145, 76, 80, 76, 78], 3: [220, 120, 115]}[n]
            table = Table(rows, colWidths=widths, repeatRows=1, hAlign='LEFT')
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#344352')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f3f5f7')]),
                ('GRID', (0, 0), (-1, -1), .4, colors.HexColor('#d9d9d9')),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ]))
            story.extend([table, Spacer(1, 12)])
        else:
            story.append(Paragraph(inline(block.replace('\n', ' ')), styles['ReportBody']))

    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor('#d9d9d9'))
        canvas.line(60, 40, A4[0] - 60, 40)
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#666666'))
        canvas.drawString(60, 27, 'Vinlab | DAT158')
        canvas.drawRightString(A4[0] - 60, 27, str(doc.page))

    doc = SimpleDocTemplate(str(ROOT / 'docs/rapport.pdf'), pagesize=A4,
                            rightMargin=70, leftMargin=70, topMargin=47, bottomMargin=58,
                            title='Vinlab vinkvalitet med maskinlæring', author='DAT158 prosjektutkast')
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print('Created docs/rapport.pdf')


if __name__ == '__main__':
    main()
