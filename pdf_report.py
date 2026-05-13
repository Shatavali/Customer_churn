# pdf_report.py
"""
PDF Report Generator using ReportLab.
Produces a per-customer churn risk report with:
  - Risk score gauge
  - Key customer details table
  - SHAP feature importance bar chart
  - AI-generated retention recommendation
"""

import io
import json
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.graphics.shapes import Drawing, Rect, String, Circle, Wedge, Line
from reportlab.graphics.charts.barcharts import HorizontalBarChart
from reportlab.graphics import renderPDF
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# ── Colour palette ────────────────────────────────────────────────────
HIGH_COLOR   = colors.HexColor('#ef4444')
MEDIUM_COLOR = colors.HexColor('#f59e0b')
LOW_COLOR    = colors.HexColor('#10b981')
ACCENT       = colors.HexColor('#6366f1')
DARK         = colors.HexColor('#1e293b')
MUTED        = colors.HexColor('#64748b')
BG_LIGHT     = colors.HexColor('#f8fafc')
BORDER       = colors.HexColor('#e2e8f0')


def _risk_color(probability):
    if probability > 0.7:
        return HIGH_COLOR
    elif probability > 0.4:
        return MEDIUM_COLOR
    return LOW_COLOR


def _risk_label(probability):
    if probability > 0.7:
        return 'HIGH RISK'
    elif probability > 0.4:
        return 'MEDIUM RISK'
    return 'LOW RISK'


def _gauge_drawing(probability):
    """Draw a simple semicircular gauge for the risk score."""
    d = Drawing(200, 110)

    # Background arc (grey)
    for i in range(180):
        w = Wedge(100, 20, 70, i, i + 1)
        w.fillColor = colors.HexColor('#e2e8f0')
        w.strokeColor = None
        d.add(w)

    # Filled arc (risk color)
    risk_angle = int(probability * 180)
    rc = _risk_color(probability)
    for i in range(risk_angle):
        w = Wedge(100, 20, 70, i, i + 1)
        w.fillColor = rc
        w.strokeColor = None
        d.add(w)

    # Inner white circle to create donut
    c = Circle(100, 20, 48)
    c.fillColor = colors.white
    c.strokeColor = None
    d.add(c)

    # Score text
    s = String(100, 30, f"{int(probability * 100)}%",
               fontName='Helvetica-Bold', fontSize=22,
               textAnchor='middle', fillColor=rc)
    d.add(s)

    label = String(100, 8, _risk_label(probability),
                   fontName='Helvetica-Bold', fontSize=8,
                   textAnchor='middle', fillColor=rc)
    d.add(label)

    return d


def _shap_bar_chart(shap_data):
    """Draw a horizontal bar chart of top SHAP features."""
    top = shap_data.get('top_positive', [])[:6]
    if not top:
        return None

    features = [f[:28] for f, _ in top]
    values   = [v for _, v in top]
    n        = len(features)
    height   = max(80, n * 22 + 30)

    d = Drawing(400, height)
    bc = HorizontalBarChart()
    bc.x            = 150
    bc.y            = 10
    bc.width        = 230
    bc.height       = height - 20
    bc.data         = [values]
    bc.valueAxis.valueMin = 0
    bc.valueAxis.valueMax = max(values) * 1.2 if values else 1
    bc.valueAxis.labels.fontSize = 7
    bc.categoryAxis.categoryNames = features
    bc.categoryAxis.labels.fontSize = 8
    bc.categoryAxis.labels.dx = -4
    bc.bars[0].fillColor   = ACCENT
    bc.bars[0].strokeColor = None
    d.add(bc)
    return d


def generate_pdf(customer_data, probability, prediction,
                 shap_data=None, ai_recommendation=None,
                 customer_id=None):
    """
    Generate a PDF report for a single customer prediction.

    Returns
    -------
    bytes : PDF file content
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )

    styles  = getSampleStyleSheet()
    story   = []
    W       = A4[0] - 4*cm  # usable width

    # ── Header bar ────────────────────────────────────────────────────
    header_data = [[
        Paragraph(
            '<font color="#6366f1" size="16"><b>ChurnGuard AI</b></font><br/>'
            '<font color="#64748b" size="9">Customer Churn Risk Report</font>',
            styles['Normal']
        ),
        Paragraph(
            f'<font color="#64748b" size="8">Report Date: {datetime.now().strftime("%d %b %Y %H:%M")}<br/>'
            f'Customer ID: {customer_id or "N/A"}<br/>'
            f'Report ID: RPT-{datetime.now().strftime("%Y%m%d%H%M%S")}</font>',
            ParagraphStyle('right', alignment=TA_RIGHT, fontSize=8)
        )
    ]]
    t = Table(header_data, colWidths=[W*0.6, W*0.4])
    t.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LINEBELOW', (0, 0), (-1, 0), 1, ACCENT),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.4*cm))

    # ── Risk Score + Gauge ────────────────────────────────────────────
    gauge   = _gauge_drawing(probability)
    rc      = _risk_color(probability)
    rc_hex  = rc.hexval() if hasattr(rc, 'hexval') else '#000'

    risk_text = Paragraph(
        f'<b><font size="13" color="{rc_hex}">{_risk_label(probability)}</font></b><br/><br/>'
        f'<font size="9" color="#64748b">Churn Probability: </font>'
        f'<b><font size="11" color="{rc_hex}">{probability:.1%}</font></b><br/><br/>'
        f'<font size="8" color="#64748b">This customer has a '
        f'{"high" if probability > 0.7 else "medium" if probability > 0.4 else "low"} '
        f'likelihood of churning based on the ML model analysis.</font>',
        styles['Normal']
    )

    gauge_table = Table(
        [[gauge, risk_text]],
        colWidths=[210, W - 210]
    )
    gauge_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BACKGROUND', (0, 0), (-1, -1), BG_LIGHT),
        ('ROUNDEDCORNERS', [8]),
        ('INNERGRID', (0, 0), (-1, -1), 0, colors.white),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER),
        ('PADDING', (0, 0), (-1, -1), 12),
    ]))
    story.append(gauge_table)
    story.append(Spacer(1, 0.5*cm))

    # ── Customer Details Table ────────────────────────────────────────
    story.append(Paragraph('<b>Customer Profile</b>',
                           ParagraphStyle('h2', fontSize=11, textColor=DARK,
                                          spaceAfter=6)))
    details = [
        ['Field', 'Value', 'Field', 'Value'],
        ['Gender',          customer_data.get('gender', '-'),
         'Senior Citizen',  'Yes' if customer_data.get('SeniorCitizen') == 1 else 'No'],
        ['Tenure',          f"{customer_data.get('tenure', '-')} months",
         'Partner',         customer_data.get('Partner', '-')],
        ['Monthly Charges', f"${customer_data.get('MonthlyCharges', '-')}",
         'Contract',        customer_data.get('Contract', '-')],
        ['Total Charges',   f"${customer_data.get('TotalCharges', '-')}",
         'Internet Service',customer_data.get('InternetService', '-')],
        ['Payment Method',  customer_data.get('PaymentMethod', '-'),
         'Tech Support',    customer_data.get('TechSupport', '-')],
        ['Paperless Billing',customer_data.get('PaperlessBilling', '-'),
         'Streaming TV',    customer_data.get('StreamingTV', '-')],
    ]
    dt = Table(details, colWidths=[W*0.22, W*0.28, W*0.22, W*0.28])
    dt.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), ACCENT),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
        ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
        ('GRID',       (0, 0), (-1, -1), 0.3, BORDER),
        ('PADDING',    (0, 0), (-1, -1), 5),
        ('FONTNAME',   (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME',   (2, 1), (2, -1), 'Helvetica-Bold'),
        ('TEXTCOLOR',  (0, 1), (0, -1), MUTED),
        ('TEXTCOLOR',  (2, 1), (2, -1), MUTED),
    ]))
    story.append(dt)
    story.append(Spacer(1, 0.5*cm))

    # ── SHAP Feature Importance ───────────────────────────────────────
    if shap_data and shap_data.get('top_positive'):
        story.append(KeepTogether([
            Paragraph('<b>Key Churn Risk Drivers (SHAP Analysis)</b>',
                      ParagraphStyle('h2', fontSize=11, textColor=DARK, spaceAfter=6)),
            Paragraph(
                '<font size="8" color="#64748b">Features ranked by their impact on the churn probability. '
                'Positive values push toward churn; negative values reduce churn risk.</font>',
                styles['Normal']
            ),
            Spacer(1, 0.3*cm),
        ]))

        # Table of top positive SHAP features
        shap_rows = [['Feature', 'SHAP Impact', 'Direction']]
        for fname, fval in shap_data.get('top_positive', [])[:6]:
            direction = '▲ Increases churn risk'
            shap_rows.append([fname[:40], f"+{fval:.4f}", direction])
        for fname, fval in shap_data.get('top_negative', [])[:3]:
            direction = '▼ Reduces churn risk'
            shap_rows.append([fname[:40], f"{fval:.4f}", direction])

        st = Table(shap_rows, colWidths=[W*0.55, W*0.2, W*0.25])
        st.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), DARK),
            ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
            ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE',   (0, 0), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
            ('GRID',       (0, 0), (-1, -1), 0.3, BORDER),
            ('PADDING',    (0, 0), (-1, -1), 5),
            ('TEXTCOLOR',  (2, 1), (2, len(shap_data.get('top_positive', [])[:6])), HIGH_COLOR),
            ('TEXTCOLOR',  (2, len(shap_data.get('top_positive', [])[:6])+1), (-1, -1), LOW_COLOR),
        ]))
        story.append(st)
        story.append(Spacer(1, 0.5*cm))

    # ── AI Recommendation ─────────────────────────────────────────────
    if ai_recommendation:
        story.append(KeepTogether([
            Paragraph('<b>AI-Generated Retention Strategy</b>',
                      ParagraphStyle('h2', fontSize=11, textColor=DARK, spaceAfter=6)),
            Paragraph(
                '<font size="8" color="#64748b">Generated by Claude AI — personalised to this customer\'s profile.</font>',
                styles['Normal']
            ),
            Spacer(1, 0.3*cm),
        ]))

        rec_style = ParagraphStyle(
            'rec', fontSize=9, leading=14,
            leftIndent=10, rightIndent=10,
            borderPad=10, backColor=BG_LIGHT,
            borderColor=ACCENT, borderWidth=1,
            borderRadius=5,
        )
        # Clean markdown-like formatting for ReportLab XML
        import re
        # Replace **text** with <b>text</b> properly
        clean_rec = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', ai_recommendation)
        clean_rec = clean_rec.replace('\n', '<br/>')
        # Escape any remaining ampersands
        clean_rec = clean_rec.replace('&', '&amp;')
        story.append(Paragraph(clean_rec, rec_style))
        story.append(Spacer(1, 0.4*cm))

    # ── Footer ────────────────────────────────────────────────────────
    story.append(HRFlowable(width=W, thickness=0.5, color=BORDER))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        '<font size="7" color="#94a3b8">This report is generated automatically by ChurnGuard AI. '
        'Predictions are probabilistic estimates and should be used alongside human judgement. '
        'Confidential — for internal use only.</font>',
        ParagraphStyle('footer', fontSize=7, textColor=MUTED, alignment=TA_CENTER)
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
