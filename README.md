# ChurnGuard AI — Setup Guide

## New Features Added
- **SHAP Explainability** — per-prediction feature importance waterfall on result page
- **SQLite Database** — replaces prediction_log.jsonl via SQLAlchemy ORM
- **Claude AI Recommendations** — personalised retention strategy per customer
- **Customer Segmentation** — KMeans clustering + PCA scatter at /segmentation
- **PDF Report Export** — downloadable per-customer report with SHAP + AI content
- **Genetic Algorithm** — preserved at /ga-optimize (unchanged)

## Installation

```bash
pip install -r requirement.txt
```

## Claude API Key (for AI Recommendations)

```bash
# Linux/Mac
export ANTHROPIC_API_KEY=your_key_here

# Windows
set ANTHROPIC_API_KEY=your_key_here
```

Without the key, the app uses a smart rule-based fallback automatically.

## Run

```bash
python app.py
```

App starts at http://localhost:5000

## New Routes

| Route | Description |
|---|---|
| `/segmentation` | Customer segmentation page |
| `/segmentation?n_clusters=5` | Adjust cluster count (2-7) |
| `/report/download` | Download PDF for current prediction (POST) |
| `/report/download/<id>` | Download PDF by prediction record ID |
| `/api/shap` | SHAP values for a customer payload (POST) |
| `/api/ai-recommend` | Claude AI recommendation (POST) |
| `/api/predictions` | All predictions from database (GET) |
| `/api/predictions/<id>` | Single prediction detail (GET) |
| `/ga-optimize` | Genetic Algorithm page (unchanged) |
| `/api/ga-run` | Trigger GA run (unchanged) |

## Database

SQLite database is created automatically at `churn_app.db`.
Tables: `predictions`, `customer_segments`, `model_metrics`

## New Files

```
Customer Churn/
├── app.py                  ← Updated (all features integrated)
├── database.py             ← NEW: SQLAlchemy models + helpers
├── shap_explainer.py       ← NEW: SHAP computation
├── ai_recommendations.py   ← NEW: Claude API integration
├── segmentation.py         ← NEW: KMeans + RFM analysis
├── pdf_report.py           ← NEW: ReportLab PDF generator
├── genetic_algorithm.py    ← PRESERVED (unchanged)
├── templates/
│   ├── segmentation.html   ← NEW: Segmentation dashboard
│   ├── ga_results.html     ← PRESERVED (unchanged)
│   ├── result.html         ← UPDATED: SHAP + AI rec + PDF button
│   └── base.html           ← UPDATED: Segments nav link added
└── requirement.txt         ← UPDATED: new packages added
```
