# segmentation.py
"""
Customer Segmentation Module
- KMeans clustering on numeric features
- RFM (Recency, Frequency, Monetary) analysis proxy using telecom features
- Returns cluster profiles and churn rates per segment
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

SEGMENT_LABELS = {
    0: ("Champions",       "🏆", "#10b981"),
    1: ("At-Risk Loyals",  "⚠️",  "#f59e0b"),
    2: ("New & Uncertain", "🆕", "#3b82f6"),
    3: ("Churners",        "🔴", "#ef4444"),
    4: ("Hibernating",     "💤", "#8b5cf6"),
}


def load_and_prepare(csv_path='Telco_Churn_Cleaned.csv'):
    """Load CSV and prepare numeric features for clustering."""
    df = pd.read_csv(csv_path)

    # Encode target if present
    if 'Churn' in df.columns:
        df['Churn_num'] = df['Churn'].map(
            lambda x: 1 if str(x).strip() in ['Yes', '1', 'True'] else 0
        )

    # Select numeric features for clustering
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    available = [c for c in numeric_cols if c in df.columns]

    # Add engineered features if present
    for col in ['ChargesPerMonth', 'ServiceCount', 'AvgMonthlySpend']:
        if col in df.columns:
            available.append(col)

    # Derived features
    if 'TotalCharges' in df.columns and 'tenure' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
        df['AvgSpend'] = df['TotalCharges'] / (df['tenure'] + 1)
        available.append('AvgSpend')

    available = list(dict.fromkeys(available))  # deduplicate
    return df, available


def run_segmentation(csv_path='Telco_Churn_Cleaned.csv', n_clusters=4):
    """
    Run KMeans segmentation and return cluster stats.

    Returns
    -------
    dict with:
        segments      : list of segment profile dicts
        pca_points    : list of {x, y, cluster, churn} for scatter plot
        rfm_summary   : RFM-style summary table
        n_clusters    : int
    """
    df, feature_cols = load_and_prepare(csv_path)

    X = df[feature_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = km.fit_predict(X_scaled)

    # PCA for 2D visualisation
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    df['pca_x'] = coords[:, 0]
    df['pca_y'] = coords[:, 1]

    # Sample points for scatter (max 500 per cluster for performance)
    pca_points = []
    for _, row in df.sample(min(len(df), 1500), random_state=42).iterrows():
        pca_points.append({
            'x':       round(float(row['pca_x']), 3),
            'y':       round(float(row['pca_y']), 3),
            'cluster': int(row['cluster']),
            'churn':   int(row.get('Churn_num', 0)),
        })

    # Build segment profiles
    segments = []
    for c in range(n_clusters):
        mask  = df['cluster'] == c
        group = df[mask]
        label, icon, color = SEGMENT_LABELS.get(c, (f"Segment {c}", "📊", "#6b7280"))

        churn_rate = float(group['Churn_num'].mean()) if 'Churn_num' in group else 0.0
        # Auto-label based on churn rate + tenure if label is generic
        if c not in SEGMENT_LABELS:
            if churn_rate > 0.55:
                label, icon, color = "High Churn Risk", "🔴", "#ef4444"
            elif churn_rate < 0.15:
                label, icon, color = "Loyal Customers", "🏆", "#10b981"
            else:
                label, icon, color = f"Segment {c+1}", "📊", "#6b7280"

        profile = {
            'cluster':    c,
            'label':      label,
            'icon':       icon,
            'color':      color,
            'count':      int(mask.sum()),
            'pct':        round(100 * mask.sum() / len(df), 1),
            'churn_rate': round(churn_rate * 100, 1),
        }

        for col in feature_cols:
            profile[f'avg_{col}'] = round(float(group[col].mean()), 2)

        if 'tenure' in feature_cols:
            profile['avg_tenure'] = round(float(group['tenure'].mean()), 1)
        if 'MonthlyCharges' in feature_cols:
            profile['avg_monthly'] = round(float(group['MonthlyCharges'].mean()), 2)
        if 'TotalCharges' in feature_cols:
            profile['avg_total'] = round(float(group['TotalCharges'].mean()), 2)

        segments.append(profile)

    # Sort by churn rate descending
    segments.sort(key=lambda s: s['churn_rate'], reverse=True)

    # RFM-style summary
    rfm_summary = _build_rfm(df)

    return {
        'segments':    segments,
        'pca_points':  pca_points,
        'rfm_summary': rfm_summary,
        'n_clusters':  n_clusters,
        'features':    feature_cols,
    }


def _build_rfm(df):
    """
    Telecom RFM proxy:
    - Recency  → inverse of tenure (new = low recency score)
    - Frequency → ServiceCount (number of services subscribed)
    - Monetary  → MonthlyCharges
    """
    rfm = pd.DataFrame()

    if 'tenure' in df.columns:
        max_t = df['tenure'].max() + 1
        rfm['Recency_score'] = (df['tenure'] / max_t * 5).round(1)

    svc_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    svc_cols = [c for c in svc_cols if c in df.columns]
    if svc_cols:
        rfm['Frequency_score'] = df[svc_cols].apply(
            lambda row: sum(1 for v in row if v not in ['No', 'No phone service', 'No internet service']),
            axis=1
        )

    if 'MonthlyCharges' in df.columns:
        rfm['Monetary_score'] = (df['MonthlyCharges'] / df['MonthlyCharges'].max() * 5).round(1)

    if 'Churn_num' in df.columns:
        rfm['Churn'] = df['Churn_num']

    if not rfm.empty:
        return rfm.describe().round(2).to_dict()
    return {}
