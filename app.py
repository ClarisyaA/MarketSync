import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from datetime import datetime

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="MarketSync — Marketing Channel Optimization DSS",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# THEME (light only — fixed)
# =============================================================================
BG          = "#f8fafc"
SIDEBAR     = "#f1f5f9"
CARD        = "#ffffff"
BORDER      = "#e2e8f0"
TEXT1       = "#0f172a"
TEXT2       = "#475569"
MUTED       = "#94a3b8"
ACCENT      = "#0d9488"
AMBER       = "#d97706"
ACCENT_DIM  = "#0d948812"
ACCENT_BDR  = "#0d948840"
PLOT_BG     = "#ffffff"
GRID        = "#f1f5f9"
COLORS      = [ACCENT, AMBER, "#818cf8", "#fb923c", "#34d399", "#60a5fa", "#f472b6", "#a3e635"]

# =============================================================================
# CSS
# =============================================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}
html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; color: {TEXT1}; }}
.stApp {{ background-color: {BG}; }}
section[data-testid="stSidebar"] {{ background-color: {SIDEBAR}; border-right: 1px solid {BORDER}; }}
section[data-testid="stSidebar"] * {{ color: {TEXT2} !important; }}
#footer {{ visibility: hidden; }}
.block-container {{ padding-top: 2rem; padding-bottom: 4rem; max-width: 1280px; }}
h1,h2,h3,h4 {{ font-family:'DM Sans',sans-serif; font-weight:600; letter-spacing:-0.02em; color:{TEXT1}; }}

.ms-masthead {{ display:flex; align-items:flex-end; justify-content:space-between; padding:2rem 0 1.5rem; border-bottom:1px solid {BORDER}; margin-bottom:2rem; }}
.ms-wordmark {{ font-size:1.75rem; font-weight:600; letter-spacing:-0.04em; color:{TEXT1}; line-height:1; }}
.ms-wordmark span {{ color:{ACCENT}; }}
.ms-tagline {{ font-size:0.8rem; color:{MUTED}; font-weight:400; letter-spacing:0.06em; text-transform:uppercase; margin-top:0.35rem; }}
.ms-badge {{ font-family:'DM Mono',monospace; font-size:0.7rem; padding:0.25rem 0.65rem; border-radius:3px; background:{CARD}; color:{ACCENT}; border:1px solid {ACCENT_BDR}; letter-spacing:0.08em; }}

.ms-section {{ display:flex; align-items:center; gap:0.75rem; margin:2.5rem 0 1.25rem; }}
.ms-section-index {{ font-family:'DM Mono',monospace; font-size:0.7rem; color:{ACCENT}; letter-spacing:0.1em; min-width:2rem; }}
.ms-section-title {{ font-size:1rem; font-weight:600; color:{TEXT1}; letter-spacing:-0.01em; }}
.ms-section-line {{ flex:1; height:1px; background:{BORDER}; }}

.ms-stat-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:1px; background:{BORDER}; border:1px solid {BORDER}; border-radius:8px; overflow:hidden; margin-bottom:1.5rem; }}
.ms-stat {{ background:{CARD}; padding:1.25rem 1.5rem; }}
.ms-stat-label {{ font-size:0.7rem; font-weight:500; color:{MUTED}; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:0.5rem; }}
.ms-stat-value {{ font-size:1.6rem; font-weight:600; color:{TEXT1}; letter-spacing:-0.03em; line-height:1; }}
.ms-stat-sub {{ font-size:0.72rem; color:{MUTED}; margin-top:0.35rem; font-family:'DM Mono',monospace; }}
.ms-stat-accent {{ color:{ACCENT}; }}

.ms-info {{ background:{CARD}; border:1px solid {BORDER}; border-left:3px solid {ACCENT}; border-radius:0 6px 6px 0; padding:1.25rem 1.5rem; margin-bottom:1.5rem; }}
.ms-info-title {{ font-size:0.78rem; font-weight:600; color:{ACCENT}; letter-spacing:0.06em; text-transform:uppercase; margin-bottom:0.6rem; }}
.ms-info p, .ms-info li {{ font-size:0.875rem; color:{TEXT2}; line-height:1.7; margin:0; }}
.ms-info ul {{ padding-left:1.25rem; margin:0.5rem 0 0; }}

.ms-warn {{ background:#fffbeb; border:1px solid #f59e0b; border-left:3px solid #f59e0b; border-radius:0 6px 6px 0; padding:1.25rem 1.5rem; margin-bottom:1.5rem; }}
.ms-warn-title {{ font-size:0.78rem; font-weight:600; color:#b45309; letter-spacing:0.06em; text-transform:uppercase; margin-bottom:0.5rem; }}
.ms-warn p {{ font-size:0.875rem; color:#92400e; margin:0; }}

.ms-cluster-card {{ background:{CARD}; border:1px solid {BORDER}; border-radius:10px; padding:1.25rem 1.25rem 1rem; }}
.ms-cluster-id {{ font-family:'DM Mono',monospace; font-size:0.65rem; color:{ACCENT}; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.75rem; border-bottom:1px solid {BORDER}; padding-bottom:0.5rem; }}
.ms-cluster-row {{ display:flex; justify-content:space-between; align-items:baseline; margin-bottom:0.4rem; }}
.ms-cluster-key {{ font-size:0.72rem; color:{MUTED}; font-weight:500; }}
.ms-cluster-val {{ font-size:0.875rem; color:{TEXT1}; font-family:'DM Mono',monospace; font-weight:500; }}

.ms-rec-card {{ background:{CARD}; border:1px solid {BORDER}; border-radius:8px; padding:1.75rem; margin-bottom:1rem; }}
.ms-rec-card.primary {{ border-top:3px solid {ACCENT}; }}
.ms-rec-card.secondary {{ border-top:3px solid {AMBER}; }}
.ms-rec-label {{ font-size:0.65rem; font-weight:600; color:{MUTED}; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.5rem; }}
.ms-rec-value {{ font-size:1.5rem; font-weight:600; color:{TEXT1}; letter-spacing:-0.02em; margin-bottom:0.75rem; }}
.ms-rec-score {{ font-family:'DM Mono',monospace; font-size:0.8rem; color:{ACCENT}; margin-bottom:1rem; }}
.ms-action-list {{ list-style:none; padding:0; margin:0; border-top:1px solid {BORDER}; padding-top:1rem; }}
.ms-action-list li {{ font-size:0.8rem; color:{TEXT2}; padding:0.35rem 0; display:flex; align-items:flex-start; gap:0.5rem; line-height:1.5; }}
.ms-action-list li::before {{ content:""; display:inline-block; width:4px; height:4px; background:{ACCENT}; border-radius:50%; margin-top:0.45rem; flex-shrink:0; }}

.ms-exec {{ background:{CARD}; border:1px solid {BORDER}; border-radius:8px; padding:2rem; margin-bottom:2rem; }}
.ms-exec-header {{ display:flex; align-items:center; justify-content:space-between; margin-bottom:1.5rem; padding-bottom:1rem; border-bottom:1px solid {BORDER}; }}
.ms-exec-title {{ font-size:0.85rem; font-weight:600; color:{TEXT2}; letter-spacing:0.04em; text-transform:uppercase; }}
.ms-exec-cluster {{ font-family:'DM Mono',monospace; font-size:0.75rem; color:{ACCENT}; background:{ACCENT_DIM}; border:1px solid {ACCENT_BDR}; padding:0.2rem 0.6rem; border-radius:3px; }}
.ms-exec-profile {{ display:inline-block; font-size:0.7rem; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; padding:0.2rem 0.65rem; border-radius:3px; }}
.ms-profile-premium {{ background:{ACCENT_DIM}; color:{ACCENT}; border:1px solid {ACCENT_BDR}; }}
.ms-profile-mid {{ background:#fef3c715; color:{AMBER}; border:1px solid #f59e0b40; }}
.ms-profile-value {{ background:{BORDER}; color:{TEXT2}; border:1px solid {BORDER}; }}

div[data-baseweb="tab-list"] {{ background:transparent; border-bottom:1px solid {BORDER}; gap:4px; }}
div[data-baseweb="tab"] {{ background:transparent !important; border:none !important; color:{MUTED} !important; font-size:0.85rem; font-weight:500; padding:0.75rem 1.5rem !important; border-bottom:2px solid transparent !important; margin-right:0.25rem; border-radius:0 !important; transition:color 0.15s; }}
div[aria-selected="true"][data-baseweb="tab"] {{ color:{TEXT1} !important; border-bottom-color:{ACCENT} !important; }}

.ms-sidebar-label {{ font-size:0.65rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; color:{MUTED}; margin-bottom:0.5rem; padding-top:1rem; }}
.ms-sidebar-section {{ border-top:1px solid {BORDER}; padding-top:1rem; margin-top:1rem; }}
.ms-sidebar-meta {{ font-size:0.75rem; color:{MUTED}; line-height:1.7; }}

div[data-testid="stExpander"] > details {{ background:{CARD}; border:1px solid {BORDER}; border-radius:6px; }}
div[data-testid="stExpander"] summary {{ color:{TEXT2}; font-size:0.875rem; font-weight:500; }}
.stButton button {{ background:{CARD}; color:{TEXT1}; border:1px solid {BORDER}; font-family:'DM Sans',sans-serif; font-size:0.85rem; font-weight:500; border-radius:6px; transition:all 0.15s; }}
.stButton button:hover {{ border-color:{ACCENT}; color:{ACCENT}; }}
.stButton button[kind="primary"] {{ background:{ACCENT} !important; color:#fff !important; border-color:{ACCENT} !important; font-weight:600; }}
.stDownloadButton button {{ background:transparent; border:1px solid {BORDER}; color:{TEXT2}; font-size:0.8rem; }}
.stDownloadButton button:hover {{ border-color:{ACCENT_BDR}; color:{ACCENT}; }}
div[data-testid="metric-container"] {{ background:{CARD}; border:1px solid {BORDER}; border-radius:6px; padding:1rem; }}
div[data-testid="stDataFrame"] {{ border:1px solid {BORDER}; border-radius:6px; overflow:hidden; }}

.ms-action-table {{ width:100%; border-collapse:collapse; font-size:0.85rem; }}
.ms-action-table th {{ text-align:left; font-size:0.65rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; color:{MUTED}; padding:0.6rem 0.75rem; border-bottom:1px solid {BORDER}; }}
.ms-action-table td {{ padding:0.75rem; color:{TEXT2}; border-bottom:1px solid {BORDER}; vertical-align:top; line-height:1.55; }}
.ms-action-table tr:last-child td {{ border-bottom:none; }}
.ms-action-table td:first-child {{ color:{TEXT1}; font-weight:500; white-space:nowrap; }}
.ms-action-table .highlight-row td {{ color:{ACCENT}; }}

.ms-footer {{ margin-top:4rem; padding-top:2rem; border-top:1px solid {BORDER}; display:flex; justify-content:space-between; align-items:center; }}
.ms-footer-left {{ font-size:0.78rem; color:{MUTED}; }}
.ms-footer-right {{ font-family:'DM Mono',monospace; font-size:0.68rem; color:{MUTED}; letter-spacing:0.08em; }}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# PLOTLY
# =============================================================================
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=PLOT_BG,
    font=dict(family="DM Sans", color=TEXT2, size=12),
    xaxis=dict(gridcolor=GRID, linecolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=MUTED, size=11)),
    yaxis=dict(gridcolor=GRID, linecolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=MUTED, size=11)),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER, borderwidth=1, font=dict(color=TEXT2, size=11)),
    margin=dict(l=16, r=16, t=40, b=16),
    title_font=dict(size=13, color=TEXT1, family="DM Sans"),
    colorway=COLORS,
)
TEAL_SCALE = [[0.0, "#e0f2fe"], [0.5, "#0891b2"], [1.0, "#0d9488"]]

def apply_theme(fig, title=None):
    fig.update_layout(**PLOTLY_LAYOUT)
    if title:
        fig.update_layout(title_text=title)
    return fig

# =============================================================================
# CONSTANTS
# =============================================================================
REQUIRED_COLS = [
    'Year_Birth', 'Income', 'Dt_Customer',
    'NumStorePurchases', 'NumWebPurchases', 'NumCatalogPurchases',
    'MntWines', 'MntFruits', 'MntMeatProducts',
    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
]
MNT_COLS = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
PROD_LABELS = {
    'MntWines': 'Wines', 'MntFruits': 'Fruits', 'MntMeatProducts': 'Meat',
    'MntFishProducts': 'Fish', 'MntSweetProducts': 'Sweets', 'MntGoldProds': 'Gold',
}
DATASET_SCHEMA = {
    'ID': 'Unique customer identifier', 'Year_Birth': 'Customer birth year',
    'Education': 'Education level', 'Marital_Status': 'Marital status',
    'Income': 'Annual household income (USD)', 'Kidhome': 'Number of children at home',
    'Teenhome': 'Number of teenagers at home', 'Dt_Customer': 'Customer enrollment date',
    'Recency': 'Days since last purchase', 'MntWines': 'Wine spend — 2yr (USD)',
    'MntFruits': 'Fruit spend — 2yr (USD)', 'MntMeatProducts': 'Meat spend — 2yr (USD)',
    'MntFishProducts': 'Fish spend — 2yr (USD)', 'MntSweetProducts': 'Sweets spend — 2yr (USD)',
    'MntGoldProds': 'Gold spend — 2yr (USD)', 'NumDealsPurchases': 'Discounted purchases',
    'NumWebPurchases': 'Web channel purchases', 'NumCatalogPurchases': 'Catalog channel purchases',
    'NumStorePurchases': 'In-store purchases', 'NumWebVisitsMonth': 'Website visits per month',
}

# =============================================================================
# HELPERS
# =============================================================================
def load_data(f):
    try:
        df = pd.read_csv(f, sep="\t")
        if df.shape[1] < 2:
            f.seek(0)
            df = pd.read_csv(f, sep=",")
        return df
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        return None

def check_cols(df):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        return False
    return True

def process_pipeline(df):
    logs, df_c = [], df.copy()
    missing_n = int(df_c['Income'].isnull().sum())
    median_inc = df_c['Income'].median()
    df_c['Income'] = df_c['Income'].fillna(median_inc)
    rows_init = len(df_c)
    df_c = df_c.dropna()
    logs.append({"step": "01", "name": "Data Cleaning & Imputation",
                 "desc": f"Income missing values ({missing_n} rows) filled with median (${median_inc:,.0f}). Remaining nulls dropped.",
                 "stats": {"Missing Filled": missing_n, "After Dropna": len(df_c), "Initial Rows": rows_init},
                 "code": "df['Income'].fillna(df['Income'].median(), inplace=True)\ndf.dropna(inplace=True)"})

    cy = datetime.now().year
    df_c['Age'] = cy - df_c['Year_Birth']
    df_c['Total_Spend'] = df_c[MNT_COLS].sum(axis=1)
    try:
        df_c['Dt_Customer'] = pd.to_datetime(df_c['Dt_Customer'], dayfirst=True)
    except Exception:
        df_c['Dt_Customer'] = pd.to_datetime(df_c['Dt_Customer'], errors='coerce')
    ref = pd.to_datetime(f"{cy}-12-31")
    df_c['Tenure_Months'] = (ref - df_c['Dt_Customer']).dt.days / 30
    logs.append({"step": "02", "name": "Feature Engineering",
                 "desc": f"Derived Age, Total_Spend, Tenure_Months. Reference year: {cy}.",
                 "features": [("Age", f"{cy} − Year_Birth", "Age in years"),
                               ("Total_Spend", "Sum of all Mnt* cols", "2-year spend"),
                               ("Tenure_Months", "(Ref − Dt_Customer) / 30", "Months as customer")],
                 "preview": df_c[['Age', 'Total_Spend', 'Tenure_Months', 'Income']].describe().round(2)})

    n_before = len(df_c)
    df_c = df_c[(df_c['Age'] < 90) & (df_c['Income'].between(1, 600_000))]
    zs = np.abs(stats.zscore(df_c[['Income', 'Total_Spend', 'Age']]))
    df_c = df_c[(zs < 3).all(axis=1)]
    logs.append({"step": "03", "name": "Outlier Removal",
                 "desc": "Domain constraints (Age<90, 0<Income<600k) then Z-score |z|<3.",
                 "stats": {"Before": n_before, "Removed": n_before-len(df_c), "Retained": len(df_c), "Rate": f"{len(df_c)/n_before*100:.1f}%"}})

    df_c['Log_Income'] = np.log1p(df_c['Income'])
    df_c['Log_Spend']  = np.log1p(df_c['Total_Spend'])
    features = ['Age', 'Tenure_Months', 'Log_Income', 'Log_Spend']
    scaler = StandardScaler()
    X_std  = scaler.fit_transform(df_c[features])
    logs.append({"step": "04", "name": "Log Transform + Standardisation",
                 "desc": "Log1p on monetary features. StandardScaler → μ=0, σ=1.",
                 "stats": {"Features": ", ".join(features), "Formula": "z = (x−μ)/σ"},
                 "preview": pd.DataFrame(X_std[:8], columns=features).round(4)})

    pca_model = PCA(n_components=2, random_state=42)
    X_pca = pca_model.fit_transform(X_std)
    df_c['PC1'], df_c['PC2'] = X_pca[:, 0], X_pca[:, 1]
    ev = pca_model.explained_variance_ratio_ * 100
    logs.append({"step": "05", "name": "Principal Component Analysis",
                 "desc": f"4D → 2D. Variance captured: {ev.sum():.1f}% (PC1:{ev[0]:.1f}%, PC2:{ev[1]:.1f}%).",
                 "stats": {"PC1": f"{ev[0]:.2f}%", "PC2": f"{ev[1]:.2f}%", "Total": f"{ev.sum():.2f}%"},
                 "loadings": pd.DataFrame(pca_model.components_, columns=features, index=['PC1','PC2']).round(4),
                 "preview": pd.DataFrame(X_pca[:8], columns=['PC1','PC2']).round(4)})
    return df_c, X_pca, X_std, logs, scaler, pca_model

def get_dominant_product(df_sub):
    if df_sub.empty:
        return "N/A", 0.0
    avgs = df_sub[MNT_COLS].mean()
    top  = avgs.idxmax()
    return PROD_LABELS.get(top, top), float(avgs[top])

def sdm_weights(df_matrix):
    sc = MinMaxScaler()
    normed = pd.DataFrame(sc.fit_transform(df_matrix), columns=df_matrix.columns)
    stds = normed.std().replace(0, 1e-6)
    w = stds / stds.sum()
    return w.values, stds

def section(index, title):
    st.markdown(
        f'<div class="ms-section">'
        f'<span class="ms-section-index">{index}</span>'
        f'<span class="ms-section-title">{title}</span>'
        f'<span class="ms-section-line"></span></div>',
        unsafe_allow_html=True,
    )

def render_cluster_card(cid_card, row):
    """Render one cluster card — kept short to avoid Streamlit HTML truncation."""
    st.markdown(
        f'<div class="ms-cluster-card">'
        f'<div class="ms-cluster-id">Cluster {cid_card}</div>'
        f'<div class="ms-cluster-row"><span class="ms-cluster-key">Avg Age</span>'
        f'<span class="ms-cluster-val">{row["Age"]:.0f} yr</span></div>'
        f'<div class="ms-cluster-row"><span class="ms-cluster-key">Avg Spend</span>'
        f'<span class="ms-cluster-val">${row["Total_Spend"]:,.0f}</span></div>'
        f'<div class="ms-cluster-row"><span class="ms-cluster-key">Avg Income</span>'
        f'<span class="ms-cluster-val">${row["Income"]:,.0f}</span></div>'
        f'<div class="ms-cluster-row"><span class="ms-cluster-key">Tenure</span>'
        f'<span class="ms-cluster-val">{row["Tenure_Months"]:.0f} mo</span></div>'
        f'<div class="ms-cluster-row"><span class="ms-cluster-key">Size</span>'
        f'<span class="ms-cluster-val">{int(row["Size"]):,} ({row["Pct"]:.0f}%)</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown(
        f'<p style="font-size:1.1rem;font-weight:600;letter-spacing:-0.03em;color:{TEXT1};">'
        f'Market<span style="color:{ACCENT};">Sync</span></p>',
        unsafe_allow_html=True,
    )
    st.markdown('<p class="ms-sidebar-meta">Marketing Channel Optimization<br>Decision Support System</p>', unsafe_allow_html=True)
    st.markdown('<div class="ms-sidebar-section">', unsafe_allow_html=True)
    st.markdown('<p class="ms-sidebar-label">Data Source</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV", type=["csv", "txt"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    with st.expander("Dataset Reference", expanded=False):
        st.markdown(
            f'<p class="ms-sidebar-meta">Customer Personality Analysis<br>'
            f'<a href="https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis" '
            f'style="color:{ACCENT};text-decoration:none;">Kaggle Dataset ↗</a><br><br>'
            f'2,240 customers — demographic, transactional, and campaign data.</p>',
            unsafe_allow_html=True,
        )
    st.markdown('<div class="ms-sidebar-section">', unsafe_allow_html=True)
    st.markdown('<p class="ms-sidebar-label">Methodology</p>', unsafe_allow_html=True)
    st.markdown('<p class="ms-sidebar-meta">PCA — Dimensionality reduction<br>K-Means — Customer segmentation<br>SAW / MCDM — Channel ranking</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# MASTHEAD
# =============================================================================
st.markdown(
    f'<div class="ms-masthead">'
    f'<div><div class="ms-wordmark">Market<span>Sync</span></div>'
    f'<div class="ms-tagline">Marketing Channel Optimization — Decision Support System</div></div>'
    f'<div class="ms-badge">DSS</div></div>',
    unsafe_allow_html=True,
)

# =============================================================================
# EMPTY STATE
# =============================================================================
if uploaded_file is None:
    st.markdown(
        f'<div class="ms-info" style="padding:2.5rem;text-align:center;">'
        f'<div class="ms-info-title" style="font-size:1rem;margin-bottom:1rem;">No Data Loaded</div>'
        f'<p>Upload the marketing campaign CSV via the sidebar.<br>'
        f'<a href="https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis" style="color:{ACCENT};">Customer Personality Analysis on Kaggle</a></p>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# =============================================================================
# LOAD & VALIDATE
# =============================================================================
df_raw = load_data(uploaded_file)
if df_raw is None or not check_cols(df_raw):
    st.stop()

with st.sidebar:
    st.markdown('<div class="ms-sidebar-section">', unsafe_allow_html=True)
    st.markdown('<p class="ms-sidebar-label">Loaded Dataset</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="ms-sidebar-meta">'
        f'Rows: <strong style="color:{TEXT1};">{len(df_raw):,}</strong><br>'
        f'Columns: <strong style="color:{TEXT1};">{df_raw.shape[1]}</strong><br>'
        f'Memory: <strong style="color:{TEXT1};">{df_raw.memory_usage(deep=True).sum()/1e6:.2f} MB</strong></p>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# PIPELINE
# =============================================================================
df_clean, X_pca, X_std, logs, scaler, pca_model = process_pipeline(df_raw)

# =============================================================================
# DATASET OVERVIEW
# =============================================================================
with st.expander("Dataset Overview", expanded=False):
    c1, c2, c3 = st.columns(3)
    c1.metric("Records", f"{len(df_raw):,}")
    c2.metric("Features", f"{df_raw.shape[1]}")
    c3.metric("Memory", f"{df_raw.memory_usage(deep=True).sum()/1e6:.2f} MB")
    st.markdown("**Column Schema**")
    st.dataframe(pd.DataFrame(DATASET_SCHEMA.items(), columns=["Column", "Description"]), use_container_width=True, height=260)
    st.markdown("**Sample Records**")
    st.dataframe(df_raw.head(5), use_container_width=True)

# =============================================================================
# TABS
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "  Preprocessing Pipeline  ",
    "  Cluster Analysis  ",
    "  SAW Decision Matrix  ",
    "  Business Recommendations  ",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    section("01", "Preprocessing Pipeline")
    st.markdown(
        '<div class="ms-info"><div class="ms-info-title">Pipeline Overview</div>'
        '<p>Five-stage pipeline transforming raw customer data into a PCA-reduced feature space for K-Means clustering.</p></div>',
        unsafe_allow_html=True,
    )
    for log in logs:
        with st.expander(f"Step {log['step']} — {log['name']}", expanded=(log['step'] == '01')):
            st.markdown(f"<p style='font-size:0.875rem;color:{TEXT2};margin-bottom:1rem;'>{log['desc']}</p>", unsafe_allow_html=True)
            if 'code' in log:
                st.code(log['code'], language='python')
            if 'features' in log:
                st.dataframe(pd.DataFrame(log['features'], columns=['Feature', 'Formula', 'Description']), use_container_width=True)
            if 'stats' in log:
                sc_ = st.columns(len(log['stats']))
                for col_s, (k, v) in zip(sc_, log['stats'].items()):
                    col_s.metric(k, v)
            if 'loadings' in log:
                st.markdown("**PCA Loadings**")
                st.dataframe(log['loadings'].style.background_gradient(cmap='RdYlGn', axis=1).format("{:.4f}"), use_container_width=True)
            if 'preview' in log:
                st.markdown("**Preview**")
                st.dataframe(log['preview'], use_container_width=True)
    st.markdown("---")
    dl1, dl2 = st.columns(2)
    dl1.download_button("Download Cleaned Data", df_clean.to_csv(index=False).encode(), 'data_cleaned.csv', 'text/csv', use_container_width=True)
    dl2.download_button("Download PCA Coordinates", df_clean[['PC1', 'PC2']].to_csv(index=False).encode(), 'pca_results.csv', 'text/csv', use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    section("02", "Cluster Analysis")

    with st.expander("Elbow Method — Optimal K Selection", expanded=True):
        inertias, sil_scores = [], []
        for k in range(2, 10):
            km_tmp  = KMeans(n_clusters=k, random_state=42, n_init=10)
            lbl_tmp = km_tmp.fit_predict(X_pca)
            inertias.append(km_tmp.inertia_)
            sil_scores.append(silhouette_score(X_pca, lbl_tmp))

        pts = list(range(2, 10))
        p1  = np.array([pts[0], inertias[0]])
        p2  = np.array([pts[-1], inertias[-1]])
        dists = [np.abs(np.cross(p2-p1, p1-np.array([pts[i], inertias[i]]))) / np.linalg.norm(p2-p1) for i in range(len(pts))]
        best_k = pts[int(np.argmax(dists))]

        fig_elbow = make_subplots(specs=[[{"secondary_y": True}]])
        fig_elbow.add_trace(go.Scatter(x=pts, y=inertias, mode='lines+markers', name='Inertia',
                                       line=dict(color=ACCENT, width=2.5), marker=dict(size=7, color=ACCENT)), secondary_y=False)
        fig_elbow.add_trace(go.Scatter(x=pts, y=sil_scores, mode='lines+markers', name='Silhouette',
                                       line=dict(color=AMBER, width=2, dash='dot'), marker=dict(size=6, color=AMBER)), secondary_y=True)
        fig_elbow.add_vline(x=best_k, line_dash="dash", line_color=MUTED,
                            annotation_text=f"K={best_k}", annotation_font_color=TEXT1, annotation_font_size=11)
        fig_elbow.update_yaxes(title_text="Inertia (WCSS)", secondary_y=False, title_font=dict(color=ACCENT), gridcolor=GRID)
        fig_elbow.update_yaxes(title_text="Silhouette Score", secondary_y=True, title_font=dict(color=AMBER), gridcolor=GRID)
        fig_elbow.update_layout(**PLOTLY_LAYOUT, height=360, title_text="Elbow Method + Silhouette Coefficient")
        st.plotly_chart(fig_elbow, use_container_width=True)
        st.markdown(f'<div class="ms-info"><div class="ms-info-title">Recommendation</div>'
                    f'<p>Geometric elbow suggests <strong style="color:{ACCENT};">K = {best_k}</strong> as optimal.</p></div>',
                    unsafe_allow_html=True)

    col_ks, _ = st.columns([1, 3])
    with col_ks:
        k_clusters = st.number_input("Number of Clusters (K)", min_value=2, max_value=8, value=int(best_k), step=1)

    kmeans   = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_pca)
    df_clean['Cluster'] = clusters

    section("", "Cluster Quality Metrics")
    sil = silhouette_score(X_pca, clusters)
    dbi = davies_bouldin_score(X_pca, clusters)
    chi = calinski_harabasz_score(X_pca, clusters)
    st.markdown(
        f'<div class="ms-stat-grid">'
        f'<div class="ms-stat"><div class="ms-stat-label">Silhouette Score</div>'
        f'<div class="ms-stat-value ms-stat-accent">{sil:.3f}</div>'
        f'<div class="ms-stat-sub">&gt; 0.50 — higher better</div></div>'
        f'<div class="ms-stat"><div class="ms-stat-label">Davies-Bouldin</div>'
        f'<div class="ms-stat-value">{dbi:.3f}</div>'
        f'<div class="ms-stat-sub">&lt; 1.00 — lower better</div></div>'
        f'<div class="ms-stat"><div class="ms-stat-label">Calinski-Harabasz</div>'
        f'<div class="ms-stat-value">{chi:.0f}</div>'
        f'<div class="ms-stat-sub">Higher is better</div></div>'
        f'<div class="ms-stat"><div class="ms-stat-label">Segments</div>'
        f'<div class="ms-stat-value">{k_clusters}</div>'
        f'<div class="ms-stat-sub">{len(df_clean):,} customers</div></div>'
        f'</div>', unsafe_allow_html=True,
    )

    section("", "Visualisations")
    vtab1, vtab2, vtab3 = st.tabs(["  PCA Scatter  ", "  Segment Distribution  ", "  Feature Comparison  "])

    with vtab1:
        fig_sc = px.scatter(df_clean, x='PC1', y='PC2', color=df_clean['Cluster'].astype(str),
                            hover_data={'Age': True, 'Income': ':,.0f', 'Total_Spend': ':,.0f'},
                            color_discrete_sequence=COLORS, labels={'color': 'Cluster'}, title="PCA Space — Customer Segments")
        fig_sc.update_traces(marker=dict(size=5, opacity=0.75))
        apply_theme(fig_sc)
        fig_sc.update_layout(height=480)
        st.plotly_chart(fig_sc, use_container_width=True)

    with vtab2:
        counts = df_clean['Cluster'].value_counts().sort_index()
        cv1, cv2 = st.columns(2)
        with cv1:
            fig_pie = go.Figure(go.Pie(labels=[f'C{i}' for i in counts.index], values=counts.values,
                                       hole=0.55, marker=dict(colors=COLORS), textfont=dict(color=TEXT2, size=11)))
            apply_theme(fig_pie, "Segment Share")
            fig_pie.update_layout(height=340)
            st.plotly_chart(fig_pie, use_container_width=True)
        with cv2:
            fig_bar_c = go.Figure(go.Bar(x=[f'Cluster {i}' for i in counts.index], y=counts.values,
                                          marker=dict(color=COLORS[:len(counts)], line=dict(width=0)),
                                          text=counts.values, textposition='outside', textfont=dict(color=MUTED, size=11)))
            apply_theme(fig_bar_c, "Count per Segment")
            fig_bar_c.update_layout(height=340, showlegend=False)
            st.plotly_chart(fig_bar_c, use_container_width=True)

    with vtab3:
        cm_viz = df_clean.groupby('Cluster')[['Age', 'Total_Spend', 'Tenure_Months', 'Income']].mean()
        feats4 = ['Age', 'Total_Spend', 'Tenure_Months', 'Income']
        titles4 = ['Avg Age (yrs)', 'Avg Total Spend ($)', 'Avg Tenure (mo)', 'Avg Income ($)']
        fig_comp = make_subplots(rows=2, cols=2, subplot_titles=titles4, vertical_spacing=0.14, horizontal_spacing=0.1)
        for feat, (r, c) in zip(feats4, [(1,1),(1,2),(2,1),(2,2)]):
            for ci in cm_viz.index:
                fig_comp.add_trace(go.Bar(name=f'C{ci}', x=[f'C{ci}'], y=[cm_viz.loc[ci, feat]],
                                          marker_color=COLORS[ci % len(COLORS)], showlegend=(r==1 and c==1)), row=r, col=c)
        fig_comp.update_layout(**PLOTLY_LAYOUT, height=520, title_text="Feature Comparison Across Segments")
        st.plotly_chart(fig_comp, use_container_width=True)

    section("", "Segment Profiles")
    cluster_means = df_clean.groupby('Cluster')[['Age', 'Total_Spend', 'Tenure_Months', 'Income']].mean()
    cluster_means['Size'] = df_clean['Cluster'].value_counts().sort_index()
    cluster_means['Pct']  = (cluster_means['Size'] / cluster_means['Size'].sum() * 100).round(1)

    st.dataframe(
        cluster_means.style
        .format({'Age': '{:.1f}', 'Total_Spend': '${:,.0f}', 'Tenure_Months': '{:.0f}',
                 'Income': '${:,.0f}', 'Size': '{:.0f}', 'Pct': '{:.1f}%'})
        .background_gradient(subset=['Total_Spend', 'Income'], cmap='YlGn'),
        use_container_width=True,
    )

    # ── Cluster Cards — rendered one per column (avoids HTML truncation) ──
    st.markdown(f"<p style='font-size:0.75rem;color:{MUTED};margin:1.5rem 0 0.75rem;font-weight:600;letter-spacing:0.06em;text-transform:uppercase;'>Cluster Cards</p>", unsafe_allow_html=True)
    card_cols = st.columns(k_clusters)
    for idx in range(k_clusters):
        if idx in cluster_means.index:
            with card_cols[idx]:
                render_cluster_card(idx, cluster_means.loc[idx])

    st.session_state['df_clustered']  = df_clean
    st.session_state['cluster_means'] = cluster_means

    st.markdown("---")
    st.download_button("Download Cluster Results", df_clean.to_csv(index=False).encode(),
                       'clustering_results.csv', 'text/csv', use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    section("03", "SAW Decision Matrix")

    if 'df_clustered' not in st.session_state:
        st.markdown('<div class="ms-warn"><div class="ms-warn-title">Prerequisites Missing</div>'
                    '<p>Complete Cluster Analysis (Tab 02) first.</p></div>', unsafe_allow_html=True)
        st.stop()

    df_final = st.session_state['df_clustered']
    means    = st.session_state['cluster_means']

    st.markdown(
        '<div class="ms-info"><div class="ms-info-title">Method — Simple Additive Weighting (SAW)</div>'
        '<p>MCDM technique evaluating channels against four weighted criteria via Standard Deviation Method.</p>'
        '<ul><li><strong>K1 — Age Fit</strong></li><li><strong>K2 — Spend Potential</strong></li>'
        '<li><strong>K3 — Loyalty (Tenure)</strong></li><li><strong>K4 — Channel Intensity</strong></li></ul></div>',
        unsafe_allow_html=True,
    )
    st.latex(r'V_i = \sum_{j=1}^{n} w_j \cdot r_{ij}, \quad r_{ij} = \frac{x_{ij}}{\max_j(x_{ij})}')

    cs, ci_ = st.columns([1, 2])
    with cs:
        target_c = st.selectbox("Select Segment", options=sorted(means.index.tolist()), format_func=lambda x: f"Cluster {x}")
    with ci_:
        ri = means.loc[target_c]
        st.markdown(
            f'<div class="ms-stat-grid" style="margin-top:0.25rem;">'
            f'<div class="ms-stat"><div class="ms-stat-label">Size</div><div class="ms-stat-value">{int(ri["Size"]):,}</div></div>'
            f'<div class="ms-stat"><div class="ms-stat-label">Avg Spend</div><div class="ms-stat-value">${ri["Total_Spend"]:,.0f}</div></div>'
            f'<div class="ms-stat"><div class="ms-stat-label">Avg Income</div><div class="ms-stat-value">${ri["Income"]:,.0f}</div></div>'
            f'</div>', unsafe_allow_html=True,
        )

    df_target = df_final[df_final['Cluster'] == target_c]
    CHANNELS  = {'In-Store': 'NumStorePurchases', 'Web': 'NumWebPurchases', 'Catalog': 'NumCatalogPurchases'}
    rows = []
    for lbl, col_ch in CHANNELS.items():
        vol = df_target[col_ch].sum()
        if vol > 0:
            k1 = (df_target['Age'] * df_target[col_ch]).sum() / vol
            k2 = (df_target['Total_Spend'] * df_target[col_ch]).sum() / vol
            k3 = (df_target['Tenure_Months'] * df_target[col_ch]).sum() / vol
        else:
            k1 = k2 = k3 = 0.0
        rows.append({'Channel': lbl, 'K1_Age_Fit': k1, 'K2_Spend_Potential': k2, 'K3_Loyalty': k3, 'K4_Intensity': float(vol)})
    X_mat = pd.DataFrame(rows).set_index('Channel')

    st.markdown(f"<h4 style='color:{TEXT1};margin-top:1.5rem;'>Step 1 — Decision Matrix</h4>", unsafe_allow_html=True)
    st.dataframe(X_mat.style.format("{:.2f}").background_gradient(cmap='YlGn'), use_container_width=True)

    if st.button("Run SAW Analysis", type="primary"):
        ws, stds = sdm_weights(X_mat)
        w_df = pd.DataFrame({'Criterion': X_mat.columns, 'Std Dev': stds.values, 'Weight': ws, 'Weight (%)': ws*100})

        st.markdown(f"<h4 style='color:{TEXT1};margin-top:1.5rem;'>Step 2 — Criteria Weights</h4>", unsafe_allow_html=True)
        wc1, wc2 = st.columns(2)
        with wc1:
            st.dataframe(w_df.style.format({'Std Dev': '{:.4f}', 'Weight': '{:.4f}', 'Weight (%)': '{:.2f}%'})
                         .background_gradient(subset=['Weight'], cmap='YlGn'), use_container_width=True)
            st.caption(f"Σ weights = {ws.sum():.4f}")
        with wc2:
            fig_w = go.Figure(go.Bar(x=w_df['Criterion'], y=w_df['Weight (%)'],
                                     marker=dict(color=w_df['Weight (%)'], colorscale=TEAL_SCALE, line=dict(width=0)),
                                     text=w_df['Weight (%)'].round(1), texttemplate='%{text:.1f}%', textposition='outside',
                                     textfont=dict(color=MUTED, size=11)))
            apply_theme(fig_w, "Weight Distribution")
            fig_w.update_layout(height=280, showlegend=False)
            st.plotly_chart(fig_w, use_container_width=True)

        st.markdown(f"<h4 style='color:{TEXT1};margin-top:1.5rem;'>Step 3 — Normalised Matrix</h4>", unsafe_allow_html=True)
        R = X_mat.copy()
        for c in X_mat.columns:
            mx = X_mat[c].max()
            R[c] = X_mat[c] / mx if mx > 0 else 0.0
        st.dataframe(R.style.format("{:.4f}").background_gradient(cmap='YlGn'), use_container_width=True)

        st.markdown(f"<h4 style='color:{TEXT1};margin-top:1.5rem;'>Step 4 — SAW Scores & Ranking</h4>", unsafe_allow_html=True)
        scores = [sum(R.iloc[i, j] * ws[j] for j in range(len(ws))) for i in range(len(R))]
        res = X_mat.copy()
        res['SAW_Score'] = scores
        res['Rank'] = pd.Series(scores).rank(ascending=False).astype(int).values
        res = res.sort_values('Rank')
        st.session_state['saw_res']   = res
        st.session_state['target_id'] = target_c

        rc1, rc2 = st.columns(2)
        with rc1:
            st.dataframe(res[['SAW_Score', 'Rank']].style.format({'SAW_Score': '{:.4f}'})
                         .highlight_max(subset=['SAW_Score'], color='#d1fae5'), use_container_width=True)
        with rc2:
            rk_df = res.reset_index().sort_values('Rank')
            bar_clrs = [ACCENT if r == 1 else BORDER for r in rk_df['Rank']]
            fig_rank = go.Figure(go.Bar(x=rk_df['Channel'], y=rk_df['SAW_Score'],
                                         marker=dict(color=bar_clrs, line=dict(width=0)),
                                         text=rk_df['SAW_Score'].round(4), texttemplate='%{text:.4f}', textposition='outside',
                                         textfont=dict(color=MUTED, size=11)))
            apply_theme(fig_rank, "Channel Ranking")
            fig_rank.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_rank, use_container_width=True)

        st.download_button("Download SAW Results", res.to_csv().encode(), f"saw_cluster_{target_c}.csv", "text/csv", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    section("04", "Business Recommendations")

    if 'saw_res' not in st.session_state:
        st.markdown('<div class="ms-warn"><div class="ms-warn-title">Prerequisites Missing</div>'
                    '<p>Run SAW Analysis in Tab 03 first.</p></div>', unsafe_allow_html=True)
        st.stop()

    res      = st.session_state['saw_res']
    cid      = st.session_state['target_id']
    df_final = st.session_state['df_clustered']
    df_tgt   = df_final[df_final['Cluster'] == cid]

    winner       = res.index[0]
    winner_score = res.iloc[0]['SAW_Score']
    runner_up    = res.index[1] if len(res) > 1 else winner
    runner_score = res.loc[runner_up, 'SAW_Score']
    top_prod, top_val = get_dominant_product(df_tgt)

    avg_age    = df_tgt['Age'].mean()
    avg_spend  = df_tgt['Total_Spend'].mean()
    avg_income = df_tgt['Income'].mean()
    population = len(df_tgt)

    q75 = df_final['Total_Spend'].quantile(0.75)
    q50 = df_final['Total_Spend'].median()
    if avg_spend > q75:
        profile_label, profile_class = "Premium", "ms-profile-premium"
    elif avg_spend > q50:
        profile_label, profile_class = "Mid-Tier", "ms-profile-mid"
    else:
        profile_label, profile_class = "Value", "ms-profile-value"

    st.markdown(
        f'<div class="ms-exec">'
        f'<div class="ms-exec-header"><span class="ms-exec-title">Executive Summary</span>'
        f'<span class="ms-exec-cluster">Cluster {cid}</span></div>'
        f'<div class="ms-stat-grid" style="border:none;background:transparent;gap:1rem;">'
        f'<div class="ms-stat" style="border:1px solid {BORDER};border-radius:6px;">'
        f'<div class="ms-stat-label">Segment Profile</div>'
        f'<div><span class="{profile_class} ms-exec-profile">{profile_label}</span></div></div>'
        f'<div class="ms-stat" style="border:1px solid {BORDER};border-radius:6px;">'
        f'<div class="ms-stat-label">Population</div><div class="ms-stat-value">{population:,}</div>'
        f'<div class="ms-stat-sub">{population/len(df_final)*100:.1f}% of base</div></div>'
        f'<div class="ms-stat" style="border:1px solid {BORDER};border-radius:6px;">'
        f'<div class="ms-stat-label">Avg Age</div><div class="ms-stat-value">{avg_age:.0f}</div>'
        f'<div class="ms-stat-sub">years</div></div>'
        f'<div class="ms-stat" style="border:1px solid {BORDER};border-radius:6px;">'
        f'<div class="ms-stat-label">Avg 2yr Spend</div>'
        f'<div class="ms-stat-value ms-stat-accent">${avg_spend:,.0f}</div></div>'
        f'<div class="ms-stat" style="border:1px solid {BORDER};border-radius:6px;">'
        f'<div class="ms-stat-label">Avg Income</div><div class="ms-stat-value">${avg_income:,.0f}</div></div>'
        f'</div></div>', unsafe_allow_html=True,
    )

    section("", "Strategic Recommendations")
    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown(
            f'<div class="ms-rec-card primary">'
            f'<div class="ms-rec-label">Primary Channel — Rank 1</div>'
            f'<div class="ms-rec-value">{winner}</div>'
            f'<div class="ms-rec-score">SAW Score: {winner_score:.4f}</div>'
            f'<ul class="ms-action-list">'
            f'<li>Allocate 50–60% of campaign budget</li>'
            f'<li>A/B test messaging and creative variants</li>'
            f'<li>Monitor CAC weekly, set ROMI threshold at 3×</li>'
            f'<li>Instrument full conversion funnel</li>'
            f'</ul></div>', unsafe_allow_html=True,
        )
    with rc2:
        st.markdown(
            f'<div class="ms-rec-card secondary">'
            f'<div class="ms-rec-label">Secondary Channel — Rank 2</div>'
            f'<div class="ms-rec-value">{runner_up}</div>'
            f'<div class="ms-rec-score">SAW Score: {runner_score:.4f}</div>'
            f'<ul class="ms-action-list">'
            f'<li>Allocate 25–30% as retargeting layer</li>'
            f'<li>Use for cross-sell and upsell messaging</li>'
            f'<li>Coordinate timing with primary channel</li>'
            f'<li>Track incremental lift over baseline</li>'
            f'</ul></div>', unsafe_allow_html=True,
        )

    section("", "Product Portfolio Analysis")
    st.markdown("""
    <style>
    button[data-baseweb="tab"] {
        margin-right: 20px;
        padding: 8px 18px;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

    ptab1, ptab2, ptab3 = st.tabs(["  Product Mix  ", "  Channel Performance  ", "  Segment Demographics  "])

    with ptab1:
        prod_avgs = df_tgt[MNT_COLS].mean().sort_values(ascending=False)
        prod_lbls = [PROD_LABELS[c] for c in prod_avgs.index]
        prod_clrs = [ACCENT if c == prod_avgs.index[0] else BORDER for c in prod_avgs.index]
        pb1, pb2  = st.columns(2)
        with pb1:
            fig_pb = go.Figure(go.Bar(x=prod_avgs.values, y=prod_lbls, orientation='h',
                                      marker=dict(color=prod_clrs, line=dict(width=0)),
                                      text=prod_avgs.values.round(1), texttemplate='$%{text:.0f}', textposition='outside',
                                      textfont=dict(color=MUTED, size=11)))
            apply_theme(fig_pb, "Avg Spend by Product")
            fig_pb.update_layout(height=320, yaxis=dict(autorange='reversed'))
            st.plotly_chart(fig_pb, use_container_width=True)
        with pb2:
            fig_pp = go.Figure(go.Pie(labels=prod_lbls, values=prod_avgs.values, hole=0.5,
                                      marker=dict(colors=COLORS), textfont=dict(color=TEXT2, size=11)))
            apply_theme(fig_pp, "Product Revenue Share")
            fig_pp.update_layout(height=320)
            st.plotly_chart(fig_pp, use_container_width=True)
        st.markdown(
            f'<div class="ms-rec-card secondary" style="margin-top:1rem;">'
            f'<div class="ms-rec-label">Hero Product</div>'
            f'<div class="ms-rec-value">{top_prod}</div>'
            f'<div class="ms-rec-score">Avg spend: ${top_val:,.0f}</div>'
            f'<ul class="ms-action-list">'
            f'<li>Feature in primary channel creative assets</li>'
            f'<li>Bundle with lower-traction SKUs</li>'
            f'<li>Anchor loyalty tier programme to this category</li>'
            f'</ul></div>', unsafe_allow_html=True,
        )

    with ptab2:
        ch_cols   = ['NumStorePurchases', 'NumWebPurchases', 'NumCatalogPurchases']
        ch_names  = ['In-Store', 'Web', 'Catalog']
        ch_totals = df_tgt[ch_cols].sum().values
        ch_avgs   = df_tgt[ch_cols].mean().values
        fig_ch = go.Figure(go.Bar(x=ch_names, y=ch_totals, marker=dict(color=COLORS[:3], line=dict(width=0)),
                                   text=ch_totals, textposition='outside', textfont=dict(color=MUTED, size=11)))
        apply_theme(fig_ch, "Total Purchases by Channel")
        fig_ch.update_layout(height=320)
        st.plotly_chart(fig_ch, use_container_width=True)
        st.dataframe(pd.DataFrame({'Channel': ch_names, 'Total': ch_totals,
                                    'Avg/Customer': ch_avgs.round(2), 'Share (%)': (ch_totals/ch_totals.sum()*100).round(1)}),
                     use_container_width=True, hide_index=True)

    with ptab3:
        dd1, dd2 = st.columns(2)
        for feat, ttl, col in [('Age','Age Distribution',dd1), ('Income','Income Distribution',dd2),
                                 ('Total_Spend','Spend Distribution',dd1), ('Tenure_Months','Tenure Distribution',dd2)]:
            with col:
                fig_h = go.Figure(go.Histogram(x=df_tgt[feat], nbinsx=24,
                                               marker=dict(color=ACCENT, line=dict(color=BG, width=0.5)), opacity=0.85))
                apply_theme(fig_h, ttl)
                fig_h.update_layout(height=240, showlegend=False)
                st.plotly_chart(fig_h, use_container_width=True)

    section("", "Segment Benchmarking")
    bench = []
    for ci in df_final['Cluster'].unique():
        cdf = df_final[df_final['Cluster'] == ci]
        bench.append({'Cluster': f'Cluster {ci}', 'Size': len(cdf),
                       'Avg Age': round(cdf['Age'].mean(), 1), 'Avg Spend ($)': round(cdf['Total_Spend'].mean(), 0),
                       'Avg Income ($)': round(cdf['Income'].mean(), 0), 'Share (%)': round(len(cdf)/len(df_final)*100, 1)})
    bench_df = pd.DataFrame(bench)

    def _hl(row):
        s = [''] * len(row)
        if row['Cluster'] == f'Cluster {cid}':
            s = [f'background-color:{ACCENT_DIM};color:{ACCENT};'] * len(row)
        return s

    st.dataframe(bench_df.style.apply(_hl, axis=1)
                 .format({'Avg Spend ($)': '${:,.0f}', 'Avg Income ($)': '${:,.0f}', 'Avg Age': '{:.1f}', 'Share (%)': '{:.1f}%'}),
                 use_container_width=True, hide_index=True)

    section("", "Business Insights & Recommendations")
    st.markdown(
        f'<div class="ms-rec-card primary">'
        f'<table class="ms-action-table">'
        f'<thead><tr><th>Horizon</th><th>Action</th><th>Owner</th><th>KPI</th></tr></thead>'
        f'<tbody>'
        f'<tr class="highlight-row"><td>Day 1–30</td><td>Launch primary campaign on <strong>{winner}</strong>. Hero SKU: {top_prod}.</td><td>Channel Lead</td><td>CTR, CVR</td></tr>'
        f'<tr><td>Day 1–30</td><td>Set up attribution pipeline and ROMI dashboard.</td><td>Analytics</td><td>Data completeness</td></tr>'
        f'<tr><td>Day 31–60</td><td>A/B test on {winner}. Activate {runner_up} as retargeting layer.</td><td>Growth</td><td>Lift vs control</td></tr>'
        f'<tr><td>Day 31–60</td><td>Bundle {top_prod} with complementary SKU. Pilot to 10% of segment.</td><td>Product</td><td>Bundle attach rate</td></tr>'
        f'<tr><td>Day 61–90</td><td>Scale winner. Launch loyalty programme for {profile_label} tier.</td><td>CRM</td><td>CLV, Churn</td></tr>'
        f'<tr><td>Day 61–90</td><td>Re-run segmentation. Validate cluster stability.</td><td>Data Science</td><td>Silhouette</td></tr>'
        f'</tbody></table></div>', unsafe_allow_html=True,
    )

# =============================================================================
# FOOTER
# =============================================================================
st.markdown(
    f'<div class="ms-footer">'
    f'<div class="ms-footer-left"><strong>MarketSync</strong> — Marketing Channel Optimization DSS — '
    f'Clarisya Adeline &middot; Nazwa Nashatasya &middot; Ammara Azwadiena Alfiantie<br>'
    f'UAS Decision Support System — Built with Streamlit, Scikit-learn &amp; Plotly</div>'
    f'<div class="ms-footer-right">PCA + K-MEANS + SAW</div>'
    f'</div>', unsafe_allow_html=True,
)