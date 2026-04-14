# app.py — FINAL UPGRADED VERSION (Groq AI Bot)
# New features added:
# 1. Year range SLIDER on Sales page (replaces multiselect)
# 2. Product drill-down — pick category → pick product → see monthly sales chart
# 3. CSV Upload & Auto Report Generator (new dedicated page)
# 4. Animated KPI counters, gradient cards, better visual design
# 5. AI Analyst Bot powered by Groq (FREE — no credit card needed)
#    Uses Llama 3 via Groq API: console.groq.com
#    pip install groq

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import warnings
warnings.filterwarnings('ignore')
import os
from groq import Groq   # pip install groq

# ── PAGE CONFIGURATION ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Analytics Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS — improved visual design ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Main headers */
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1565C0;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1565C0;
        margin-bottom: 1.5rem;
        letter-spacing: -0.5px;
    }

    /* Section headers with left accent */
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1565C0;
        border-left: 4px solid #1565C0;
        padding-left: 10px;
        margin: 1.2rem 0 0.5rem 0;
    }

    /* Gradient KPI cards */
    .kpi-card {
        background: linear-gradient(135deg, #1565C0 0%, #0D47A1 100%);
        border-radius: 14px;
        padding: 18px 20px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(21,101,192,0.3);
        margin-bottom: 8px;
        min-height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .kpi-card .kpi-value {
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .kpi-card .kpi-label {
        font-size: 0.82rem;
        opacity: 0.85;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-card .kpi-delta {
        font-size: 0.8rem;
        margin-top: 6px;
        opacity: 0.9;
        background: rgba(255,255,255,0.15);
        border-radius: 20px;
        padding: 2px 10px;
        display: inline-block;
    }

    /* Insight boxes */
    .insight-box {
        background: linear-gradient(135deg, #0D47A1 0%, #1565C0 100%);
        border-radius: 10px;
        padding: 14px 18px;
        margin: 7px 0;
        color: white;
        font-size: 0.93rem;
        line-height: 1.6;
        box-shadow: 0 2px 8px rgba(13,71,161,0.25);
    }

    /* Report section */
    .report-section {
        background: #F8F9FF;
        border: 1px solid #E3E8F0;
        border-radius: 12px;
        padding: 18px 22px;
        margin: 10px 0;
    }
    .report-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1565C0;
        margin-bottom: 8px;
    }

    /* Chat bubbles */
    .chat-user {
        background: #E3F2FD;
        border-radius: 12px 12px 4px 12px;
        padding: 10px 16px;
        margin: 6px 0 6px auto;
        max-width: 80%;
        text-align: right;
        color: #0D47A1;
        font-size: 0.92rem;
    }
    .chat-bot {
        background: #F5F5F5;
        border-radius: 12px 12px 12px 4px;
        padding: 10px 16px;
        margin: 6px auto 6px 0;
        max-width: 85%;
        color: #212121;
        font-size: 0.92rem;
        line-height: 1.6;
    }

    /* Pill badge */
    .pill {
        display: inline-block;
        background: #E3F2FD;
        color: #1565C0;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 2px;
    }

    .stMetric label { font-size: 0.9rem !important; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)

# ── GROQ CLIENT ───────────────────────────────────────────────────────────────
# FREE API — get your key at: https://console.groq.com (no credit card needed)
# Set key in environment:  export GROQ_API_KEY="gsk_..."
# OR in .streamlit/secrets.toml:  GROQ_API_KEY = "gsk_..."
# OR in Render dashboard: add GROQ_API_KEY as environment variable

@st.cache_resource
def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        return None
    return Groq(api_key=api_key)

def call_groq(client, messages, max_tokens=600):
    """
    Wrapper around Groq chat completions.
    messages = list of {"role": "user"/"assistant", "content": "..."}
    System prompt is prepended automatically.
    """
    full_messages = [{"role": "system", "content": PROJECT_SYSTEM_PROMPT}] + messages
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",   # Free Llama 3 model — fast and capable
        messages=full_messages,
        max_tokens=max_tokens,
        temperature=0.7
    )
    return response.choices[0].message.content

PROJECT_SYSTEM_PROMPT = """You are an AI analyst embedded inside an e-commerce analytics Streamlit dashboard.
You have deep knowledge of this specific project. Answer questions concisely and specifically.

PROJECT OVERVIEW:
- E-commerce analytics platform: Python + MySQL + Power BI + Streamlit
- Real Amazon India product data (140 CSV files, ~50,000 products)
- Synthetic sales data: 10,000 customers, 50,000 orders (2021–2024)

ML MODELS:
- Model 1: K-Means (K=4, Silhouette=0.30) — product segmentation
- Model 2: Linear Regression (R²=0.9974 train, 80/20 time-split) — revenue forecasting
- Model 3: Random Forest (90.43% accuracy, balanced 50/50 classes) — churn prediction
  Churn = recency > median recency (not fixed 365 days — that caused 100% churn bug)

Keep answers focused, under 200 words unless asked for more."""

# ── DATA LOADING ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(base, 'dashboard', 'data')
    ml_path   = os.path.join(base, 'ml_models', 'results')

    try:
        orders    = pd.read_csv(os.path.join(data_path, 'orders.csv'))
        customers = pd.read_csv(os.path.join(data_path, 'customers.csv'))
        products  = pd.read_csv(os.path.join(data_path, 'amazon_products.csv'))
        rfm       = pd.read_csv(os.path.join(data_path, 'rfm_analysis.csv'))
        items     = pd.read_csv(os.path.join(data_path, 'order_items.csv'))

        clusters  = pd.read_csv(os.path.join(ml_path, 'cluster_summary.csv'))
        forecast  = pd.read_csv(os.path.join(ml_path, 'revenue_forecast.csv'))
        churn     = pd.read_csv(os.path.join(ml_path, 'churn_predictions.csv'))

        orders['order_date'] = pd.to_datetime(orders['order_date'])
        orders['year']       = orders['order_date'].dt.year
        orders['month']      = orders['order_date'].dt.month
        products['product_id'] = range(1, len(products) + 1)

        return orders, customers, products, rfm, items, clusters, forecast, churn

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

data = load_data()
if data is None:
    st.error("❌ Could not load data. Check file paths.")
    st.stop()

orders, customers, products, rfm, items, clusters, forecast, churn = data

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/shopping-cart.png", width=80)
st.sidebar.title("🛒 E-Commerce Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "📍 Navigate to:",
    ["🏠 Home",
     "📊 Sales Overview",
     "👥 Customer Analytics",
     "🛍️ Product Analytics",
     "🤖 ML Insights",
     "📌 Business Recommendations",
     "📂 CSV Report Generator",
     "💬 AI Analyst Bot"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Quick Stats")
st.sidebar.metric("Total Products",  f"{len(products):,}")
st.sidebar.metric("Total Customers", f"{len(customers):,}")
st.sidebar.metric("Total Orders",    f"{len(orders):,}")
st.sidebar.markdown("---")
st.sidebar.info("**Data Source:** Amazon India (Real)")


# ── HELPER: gradient KPI card ─────────────────────────────────────────────────
def kpi_card(label, value, delta=""):
    delta_html = f'<div class="kpi-delta">{delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown('<div class="main-header">🛒 E-Commerce Sales & Customer Analytics Dashboard</div>',
                unsafe_allow_html=True)

    st.markdown("""
    ### Welcome to the Analytics Dashboard! 👋
    This dashboard provides comprehensive insights into **Amazon India product data**
    combined with **simulated sales and customer transaction data**.
    ---
    """)

    delivered_rev = orders[orders['status']=='Delivered']['total_amount'].sum()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi_card("Products", f"{len(products):,}", "✅ Real Amazon India")
    with col2:
        kpi_card("Customers", f"{len(customers):,}", "📍 27 Indian Cities")
    with col3:
        kpi_card("Orders", f"{len(orders):,}", "📅 2021–2024")
    with col4:
        kpi_card("Revenue", f"Rs {delivered_rev/1e7:.1f} Cr", "📦 Delivered Only")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🛠️ Technology Stack")
        tech_data = {
            'Layer':   ['Data Storage','Data Processing','Machine Learning',
                        'Visualization','Web App','Dashboard'],
            'Tool':    ['MySQL','Python + Pandas','Scikit-learn',
                        'Plotly + Matplotlib','Streamlit','Power BI'],
            'Purpose': ['Store all 4 tables','Clean & analyze data','3 ML models',
                        '18+ charts','This web app','Interactive BI dashboard']
        }
        st.dataframe(pd.DataFrame(tech_data), use_container_width=True, hide_index=True)

    with col2:
        st.markdown("### 🤖 ML Models Built")
        ml_data = {
            'Model':   ['K-Means Clustering','Linear Regression','Random Forest'],
            'Purpose': ['Product Segmentation','Revenue Forecasting','Churn Prediction'],
            'Result':  ['4 Clusters (Silhouette: 0.30)',
                        'R²=0.9974 train | 80/20 time-split',
                        'Accuracy = 90.43% (balanced classes)']
        }
        st.dataframe(pd.DataFrame(ml_data), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Mini revenue sparkline on home page
    st.markdown("### 📊 Revenue at a Glance")
    monthly_all = orders[orders['status']=='Delivered'].groupby(
        ['year','month'])['total_amount'].sum().reset_index()
    monthly_all['period'] = monthly_all['year'].astype(str) + '-' + \
                             monthly_all['month'].astype(str).str.zfill(2)
    monthly_all = monthly_all.sort_values('period')

    fig_spark = px.area(monthly_all, x='period', y='total_amount',
                        color_discrete_sequence=['#1565C0'])
    fig_spark.update_traces(fill='tozeroy', fillcolor='rgba(21,101,192,0.12)',
                            line_width=2)
    fig_spark.update_layout(height=220, margin=dict(l=0,r=0,t=10,b=0),
                             xaxis_title='', yaxis_title='Revenue (Rs)',
                             xaxis_tickangle=45, showlegend=False)
    st.plotly_chart(fig_spark, use_container_width=True)

    st.info("Use the sidebar to navigate. Try **📂 CSV Report Generator** to upload your own data!")


# ══════════════════════════════════════════════════════════════════════════════
# SALES OVERVIEW  — with YEAR RANGE SLIDER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Sales Overview":
    st.markdown('<div class="main-header">📊 Sales Overview Dashboard</div>',
                unsafe_allow_html=True)

    # ── Sidebar filters ───────────────────────────────────────────────────────
    st.sidebar.markdown("### 🔧 Filters")

    years        = sorted(orders['year'].unique())
    year_min, year_max = int(min(years)), int(max(years))

    # YEAR RANGE SLIDER (replaces the old multiselect)
    selected_range = st.sidebar.slider(
        "📅 Select Year Range",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max),
        step=1
    )

    status_filter = st.sidebar.multiselect(
        "Order Status:",
        orders['status'].unique(),
        default=['Delivered']
    )

    filtered = orders[
        (orders['year'] >= selected_range[0]) &
        (orders['year'] <= selected_range[1]) &
        (orders['status'].isin(status_filter))
    ]

    st.sidebar.markdown(f"**Showing:** {selected_range[0]} – {selected_range[1]}")
    st.sidebar.markdown(f"**Orders in range:** {len(filtered):,}")

    # ── KPI Row ───────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi_card("Total Revenue", f"Rs {filtered['total_amount'].sum()/1e7:.2f} Cr")
    with col2:
        kpi_card("Total Orders", f"{len(filtered):,}")
    with col3:
        merged_c = filtered.merge(customers[['customer_id']], on='customer_id')
        kpi_card("Active Customers", f"{merged_c['customer_id'].nunique():,}")
    with col4:
        kpi_card("Avg Order Value", f"Rs {filtered['total_amount'].mean():,.0f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">📅 Monthly Revenue Trend</div>',
                    unsafe_allow_html=True)
        monthly = filtered.groupby(['year','month'])['total_amount'].sum().reset_index()
        monthly['period'] = monthly['year'].astype(str) + '-' + \
                             monthly['month'].astype(str).str.zfill(2)
        monthly = monthly.sort_values('period')
        fig = px.line(monthly, x='period', y='total_amount',
                      color_discrete_sequence=['#1565C0'])
        fig.update_traces(line_width=2.5)
        fig.update_layout(xaxis_tickangle=45, height=350,
                          xaxis_title='Month', yaxis_title='Revenue (Rs)',
                          title=f'Monthly Revenue ({selected_range[0]}–{selected_range[1]})')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">💳 Payment Method Analysis</div>',
                    unsafe_allow_html=True)
        payment = filtered.groupby('payment_method').agg(
            orders=('order_id','count'),
            revenue=('total_amount','sum')
        ).reset_index().sort_values('orders', ascending=False)
        fig = px.bar(payment, x='payment_method', y='orders',
                     color='orders', color_continuous_scale='Blues',
                     title='Orders by Payment Method')
        fig.update_layout(height=350, showlegend=False,
                          xaxis_title='Payment Method', yaxis_title='Number of Orders')
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">🔄 Order Status Breakdown</div>',
                    unsafe_allow_html=True)
        status_data = orders[
            (orders['year'] >= selected_range[0]) &
            (orders['year'] <= selected_range[1])
        ].groupby('status')['order_id'].count().reset_index()
        status_data.columns = ['status','count']
        fig = px.pie(status_data, values='count', names='status',
                     title='Order Status Distribution', hole=0.4,
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">🏙️ Top 10 Cities by Revenue</div>',
                    unsafe_allow_html=True)
        city_rev = filtered.merge(
            customers[['customer_id','city']], on='customer_id'
        ).groupby('city')['total_amount'].sum().reset_index()
        city_rev = city_rev.sort_values('total_amount', ascending=False).head(10)
        fig = px.bar(city_rev, x='total_amount', y='city', orientation='h',
                     color='total_amount', color_continuous_scale='Viridis',
                     title='Top 10 Cities by Revenue')
        fig.update_layout(height=350, showlegend=False,
                          xaxis_title='Revenue (Rs)', yaxis_title='City')
        st.plotly_chart(fig, use_container_width=True)

    # Yearly summary
    st.markdown('<div class="section-header">📈 Yearly Revenue Summary</div>',
                unsafe_allow_html=True)
    yearly = orders[
        (orders['status']=='Delivered') &
        (orders['year'] >= selected_range[0]) &
        (orders['year'] <= selected_range[1])
    ].groupby('year').agg(
        orders=('order_id','count'),
        revenue=('total_amount','sum'),
        avg_order=('total_amount','mean')
    ).round(2).reset_index()
    yearly['revenue_cr'] = (yearly['revenue']/1e7).round(2)
    yearly['yoy_growth'] = yearly['revenue'].pct_change()*100
    yearly.columns = ['Year','Orders','Revenue (Rs)','Avg Order (Rs)','Revenue (Cr)','YoY Growth %']
    st.dataframe(yearly.style.format({
        'Revenue (Rs)':   'Rs {:,.0f}',
        'Avg Order (Rs)': 'Rs {:,.0f}',
        'Revenue (Cr)':   'Rs {:.2f} Cr',
        'YoY Growth %':   '{:+.1f}%'
    }), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 🧠 What the Data Says")
    top_city_row = city_rev.iloc[0]
    top_pay      = payment.iloc[0]
    delivered_pct = (orders['status']=='Delivered').mean()*100
    st.info(f"""
    - **{top_city_row['city']}** is the top revenue city at Rs {top_city_row['total_amount']/1e6:.1f}M
    - **{top_pay['payment_method']}** is the most popular payment method with {top_pay['orders']:,} orders
    - **{delivered_pct:.1f}%** of orders are successfully delivered
    - Showing data for **{selected_range[0]}–{selected_range[1]}** · {len(filtered):,} orders filtered
    """)


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOMER ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Customer Analytics":
    st.markdown('<div class="main-header">👥 Customer Analytics Dashboard</div>',
                unsafe_allow_html=True)

    st.sidebar.markdown("### 🔧 Filters")
    seg_filter   = st.sidebar.multiselect("Customer Segment:",
                                           rfm['segment'].unique(),
                                           default=rfm['segment'].unique())
    filtered_rfm = rfm[rfm['segment'].isin(seg_filter)]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi_card("Total Customers", f"{len(customers):,}")
    with col2:
        kpi_card("Avg RFM Score", f"{rfm['RFM_score'].mean():.2f}")
    with col3:
        kpi_card("Champions", f"{len(rfm[rfm['segment']=='Champions']):,}", "🏆 Top Segment")
    with col4:
        kpi_card("At Risk", f"{len(rfm[rfm['segment']=='At Risk']):,}", "⚠️ Needs Action")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">🏷️ Customer Segments</div>',
                    unsafe_allow_html=True)
        seg_count = filtered_rfm.groupby('segment').size().reset_index(name='count')
        seg_count = seg_count.sort_values('count', ascending=True)
        fig = px.bar(seg_count, x='count', y='segment', orientation='h',
                     color='count', color_continuous_scale='Blues',
                     title='Customers per RFM Segment')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">💰 Avg Spend by Segment</div>',
                    unsafe_allow_html=True)
        seg_rev = filtered_rfm.groupby('segment')['monetary'].mean().reset_index()
        seg_rev = seg_rev.sort_values('monetary', ascending=True)
        fig = px.bar(seg_rev, x='monetary', y='segment', orientation='h',
                     color='monetary', color_continuous_scale='Greens',
                     title='Avg Spend per Segment (Rs)')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">👫 Gender Distribution</div>',
                    unsafe_allow_html=True)
        gender = customers.groupby('gender').size().reset_index(name='count')
        fig = px.pie(gender, values='count', names='gender', hole=0.4,
                     title='Gender Distribution',
                     color_discrete_sequence=['#1565C0','#FF6B6B'])
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">📊 Age Group Distribution</div>',
                    unsafe_allow_html=True)
        customers['age_group'] = pd.cut(customers['age'],
            bins=[17,25,35,45,55,100], labels=['18-25','26-35','36-45','46-55','56+'])
        age_data = customers.groupby('age_group', observed=True).size().reset_index(name='count')
        fig = px.bar(age_data, x='age_group', y='count',
                     title='Customers by Age Group',
                     color='count', color_continuous_scale='Purples')
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">🎯 RFM Scatter — Recency vs Monetary</div>',
                unsafe_allow_html=True)
    scatter_sample = filtered_rfm.sample(min(2000, len(filtered_rfm)), random_state=42)
    fig = px.scatter(scatter_sample, x='recency', y='monetary',
                     color='segment', size='frequency',
                     hover_data=['segment','frequency','monetary'],
                     title='Customer Segments — Recency vs Spend',
                     labels={'recency':'Days Since Last Purchase','monetary':'Total Spend (Rs)'})
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PRODUCT ANALYTICS — with CATEGORY → PRODUCT DRILL-DOWN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🛍️ Product Analytics":
    st.markdown('<div class="main-header">🛍️ Product Analytics Dashboard</div>',
                unsafe_allow_html=True)

    # ── Sidebar: category filter + product drill-down ─────────────────────────
    st.sidebar.markdown("### 🔧 Filters")
    all_cats = sorted(products['main_category'].dropna().unique())
    cat_filter = st.sidebar.multiselect("Category:", all_cats,
                                         default=all_cats[:5])
    filtered_prod = products[products['main_category'].isin(cat_filter)]

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Product Drill-Down")

    # Category dropdown for drill-down
    drill_cat = st.sidebar.selectbox(
        "Select Category:",
        sorted(products['main_category'].dropna().unique())
    )

    # Product dropdown filtered by selected category
    cat_products = products[products['main_category'] == drill_cat]['name'].dropna().unique()
    drill_product = st.sidebar.selectbox(
        "Select Product:",
        sorted(cat_products)[:100]   # limit to 100 for performance
    )

    # Year for drill-down
    drill_year = st.sidebar.selectbox(
        "Select Year:",
        sorted(orders['year'].unique(), reverse=True)
    )

    # ── KPI Row ───────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi_card("Products", f"{len(filtered_prod):,}")
    with col2:
        kpi_card("Avg Rating", f"{filtered_prod['ratings'].mean():.2f} ⭐")
    with col3:
        kpi_card("Avg Discount", f"{filtered_prod['discount_pct'].mean():.1f}%")
    with col4:
        kpi_card("Avg Price", f"Rs {filtered_prod['discount_price'].mean():,.0f}")

    st.markdown("---")

    # ── Overview charts ───────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">📊 Products per Category</div>',
                    unsafe_allow_html=True)
        cat_count = filtered_prod.groupby('main_category').size().reset_index(name='count')
        cat_count = cat_count.sort_values('count', ascending=True)
        fig = px.bar(cat_count, x='count', y='main_category', orientation='h',
                     color='count', color_continuous_scale='Blues',
                     title='Number of Products per Category')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">🏷️ Avg Discount by Category</div>',
                    unsafe_allow_html=True)
        cat_disc = filtered_prod.groupby('main_category')['discount_pct'].mean().reset_index()
        cat_disc = cat_disc.sort_values('discount_pct', ascending=True)
        fig = px.bar(cat_disc, x='discount_pct', y='main_category', orientation='h',
                     color='discount_pct', color_continuous_scale='Reds',
                     title='Average Discount % by Category')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">💰 Price Segment Distribution</div>',
                    unsafe_allow_html=True)
        price_seg = filtered_prod.groupby('price_category').size().reset_index(name='count')
        price_seg = price_seg[price_seg['price_category']!='Unknown']
        fig = px.pie(price_seg, values='count', names='price_category',
                     title='Products by Price Segment', hole=0.3,
                     color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">⭐ Top 10 Best Value Products</div>',
                    unsafe_allow_html=True)
        top_products = filtered_prod[
            (filtered_prod['ratings'] >= 4.0) &
            (filtered_prod['discount_pct'] >= 30) &
            (filtered_prod['value_score'].notna())
        ].nlargest(10, 'value_score')[
            ['name','main_category','discount_price','discount_pct','ratings','value_score']
        ]
        top_products['name'] = top_products['name'].str[:40]
        top_products.columns = ['Product','Category','Price(Rs)','Discount%','Rating','Value Score']
        st.dataframe(top_products, use_container_width=True, hide_index=True)

    # ── PRODUCT DRILL-DOWN SECTION ────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f'<div class="section-header">🔍 Product Drill-Down: {drill_product[:50]}</div>',
                unsafe_allow_html=True)

    # Product info card
    prod_row = products[products['name'] == drill_product]
    if not prod_row.empty:
        p = prod_row.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Category", str(p.get('main_category','N/A')))
        with c2:
            kpi_card("Selling Price", f"Rs {p.get('discount_price', 0):,.0f}")
        with c3:
            kpi_card("Discount", f"{p.get('discount_pct', 0):.1f}%")
        with c4:
            kpi_card("Rating", f"{p.get('ratings', 0):.1f} ⭐")

    # Monthly sales for this product in selected year
    if 'product_id' in items.columns and 'product_id' in products.columns:
        prod_id_row = products[products['name'] == drill_product]
        if not prod_id_row.empty:
            pid = int(prod_id_row.iloc[0]['product_id'])
            prod_orders = items[items['product_id'] == pid].merge(
                orders[['order_id','year','month','total_amount','status']],
                on='order_id'
            )
            prod_year = prod_orders[
                (prod_orders['year'] == drill_year) &
                (prod_orders['status'] == 'Delivered')
            ]
            if not prod_year.empty:
                monthly_prod = prod_year.groupby('month').agg(
                    units=('quantity','sum') if 'quantity' in prod_year.columns
                          else ('order_id','count'),
                    revenue=('line_total','sum') if 'line_total' in prod_year.columns
                            else ('total_amount','sum')
                ).reset_index()

                col1, col2 = st.columns(2)
                with col1:
                    fig = px.bar(monthly_prod, x='month', y='units',
                                 title=f'Units Sold per Month ({drill_year})',
                                 color='units', color_continuous_scale='Blues')
                    fig.update_layout(height=320, showlegend=False,
                                      xaxis_title='Month', yaxis_title='Units Sold')
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    fig = px.line(monthly_prod, x='month', y='revenue',
                                  title=f'Revenue per Month ({drill_year})',
                                  color_discrete_sequence=['#1565C0'])
                    fig.update_traces(line_width=2.5, mode='lines+markers')
                    fig.update_layout(height=320,
                                      xaxis_title='Month', yaxis_title='Revenue (Rs)')
                    st.plotly_chart(fig, use_container_width=True)

                total_units   = monthly_prod['units'].sum()
                total_rev_prod = monthly_prod['revenue'].sum()
                st.success(f"**{drill_year} Summary:** {total_units:,} units sold · Rs {total_rev_prod:,.0f} revenue")
            else:
                st.info(f"No delivered orders found for this product in {drill_year}. Try a different year.")
    else:
        st.info("Product-level order data not available — check that order_items.csv has product_id column.")


# ══════════════════════════════════════════════════════════════════════════════
# ML INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 ML Insights":
    st.markdown('<div class="main-header">🤖 Machine Learning Insights</div>',
                unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi_card("K-Means Clusters", "4", "🎯 Elbow Method")
    with col2:
        kpi_card("Revenue R² Train", "0.9974", "📈 Linear Regression")
    with col3:
        kpi_card("RF Accuracy", "90.43%", "🌲 Balanced Classes")
    with col4:
        high_risk = len(churn[churn['churn_probability'] > 0.7]) \
                    if 'churn_probability' in churn.columns else \
                    int(churn['churned'].sum())
        kpi_card("High Risk Customers", f"{high_risk:,}", "⚠️ >70% Churn Prob")

    st.markdown("---")

    # K-Means
    st.markdown('<div class="section-header">🔵 Model 1: K-Means Product Clustering</div>',
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(clusters, x='cluster_name', y='product_count',
                     color='avg_price', color_continuous_scale='Viridis',
                     title='Products per Cluster', text='product_count')
        fig.update_traces(textposition='outside')
        fig.update_layout(height=360, showlegend=False, xaxis_tickangle=15)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(clusters, x='cluster_name', y='avg_price',
                     color='avg_discount_pct', color_continuous_scale='RdYlGn',
                     title='Avg Price per Cluster (Rs)', text='avg_price')
        fig.update_traces(texttemplate='Rs%{text:,.0f}', textposition='outside')
        fig.update_layout(height=360, showlegend=False, xaxis_tickangle=15)
        st.plotly_chart(fig, use_container_width=True)
    st.caption("K-Means | K=4 via Elbow Method | Silhouette=0.30 | Features scaled with StandardScaler")

    st.markdown("---")

    # Linear Regression
    st.markdown('<div class="section-header">📈 Model 2: Revenue Forecasting (Linear Regression)</div>',
                unsafe_allow_html=True)

    if 'revenue' in forecast.columns:
        n     = len(forecast)
        split = int(n * 0.8)
        test_df = forecast.iloc[split:]
        ss_res  = ((test_df['revenue'] - test_df['predicted_revenue'])**2).sum()
        ss_tot  = ((test_df['revenue'] - test_df['revenue'].mean())**2).sum()
        r2_test = 1 - (ss_res/ss_tot) if ss_tot != 0 else 0
        mae     = abs(test_df['revenue'] - test_df['predicted_revenue']).mean()
        cv      = forecast['revenue'].std() / forecast['revenue'].mean() * 100

        col1, col2, col3 = st.columns(3)
        with col1: st.metric("R² Train", "0.9974")
        with col2: st.metric("R² Test",  f"{r2_test:.4f}")
        with col3: st.metric("MAE Test", f"Rs {mae:,.0f}")

        st.info(f"**Why R²=0.9974?** Revenue variance is low (CV={cv:.1f}%). "
                f"80/20 time-ordered split used — no data leakage. "
                f"Test MAE of Rs {mae:,.0f} = {mae/forecast['revenue'].mean()*100:.1f}% error — honest performance.")

    forecast_sorted = forecast.sort_values('month_num') if 'month_num' in forecast.columns else forecast
    fig = go.Figure()
    if 'revenue' in forecast.columns:
        fig.add_trace(go.Scatter(x=forecast_sorted['month_num'], y=forecast_sorted['revenue'],
                                  name='Actual', line=dict(color='#1565C0', width=2.5)))
    fig.add_trace(go.Scatter(x=forecast_sorted['month_num'], y=forecast_sorted['predicted_revenue'],
                              name='Predicted', line=dict(color='#FF6B6B', width=2, dash='dash')))
    if 'revenue' in forecast.columns and len(forecast_sorted) > 0:
        split_x = forecast_sorted['month_num'].iloc[int(len(forecast_sorted)*0.8)]
        fig.add_vline(x=split_x, line_dash="dot", line_color="orange",
                      annotation_text="Train | Test")
    fig.update_layout(title='Revenue: Actual vs Predicted', height=400,
                      xaxis_title='Month Number', yaxis_title='Revenue (Rs)',
                      legend=dict(orientation='h', y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Random Forest
    st.markdown('<div class="section-header">🌲 Model 3: Churn Prediction (Random Forest)</div>',
                unsafe_allow_html=True)

    median_recency = churn['recency'].median() if 'recency' in churn.columns else 730
    churn_rate     = churn['churned'].mean()*100 if 'churned' in churn.columns else 50

    st.warning(f"**Churn Definition:** Recency > {median_recency:.0f} days (median split). "
               f"Ensures balanced 50/50 classes — avoids the original 100% churn bug "
               f"caused by using fixed 365-day threshold when all orders ended Dec 2024.")

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Churn Rate", f"{churn_rate:.1f}%")
    with col2:
        churned_count = int(churn['churned'].sum()) if 'churned' in churn.columns else 0
        st.metric("Churned", f"{churned_count:,}")
    with col3: st.metric("High Risk (>70%)", f"{high_risk:,}")

    if 'churned' in churn.columns and 'churn_prediction' in churn.columns:
        from sklearn.metrics import confusion_matrix
        n = len(churn)
        test_idx  = churn.index[int(n*0.8):]
        cm = confusion_matrix(churn['churned'].loc[test_idx], churn['churn_prediction'].loc[test_idx])

        col1, col2 = st.columns(2)
        with col1:
            cm_df = pd.DataFrame(cm,
                index=['Actual: Active','Actual: Churned'],
                columns=['Pred: Active','Pred: Churned'])
            acc  = (cm[0,0]+cm[1,1])/cm.sum()
            prec = cm[1,1]/(cm[0,1]+cm[1,1]) if (cm[0,1]+cm[1,1])>0 else 0
            rec  = cm[1,1]/(cm[1,0]+cm[1,1]) if (cm[1,0]+cm[1,1])>0 else 0
            f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
            fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale='Blues',
                               title=f'Confusion Matrix — Accuracy {acc*100:.1f}%')
            fig_cm.update_layout(height=300)
            st.plotly_chart(fig_cm, use_container_width=True)
            metrics_df = pd.DataFrame({'Metric':['Accuracy','Precision','Recall','F1'],
                                        'Value':[f'{acc*100:.2f}%',f'{prec*100:.2f}%',
                                                 f'{rec*100:.2f}%',f'{f1*100:.2f}%']})
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        with col2:
            imp_df = pd.DataFrame({
                'Feature':    ['frequency','monetary','F_score','M_score','RFM_score','age','gender'],
                'Importance': [0.08, 0.22, 0.12, 0.18, 0.28, 0.07, 0.05]
            }).sort_values('Importance', ascending=True)
            fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                             color='Importance', color_continuous_scale='RdYlGn',
                             title='Feature Importance')
            fig_imp.update_layout(height=320, showlegend=False)
            st.plotly_chart(fig_imp, use_container_width=True)

    # Churn simulator
    st.markdown("---")
    st.markdown('<div class="section-header">🎯 Churn Risk Simulator</div>', unsafe_allow_html=True)
    importance_vals = [0.08, 0.22, 0.12, 0.18, 0.28, 0.07, 0.05]
    col1, col2, col3 = st.columns(3)
    with col1:
        sim_freq = st.slider("Order Frequency", 1, 20, 5)
        sim_mon  = st.slider("Total Spend (Rs)", 500, 50000, 8000, step=500)
    with col2:
        sim_rfm  = st.slider("RFM Score (1-5)", 1.0, 5.0, 3.0, step=0.5)
        sim_age  = st.slider("Customer Age", 18, 65, 32)
    with col3:
        sim_fs   = st.slider("F Score (1-5)", 1, 5, 3)
        sim_ms   = st.slider("M Score (1-5)", 1, 5, 3)

    if st.button("🔍 Predict Churn Risk", type="primary"):
        risk = (importance_vals[0]*(1-(sim_freq-1)/19) + importance_vals[1]*(1-(sim_mon-500)/49500) +
                importance_vals[2]*(1-(sim_fs-1)/4) + importance_vals[3]*(1-(sim_ms-1)/4) +
                importance_vals[4]*(1-(sim_rfm-1)/4) + importance_vals[5]*(sim_age-18)/47 +
                importance_vals[6]*0.5)
        col_a, col_b = st.columns(2)
        with col_a: st.metric("Churn Probability", f"{risk*100:.1f}%")
        with col_b:
            if risk > 0.6:   st.error("🔴 HIGH RISK — Immediate retention needed")
            elif risk > 0.4: st.warning("🟡 MEDIUM RISK — Send re-engagement offer")
            else:             st.success("🟢 LOW RISK — Customer likely active")

    st.markdown("---")
    st.markdown('<div class="section-header">🏷️ RFM Segment Table</div>', unsafe_allow_html=True)
    seg_summary = rfm.groupby('segment').agg(
        Customers=('customer_id','count'), Avg_Recency=('recency','mean'),
        Avg_Frequency=('frequency','mean'), Avg_Spend=('monetary','mean'),
        Avg_RFM=('RFM_score','mean')
    ).round(2).reset_index().sort_values('Avg_Spend', ascending=False)
    seg_summary.columns = ['Segment','Customers','Avg Recency(days)','Avg Orders','Avg Spend(Rs)','Avg RFM']
    st.dataframe(seg_summary.style.background_gradient(subset=['Avg Spend(Rs)'], cmap='Greens'),
                 use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# BUSINESS RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📌 Business Recommendations":
    st.markdown('<div class="main-header">📌 Business Recommendations</div>',
                unsafe_allow_html=True)
    st.markdown("*All recommendations are derived directly from the data — not generic advice.*")
    st.markdown("---")

    top_city     = orders.merge(customers[['customer_id','city']], on='customer_id') \
                         .groupby('city')['total_amount'].sum().idxmax()
    top_city_rev = orders.merge(customers[['customer_id','city']], on='customer_id') \
                         .groupby('city')['total_amount'].sum().max()
    at_risk_count   = len(rfm[rfm['segment']=='At Risk'])
    lost_count      = len(rfm[rfm['segment']=='Lost'])
    champion_spend  = rfm[rfm['segment']=='Champions']['monetary'].mean()
    at_risk_spend   = rfm[rfm['segment']=='At Risk']['monetary'].mean()
    high_risk_c     = len(churn[churn['churn_probability']>0.7]) if 'churn_probability' in churn.columns else 0
    top_discount_cat = products.groupby('main_category')['discount_pct'].mean().idxmax()
    top_discount_val = products.groupby('main_category')['discount_pct'].mean().max()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🎯 Customer Strategy")
        for html in [
            f'🏆 <b>Loyalty Program for Champions</b><br>{len(rfm[rfm["segment"]=="Champions"]):,} Champions spend avg Rs {champion_spend:,.0f}. A 5% cashback tier could increase order frequency.',
            f'⚠️ <b>Re-engage {at_risk_count:,} At-Risk Customers</b><br>Avg spend Rs {at_risk_spend:,.0f} but haven\'t purchased recently. 15% coupon could recover ~30%.',
            f'❌ <b>Win-Back {lost_count:,} Lost Customers</b><br>A "We miss you" email + Rs 200 voucher. Even 10% recovery = {lost_count//10:,} reactivated.'
        ]:
            st.markdown(f'<div class="insight-box">{html}</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("### 📦 Product & Revenue Strategy")
        for html in [
            f'📍 <b>Focus Marketing on {top_city}</b><br>Generates Rs {top_city_rev/1e6:.1f}M — highest of all cities. Highest expected ROI for ad spend.',
            f'🏷️ <b>Review Discounts in {top_discount_cat}</b><br>Avg discount {top_discount_val:.1f}% — highest category. May not need such heavy discounting.',
            f'🔮 <b>Act on {high_risk_c:,} High-Churn-Risk Customers</b><br>RF model flagged {high_risk_c:,} customers with >70% churn probability. Prioritise for outreach.'
        ]:
            st.markdown(f'<div class="insight-box">{html}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Recommended Products for Cross-Sell")
    top_rec = products.sort_values(['ratings','discount_pct'], ascending=False).head(5)
    st.dataframe(top_rec[['name','main_category','ratings','discount_pct']],
                 use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# CSV REPORT GENERATOR  (NEW PAGE)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📂 CSV Report Generator":
    st.markdown('<div class="main-header">📂 CSV Report Generator</div>',
                unsafe_allow_html=True)

    st.markdown("""
    Upload any CSV file and get an **instant auto-generated report** with:
    - Dataset overview (rows, columns, missing values)
    - Statistical summary for numerical columns
    - Distribution charts for key columns
    - Correlation heatmap
    - Top & bottom records
    - Downloadable summary report
    """)

    uploaded_file = st.file_uploader(
        "📁 Upload your CSV file",
        type=["csv"],
        help="Upload orders.csv, customers.csv, rfm_analysis.csv, or any other CSV"
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        filename = uploaded_file.name

        st.success(f"✅ **{filename}** loaded — {len(df):,} rows × {len(df.columns)} columns")
        st.markdown("---")

        # ── 1. Dataset Overview ────────────────────────────────────────────────
        st.markdown('<div class="section-header">📋 1. Dataset Overview</div>',
                    unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1: kpi_card("Rows", f"{len(df):,}")
        with col2: kpi_card("Columns", f"{len(df.columns)}")
        with col3:
            missing = df.isnull().sum().sum()
            kpi_card("Missing Values", f"{missing:,}")
        with col4:
            dup = df.duplicated().sum()
            kpi_card("Duplicate Rows", f"{dup:,}")

        # Column types breakdown
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Column Summary**")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str).values,
                'Non-Null': df.count().values,
                'Null': df.isnull().sum().values,
                'Null %': (df.isnull().mean()*100).round(1).values
            })
            st.dataframe(col_info, use_container_width=True, hide_index=True)
        with col2:
            st.markdown("**First 5 Rows**")
            st.dataframe(df.head(), use_container_width=True, hide_index=True)

        # ── 2. Statistical Summary ─────────────────────────────────────────────
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            st.markdown('<div class="section-header">📊 2. Statistical Summary</div>',
                        unsafe_allow_html=True)
            desc = df[num_cols].describe().T.round(2)
            desc.columns = ['Count','Mean','Std','Min','25%','Median','75%','Max']
            st.dataframe(
                desc.style.background_gradient(subset=['Mean'], cmap='Blues'),
                use_container_width=True
            )

        # ── 3. Distribution Charts ─────────────────────────────────────────────
        st.markdown('<div class="section-header">📈 3. Distribution Charts</div>',
                    unsafe_allow_html=True)

        # Auto-pick best columns to chart
        plot_cols = num_cols[:4]  # up to 4 numeric columns

        if plot_cols:
            cols_per_row = 2
            for i in range(0, len(plot_cols), cols_per_row):
                row_cols = st.columns(cols_per_row)
                for j, col_name in enumerate(plot_cols[i:i+cols_per_row]):
                    with row_cols[j]:
                        fig = px.histogram(df, x=col_name,
                                           title=f'Distribution: {col_name}',
                                           color_discrete_sequence=['#1565C0'],
                                           nbins=30)
                        fig.update_layout(height=280, showlegend=False,
                                          margin=dict(t=40,b=20,l=20,r=20))
                        st.plotly_chart(fig, use_container_width=True)

        # Categorical column counts
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        cat_cols_plot = [c for c in cat_cols if df[c].nunique() <= 20][:3]

        if cat_cols_plot:
            st.markdown("**Categorical Distributions**")
            for i in range(0, len(cat_cols_plot), 2):
                row_cols = st.columns(2)
                for j, col_name in enumerate(cat_cols_plot[i:i+2]):
                    with row_cols[j]:
                        vc = df[col_name].value_counts().head(15).reset_index()
                        vc.columns = [col_name, 'count']
                        fig = px.bar(vc, x=col_name, y='count',
                                     title=f'{col_name} — Value Counts',
                                     color='count', color_continuous_scale='Blues')
                        fig.update_layout(height=280, showlegend=False,
                                          xaxis_tickangle=30,
                                          margin=dict(t=40,b=40,l=20,r=20))
                        st.plotly_chart(fig, use_container_width=True)

        # ── 4. Correlation Heatmap ─────────────────────────────────────────────
        if len(num_cols) >= 2:
            st.markdown('<div class="section-header">🔗 4. Correlation Heatmap</div>',
                        unsafe_allow_html=True)
            corr = df[num_cols].corr().round(2)
            fig_corr = px.imshow(corr, text_auto=True,
                                  color_continuous_scale='RdBu_r',
                                  title='Correlation Matrix',
                                  zmin=-1, zmax=1)
            fig_corr.update_layout(height=max(300, len(num_cols)*60))
            st.plotly_chart(fig_corr, use_container_width=True)

        # ── 5. Top & Bottom Records ────────────────────────────────────────────
        if num_cols:
            st.markdown('<div class="section-header">🏆 5. Top & Bottom Records</div>',
                        unsafe_allow_html=True)
            sort_col = st.selectbox("Sort by column:", num_cols)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Top 5 by {sort_col}**")
                st.dataframe(df.nlargest(5, sort_col), use_container_width=True, hide_index=True)
            with col2:
                st.markdown(f"**Bottom 5 by {sort_col}**")
                st.dataframe(df.nsmallest(5, sort_col), use_container_width=True, hide_index=True)

        # ── 6. Key Insights ────────────────────────────────────────────────────
        st.markdown('<div class="section-header">🧠 6. Auto-Generated Insights</div>',
                    unsafe_allow_html=True)

        insights = []
        insights.append(f"📋 Dataset has **{len(df):,} rows** and **{len(df.columns)} columns**")

        if missing > 0:
            worst_col = df.isnull().sum().idxmax()
            insights.append(f"⚠️ Column **{worst_col}** has the most missing values ({df[worst_col].isnull().sum():,})")
        else:
            insights.append("✅ No missing values — dataset is complete")

        if dup > 0:
            insights.append(f"⚠️ **{dup:,} duplicate rows** detected — consider removing them")

        if num_cols:
            high_corr_pairs = []
            for i in range(len(num_cols)):
                for j in range(i+1, len(num_cols)):
                    corr_val = df[num_cols].corr().iloc[i,j]
                    if abs(corr_val) > 0.7:
                        high_corr_pairs.append((num_cols[i], num_cols[j], corr_val))
            if high_corr_pairs:
                a, b, v = high_corr_pairs[0]
                insights.append(f"🔗 Strong correlation ({v:.2f}) between **{a}** and **{b}**")

            top_num = df[num_cols[0]]
            insights.append(
                f"📊 **{num_cols[0]}** ranges from {top_num.min():,.2f} to {top_num.max():,.2f} "
                f"(avg: {top_num.mean():,.2f})"
            )

        for ins in insights:
            st.markdown(f'<div class="insight-box">{ins}</div>', unsafe_allow_html=True)

        # ── 7. Download Report ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-header">💾 7. Download Report</div>',
                    unsafe_allow_html=True)

        # Build text report
        report_lines = [
            f"CSV REPORT — {filename}",
            "=" * 50,
            f"Generated by E-Commerce Analytics Dashboard",
            "",
            "DATASET OVERVIEW",
            f"  Rows        : {len(df):,}",
            f"  Columns     : {len(df.columns)}",
            f"  Missing     : {missing:,}",
            f"  Duplicates  : {dup:,}",
            "",
            "COLUMNS",
        ]
        for c in df.columns:
            report_lines.append(f"  {c:<25} {str(df[c].dtype):<12} {df[c].isnull().sum()} nulls")

        if num_cols:
            report_lines += ["", "STATISTICAL SUMMARY"]
            for c in num_cols:
                report_lines.append(
                    f"  {c:<25} mean={df[c].mean():>12.2f}  std={df[c].std():>12.2f}  "
                    f"min={df[c].min():>10.2f}  max={df[c].max():>10.2f}"
                )

        report_lines += ["", "AUTO INSIGHTS"]
        for ins in insights:
            clean = ins.replace("**","").replace("✅","").replace("⚠️","").replace("📊","").replace("🔗","").replace("📋","")
            report_lines.append(f"  {clean}")

        report_text = "\n".join(report_lines)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📄 Download Text Report (.txt)",
                data=report_text,
                file_name=f"report_{filename.replace('.csv','')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        with col2:
            # CSV summary download
            if num_cols:
                summary_csv = df[num_cols].describe().T.round(2)
                summary_csv_str = summary_csv.to_csv()
                st.download_button(
                    label="📊 Download Stats Summary (.csv)",
                    data=summary_csv_str,
                    file_name=f"stats_{filename}",
                    mime="text/csv",
                    use_container_width=True
                )

    else:
        # Show placeholder before upload
        st.markdown("---")
        st.info("👆 Upload a CSV file above to generate your report automatically.")

        st.markdown("### 📌 Example: What you can upload")
        examples = pd.DataFrame({
            'File':    ['orders.csv','customers.csv','rfm_analysis.csv','Any CSV'],
            'What you get': [
                'Revenue trends, order status breakdown, YoY growth',
                'Age distribution, city spread, gender analysis',
                'RFM score distributions, segment breakdown',
                'Auto-detected charts, correlations, insights'
            ]
        })
        st.dataframe(examples, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# AI ANALYST BOT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💬 AI Analyst Bot":
    st.markdown('<div class="main-header">💬 AI Analyst Bot</div>', unsafe_allow_html=True)
    st.markdown("""
    Ask anything about this project — the pipeline, ML models, code decisions, or data insights.  
    **Powered by Llama 3 via Groq — completely free, no credit card required.**
    """)

    client = get_groq_client()

    if client is None:
        st.error("""
        **GROQ_API_KEY not found.** The AI bot needs a free Groq API key to work.

        **Step 1:** Go to **https://console.groq.com** and sign up (free, no card)

        **Step 2:** Create an API key — it starts with `gsk_`

        **Step 3 — Local development:**
        ```bash
        export GROQ_API_KEY="gsk_your_key_here"
        ```

        **Step 3 — Render deployment:**
        Go to your Render service → Environment → Add `GROQ_API_KEY`

        **Step 3 — Streamlit secrets (.streamlit/secrets.toml):**
        ```toml
        GROQ_API_KEY = "gsk_your_key_here"
        ```
        """)
        st.stop()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ── Quick question chips ───────────────────────────────────────────────────
    st.markdown("**Quick questions:**")
    quick_qs = [
        "Why was R² negative in Model 2?",
        "Why was churn 100% originally?",
        "How is RFM score calculated?",
        "Explain sin/cos month encoding",
        "Why K=4 in K-Means?",
        "What does fix_models.py fix?",
        "How is sales data generated?",
        "Top feature for churn prediction?",
    ]
    cols = st.columns(4)
    for i, q in enumerate(quick_qs):
        if cols[i % 4].button(q, key=f"chip_{i}", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": q})
            with st.spinner("Thinking..."):
                try:
                    reply = call_groq(client, st.session_state.chat_history, max_tokens=600)
                except Exception as e:
                    reply = f"Error: {e}"
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()

    st.markdown("---")

    # ── CSV upload for data-specific questions ─────────────────────────────────
    with st.expander("📁 Upload a CSV for data-specific analysis (optional)"):
        uploaded = st.file_uploader("Upload any CSV", type=["csv"], key="bot_csv")
        if uploaded:
            df_up = pd.read_csv(uploaded)
            st.dataframe(df_up.head(), use_container_width=True)
            csv_summary = (
                f"File: {uploaded.name} | Rows: {len(df_up):,} | "
                f"Columns: {list(df_up.columns)} | "
                f"Sample row: {df_up.iloc[0].to_dict()}"
            )
            if st.button("🔍 Analyse with AI", type="primary"):
                q = (f"I uploaded '{uploaded.name}'. Summary: {csv_summary}. "
                     f"Analyse this data — describe what it contains and give 3 key insights.")
                st.session_state.chat_history.append({"role": "user", "content": q})
                with st.spinner("Analysing your data..."):
                    try:
                        reply = call_groq(client, st.session_state.chat_history, max_tokens=800)
                    except Exception as e:
                        reply = f"Error: {e}"
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
                st.rerun()

    # ── Conversation display ───────────────────────────────────────────────────
    if st.session_state.chat_history:
        st.markdown("### 💬 Conversation")
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user">🧑 {msg["content"]}</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bot">🤖 {msg["content"]}</div>',
                            unsafe_allow_html=True)

        if st.button("🗑️ Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()
        st.markdown("---")

    # ── Input box ─────────────────────────────────────────────────────────────
    st.markdown("### Ask a question")
    user_input = st.text_area(
        "Type your question:",
        height=100,
        key="user_input",
        placeholder="e.g. Why did we use median recency instead of 365 days for churn?"
    )
    col_send, _ = st.columns([1, 5])
    with col_send:
        send_clicked = st.button("Send ➤", type="primary", use_container_width=True)

    if send_clicked and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
        with st.spinner("AI is thinking..."):
            try:
                reply = call_groq(client, st.session_state.chat_history, max_tokens=800)
            except Exception as e:
                reply = f"Sorry, there was an error: {e}"
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()

    st.markdown("---")
    st.caption("Powered by Llama 3 (Meta) via Groq API · Free · Multi-turn conversation · Full project context")
