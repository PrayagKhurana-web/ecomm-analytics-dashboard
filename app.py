# app.py
# PURPOSE: Streamlit Web Application for E-Commerce Analytics Dashboard
# This converts our Python analysis into a live interactive web app

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
import os

# ── PAGE CONFIGURATION ────────────────────────────────────────────────────────
# WHAT: Must be the FIRST streamlit command in the script
# Sets browser tab title, icon, and layout
st.set_page_config(
    page_title="E-Commerce Analytics Dashboard",
    page_icon="🛒",
    layout="wide",           # Use full browser width
    initial_sidebar_state="expanded"  # Sidebar open by default
)

# ── CUSTOM CSS STYLING ────────────────────────────────────────────────────────
# WHAT: Custom CSS to make the app look more professional
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1565C0;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1565C0;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1565C0, #0D47A1);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: bold;
        color: #1565C0;
        border-left: 4px solid #1565C0;
        padding-left: 10px;
        margin: 1rem 0;
    }
    .stMetric label { font-size: 0.9rem !important; }
</style>
""", unsafe_allow_html=True)

# ── DATA LOADING ──────────────────────────────────────────────────────────────
# WHAT IS @st.cache_data:
# Caches the data so it doesn't reload every time user interacts
# First load takes time, subsequent loads are instant from cache
# WHY: Without cache, data reloads on every button click — very slow

@st.cache_data
def load_data():
    """Load all CSV files into DataFrames"""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base, 'dashboard', 'data')
    ml_path   = os.path.join(base, 'ml_models', 'results')

    try:
        orders    = pd.read_csv(os.path.join(data_path, 'orders.csv'))
        customers = pd.read_csv(os.path.join(data_path, 'customers.csv'))
        products  = pd.read_csv(os.path.join(data_path, 'amazon_products.csv'))
        rfm       = pd.read_csv(os.path.join(data_path, 'rfm_analysis.csv'))
        items     = pd.read_csv(os.path.join(data_path, 'order_items.csv'))
        clusters  = pd.read_csv(os.path.join(ml_path,  'cluster_summary.csv'))
        forecast  = pd.read_csv(os.path.join(ml_path,  'revenue_forecast.csv'))
        churn     = pd.read_csv(os.path.join(ml_path,  'churn_predictions.csv'))

        # Convert date columns
        orders['order_date'] = pd.to_datetime(orders['order_date'])
        orders['year']  = orders['order_date'].dt.year
        orders['month'] = orders['order_date'].dt.month

        return orders, customers, products, rfm, items, clusters, forecast, churn
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load all data
data = load_data()
if data is None:
    st.error("❌ Could not load data. Check file paths.")
    st.stop()

orders, customers, products, rfm, items, clusters, forecast, churn = data

# ── SIDEBAR NAVIGATION ────────────────────────────────────────────────────────
# WHAT IS st.sidebar: Creates a collapsible side panel
# All widgets inside sidebar appear on the left side

st.sidebar.image(
    "https://img.icons8.com/color/96/shopping-cart.png",
    width=80
)
st.sidebar.title("🛒 E-Commerce Analytics")
st.sidebar.markdown("**BCA Final Year Project**")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "📍 Navigate to:",
    ["🏠 Home",
     "📊 Sales Overview",
     "👥 Customer Analytics",
     "🛍️ Product Analytics",
     "🤖 ML Insights"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Quick Stats")
st.sidebar.metric("Total Products", f"{len(products):,}")
st.sidebar.metric("Total Customers", f"{len(customers):,}")
st.sidebar.metric("Total Orders", f"{len(orders):,}")
st.sidebar.markdown("---")
st.sidebar.info("**Data Source:** Amazon India (Real) + Simulated Sales Data")

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

    # Project overview cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📦 Products", f"{len(products):,}", "Real Amazon India Data")
    with col2:
        st.metric("👥 Customers", f"{len(customers):,}", "27 Indian Cities")
    with col3:
        st.metric("📋 Orders", f"{len(orders):,}", "2021-2024")
    with col4:
        delivered = orders[orders['status']=='Delivered']['total_amount'].sum()
        st.metric("💰 Revenue", f"Rs {delivered/1e7:.1f} Cr", "Total Delivered")

    st.markdown("---")

    # Tech stack
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🛠️ Technology Stack")
        tech_data = {
            'Layer': ['Data Storage','Data Processing','Machine Learning',
                      'Visualization','Web App','Dashboard'],
            'Tool': ['MySQL','Python + Pandas','Scikit-learn',
                     'Plotly + Matplotlib','Streamlit','Power BI'],
            'Purpose': ['Store all 4 tables','Clean & analyze data','3 ML models',
                        '18+ charts','This web app','Interactive BI dashboard']
        }
        st.dataframe(pd.DataFrame(tech_data), use_container_width=True, hide_index=True)

    with col2:
        st.markdown("### 🤖 ML Models Built")
        ml_data = {
            'Model': ['K-Means Clustering','Linear Regression','Random Forest'],
            'Purpose': ['Product Segmentation','Revenue Forecasting','Churn Prediction'],
            'Result': ['4 Clusters (Silhouette: 0.30)',
                       'R² = 0.9974 (99.74% accuracy)',
                       'Accuracy = 90.43%']
        }
        st.dataframe(pd.DataFrame(ml_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 📍 Navigate Using the Sidebar →")
    st.info("Use the sidebar on the left to navigate between different analytics pages.")

# ══════════════════════════════════════════════════════════════════════════════
# SALES OVERVIEW PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Sales Overview":
    st.markdown('<div class="main-header">📊 Sales Overview Dashboard</div>',
                unsafe_allow_html=True)

    # ── Sidebar Filters ───────────────────────────────────────────────────────
    st.sidebar.markdown("### 🔧 Filters")
    years = sorted(orders['year'].unique())
    selected_years = st.sidebar.multiselect(
        "Select Year(s):", years, default=years
    )
    status_filter = st.sidebar.multiselect(
        "Order Status:",
        orders['status'].unique(),
        default=['Delivered']
    )

    # Apply filters
    filtered = orders[
        (orders['year'].isin(selected_years)) &
        (orders['status'].isin(status_filter))
    ]

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 Total Revenue",
                  f"Rs {filtered['total_amount'].sum()/1e7:.2f} Cr")
    with col2:
        st.metric("📦 Total Orders", f"{len(filtered):,}")
    with col3:
        merged = filtered.merge(customers[['customer_id']], on='customer_id')
        st.metric("👥 Active Customers", f"{merged['customer_id'].nunique():,}")
    with col4:
        st.metric("📈 Avg Order Value",
                  f"Rs {filtered['total_amount'].mean():,.0f}")

    st.markdown("---")

    # ── Charts Row 1 ──────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">📅 Monthly Revenue Trend</div>',
                    unsafe_allow_html=True)
        monthly = filtered.groupby(['year','month'])['total_amount'].sum().reset_index()
        monthly['period'] = monthly['year'].astype(str) + '-' + \
                             monthly['month'].astype(str).str.zfill(2)
        monthly = monthly.sort_values('period')

        fig = px.line(monthly, x='period', y='total_amount',
                      title='Monthly Revenue (Rs)',
                      color_discrete_sequence=['#1565C0'])
        fig.update_layout(xaxis_tickangle=45, height=350,
                          xaxis_title='Month', yaxis_title='Revenue (Rs)')
        fig.update_traces(line_width=2.5)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">💳 Payment Method Analysis</div>',
                    unsafe_allow_html=True)
        payment = filtered.groupby('payment_method').agg(
            orders=('order_id','count'),
            revenue=('total_amount','sum')
        ).reset_index().sort_values('orders', ascending=False)

        fig = px.bar(payment, x='payment_method', y='orders',
                     title='Orders by Payment Method',
                     color='orders',
                     color_continuous_scale='Blues')
        fig.update_layout(height=350, showlegend=False,
                          xaxis_title='Payment Method',
                          yaxis_title='Number of Orders')
        st.plotly_chart(fig, use_container_width=True)

    # ── Charts Row 2 ──────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">🔄 Order Status Breakdown</div>',
                    unsafe_allow_html=True)
        status_data = orders.groupby('status')['order_id'].count().reset_index()
        status_data.columns = ['status','count']

        fig = px.pie(status_data, values='count', names='status',
                     title='Order Status Distribution',
                     hole=0.4,
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

        fig = px.bar(city_rev, x='total_amount', y='city',
                     orientation='h',
                     title='Top 10 Cities by Revenue',
                     color='total_amount',
                     color_continuous_scale='Viridis')
        fig.update_layout(height=350, showlegend=False,
                          xaxis_title='Revenue (Rs)',
                          yaxis_title='City')
        st.plotly_chart(fig, use_container_width=True)

    # ── Yearly Growth Table ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">📈 Yearly Revenue Summary</div>',
                unsafe_allow_html=True)
    yearly = orders[orders['status']=='Delivered'].groupby('year').agg(
        orders=('order_id','count'),
        revenue=('total_amount','sum'),
        avg_order=('total_amount','mean')
    ).round(2).reset_index()
    yearly['revenue_cr'] = (yearly['revenue']/1e7).round(2)
    yearly['yoy_growth'] = yearly['revenue'].pct_change()*100
    yearly.columns = ['Year','Orders','Revenue (Rs)','Avg Order (Rs)',
                       'Revenue (Cr)','YoY Growth %']
    st.dataframe(yearly.style.format({
        'Revenue (Rs)': 'Rs {:,.0f}',
        'Avg Order (Rs)': 'Rs {:,.0f}',
        'Revenue (Cr)': 'Rs {:.2f} Cr',
        'YoY Growth %': '{:+.1f}%'
    }), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOMER ANALYTICS PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Customer Analytics":
    st.markdown('<div class="main-header">👥 Customer Analytics Dashboard</div>',
                unsafe_allow_html=True)

    # Sidebar filter
    st.sidebar.markdown("### 🔧 Filters")
    seg_filter = st.sidebar.multiselect(
        "Customer Segment:",
        rfm['segment'].unique(),
        default=rfm['segment'].unique()
    )
    filtered_rfm = rfm[rfm['segment'].isin(seg_filter)]

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("👥 Total Customers", f"{len(customers):,}")
    with col2:
        st.metric("⭐ Avg RFM Score", f"{rfm['RFM_score'].mean():.2f}")
    with col3:
        st.metric("🏆 Champions", f"{len(rfm[rfm['segment']=='Champions']):,}")
    with col4:
        st.metric("⚠️ At Risk", f"{len(rfm[rfm['segment']=='At Risk']):,}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">🏷️ Customer Segments</div>',
                    unsafe_allow_html=True)
        seg_count = filtered_rfm.groupby('segment').size().reset_index(name='count')
        seg_count = seg_count.sort_values('count', ascending=True)

        fig = px.bar(seg_count, x='count', y='segment',
                     orientation='h',
                     color='count',
                     color_continuous_scale='Blues',
                     title='Customers per RFM Segment')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">💰 Revenue by Segment</div>',
                    unsafe_allow_html=True)
        seg_rev = filtered_rfm.groupby('segment')['monetary'].mean().reset_index()
        seg_rev = seg_rev.sort_values('monetary', ascending=True)

        fig = px.bar(seg_rev, x='monetary', y='segment',
                     orientation='h',
                     color='monetary',
                     color_continuous_scale='Greens',
                     title='Avg Spend per Segment (Rs)')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">👫 Gender Distribution</div>',
                    unsafe_allow_html=True)
        gender = customers.groupby('gender').size().reset_index(name='count')
        fig = px.pie(gender, values='count', names='gender',
                     hole=0.4, title='Gender Distribution',
                     color_discrete_sequence=['#1565C0','#FF6B6B'])
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">📊 Age Group Distribution</div>',
                    unsafe_allow_html=True)
        customers['age_group'] = pd.cut(
            customers['age'],
            bins=[17,25,35,45,55,100],
            labels=['18-25','26-35','36-45','46-55','56+']
        )
        age_data = customers.groupby('age_group', observed=True).size().reset_index(name='count')
        fig = px.bar(age_data, x='age_group', y='count',
                     title='Customers by Age Group',
                     color='count',
                     color_continuous_scale='Purples')
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # RFM Scatter Plot
    st.markdown('<div class="section-header">🎯 RFM Scatter — Recency vs Monetary</div>',
                unsafe_allow_html=True)
    scatter_sample = filtered_rfm.sample(min(2000, len(filtered_rfm)), random_state=42)
    fig = px.scatter(scatter_sample, x='recency', y='monetary',
                     color='segment', size='frequency',
                     hover_data=['segment','frequency','monetary'],
                     title='Customer Segments — Recency vs Spend',
                     labels={'recency':'Days Since Last Purchase',
                             'monetary':'Total Spend (Rs)'})
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PRODUCT ANALYTICS PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🛍️ Product Analytics":
    st.markdown('<div class="main-header">🛍️ Product Analytics Dashboard</div>',
                unsafe_allow_html=True)

    # Sidebar filter
    st.sidebar.markdown("### 🔧 Filters")
    cat_filter = st.sidebar.multiselect(
        "Category:",
        sorted(products['main_category'].dropna().unique()),
        default=sorted(products['main_category'].dropna().unique())[:5]
    )

    filtered_prod = products[products['main_category'].isin(cat_filter)]

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📦 Products", f"{len(filtered_prod):,}")
    with col2:
        st.metric("⭐ Avg Rating",
                  f"{filtered_prod['ratings'].mean():.2f}")
    with col3:
        st.metric("🏷️ Avg Discount",
                  f"{filtered_prod['discount_pct'].mean():.1f}%")
    with col4:
        st.metric("💰 Avg Price",
                  f"Rs {filtered_prod['discount_price'].mean():,.0f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">📊 Products per Category</div>',
                    unsafe_allow_html=True)
        cat_count = filtered_prod.groupby('main_category').size().reset_index(name='count')
        cat_count = cat_count.sort_values('count', ascending=True)
        fig = px.bar(cat_count, x='count', y='main_category',
                     orientation='h', color='count',
                     color_continuous_scale='Blues',
                     title='Number of Products per Category')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">🏷️ Avg Discount by Category</div>',
                    unsafe_allow_html=True)
        cat_disc = filtered_prod.groupby('main_category')['discount_pct'].mean().reset_index()
        cat_disc = cat_disc.sort_values('discount_pct', ascending=True)
        fig = px.bar(cat_disc, x='discount_pct', y='main_category',
                     orientation='h', color='discount_pct',
                     color_continuous_scale='Reds',
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
            ['name','main_category','discount_price','discount_pct',
             'ratings','value_score']
        ]
        top_products['name'] = top_products['name'].str[:40]
        top_products.columns = ['Product','Category','Price(Rs)',
                                  'Discount%','Rating','Value Score']
        st.dataframe(top_products, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# ML INSIGHTS PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 ML Insights":
    st.markdown('<div class="main-header">🤖 Machine Learning Insights</div>',
                unsafe_allow_html=True)

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 K-Means Clusters", "4")
    with col2:
        st.metric("📈 Revenue R²", "0.9974")
    with col3:
        st.metric("🌲 RF Accuracy", "90.43%")
    with col4:
        high_risk = len(churn[churn['churn_probability'] > 0.7]) \
                    if 'churn_probability' in churn.columns else \
                    int(churn['churned'].sum())
        st.metric("⚠️ High Risk Customers", f"{high_risk:,}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">🔵 K-Means Product Clusters</div>',
                    unsafe_allow_html=True)
        fig = px.bar(clusters, x='cluster_name', y='product_count',
                     color='avg_price',
                     color_continuous_scale='Viridis',
                     title='Products per Cluster',
                     text='product_count')
        fig.update_traces(textposition='outside')
        fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">💰 Avg Price per Cluster</div>',
                    unsafe_allow_html=True)
        fig = px.bar(clusters, x='cluster_name', y='avg_price',
                     color='avg_discount_pct',
                     color_continuous_scale='RdYlGn',
                     title='Average Price per Cluster (Rs)',
                     text='avg_price')
        fig.update_traces(texttemplate='Rs%{text:,.0f}', textposition='outside')
        fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Revenue Forecast
    st.markdown('<div class="section-header">📈 Revenue Forecast — Actual vs Predicted</div>',
                unsafe_allow_html=True)
    forecast['period'] = forecast['year'].astype(str) + '-' + \
                          forecast['month'].astype(str).str.zfill(2)
    forecast_sorted = forecast.sort_values('period')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast_sorted['period'],
        y=forecast_sorted['revenue'],
        name='Actual Revenue',
        line=dict(color='#1565C0', width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=forecast_sorted['period'],
        y=forecast_sorted['predicted_revenue'],
        name='Predicted Revenue',
        line=dict(color='#FF6B6B', width=2, dash='dash')
    ))
    fig.update_layout(
        title='Revenue: Actual vs ML Predicted (Linear Regression R²=0.9974)',
        xaxis_title='Month',
        yaxis_title='Revenue (Rs)',
        height=400,
        xaxis_tickangle=45,
        legend=dict(orientation='h', y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

    # RFM Segment Table
    st.markdown('<div class="section-header">🏷️ RFM Segment Analysis</div>',
                unsafe_allow_html=True)
    seg_summary = rfm.groupby('segment').agg(
        Customers=('customer_id','count'),
        Avg_Recency=('recency','mean'),
        Avg_Frequency=('frequency','mean'),
        Avg_Spend=('monetary','mean'),
        Avg_RFM=('RFM_score','mean')
    ).round(2).reset_index()
    seg_summary = seg_summary.sort_values('Avg_Spend', ascending=False)
    seg_summary.columns = ['Segment','Customers','Avg Recency(days)',
                             'Avg Orders','Avg Spend(Rs)','Avg RFM Score']
    st.dataframe(
        seg_summary.style.background_gradient(subset=['Avg Spend(Rs)'],
                                               cmap='Greens'),
        use_container_width=True,
        hide_index=True
    )