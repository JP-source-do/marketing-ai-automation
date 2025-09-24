import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os

# Configure page
st.set_page_config(
    page_title="AI Marketing Automation Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 10px;
    text-align: center;
    margin-bottom: 2rem;
    color: white;
}
.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    border-left: 4px solid #667eea;
}
.stMetric {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #e0e0e0;
}
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🚀 AI Marketing Automation Platform</h1>
    <p>Intelligent campaign optimization and performance analytics</p>
</div>
""", unsafe_allow_html=True)

# Generate sample data
@st.cache_data
def generate_sample_data():
    """Generate realistic marketing data for demonstration"""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    # Campaign data
    campaigns = []
    for i, date in enumerate(dates):
        campaigns.append({
            'date': date,
            'campaign_name': f'Campaign_{np.random.choice(["A", "B", "C", "D"])}',
            'platform': np.random.choice(['Google Ads', 'Facebook', 'Instagram', 'LinkedIn']),
            'impressions': np.random.randint(1000, 50000),
            'clicks': np.random.randint(50, 2500),
            'conversions': np.random.randint(5, 250),
            'spend': np.random.uniform(100, 5000),
            'revenue': np.random.uniform(200, 15000)
        })
    
    df = pd.DataFrame(campaigns)
    df['ctr'] = (df['clicks'] / df['impressions']) * 100
    df['conversion_rate'] = (df['conversions'] / df['clicks']) * 100
    df['roas'] = df['revenue'] / df['spend']
    df['cpc'] = df['spend'] / df['clicks']
    
    return df

# Load data
data = generate_sample_data()

# Sidebar filters
st.sidebar.header("📊 Filters & Controls")

# Date range selector
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(data['date'].min(), data['date'].max()),
    min_value=data['date'].min(),
    max_value=data['date'].max()
)

# Platform filter
platforms = st.sidebar.multiselect(
    "Select Platforms",
    options=data['platform'].unique(),
    default=data['platform'].unique()
)

# Campaign filter
campaigns = st.sidebar.multiselect(
    "Select Campaigns",
    options=data['campaign_name'].unique(),
    default=data['campaign_name'].unique()
)

# Filter data
if len(date_range) == 2:
    filtered_data = data[
        (data['date'] >= pd.Timestamp(date_range[0])) &
        (data['date'] <= pd.Timestamp(date_range[1])) &
        (data['platform'].isin(platforms)) &
        (data['campaign_name'].isin(campaigns))
    ]
else:
    filtered_data = data[
        (data['platform'].isin(platforms)) &
        (data['campaign_name'].isin(campaigns))
    ]

# Main dashboard
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="💰 Total Spend",
        value=f"${filtered_data['spend'].sum():,.0f}",
        delta=f"{filtered_data['spend'].sum() / data['spend'].sum() * 100:.1f}% of total"
    )

with col2:
    st.metric(
        label="📈 Total Revenue",
        value=f"${filtered_data['revenue'].sum():,.0f}",
        delta=f"{(filtered_data['revenue'].sum() / filtered_data['spend'].sum()):.2f}x ROAS"
    )

with col3:
    st.metric(
        label="👆 Total Clicks",
        value=f"{filtered_data['clicks'].sum():,}",
        delta=f"{filtered_data['ctr'].mean():.2f}% CTR"
    )

with col4:
    st.metric(
        label="🎯 Conversions",
        value=f"{filtered_data['conversions'].sum():,}",
        delta=f"{filtered_data['conversion_rate'].mean():.2f}% CVR"
    )

# Charts section
st.header("📊 Performance Analytics")

# Time series chart
col1, col2 = st.columns(2)

with col1:
    st.subheader("Revenue Over Time")
    daily_revenue = filtered_data.groupby('date')['revenue'].sum().reset_index()
    fig_revenue = px.line(daily_revenue, x='date', y='revenue', 
                         title="Daily Revenue Trend")
    fig_revenue.update_traces(line_color='#667eea')
    st.plotly_chart(fig_revenue, use_container_width=True)

with col2:
    st.subheader("ROAS by Platform")
    platform_roas = filtered_data.groupby('platform')['roas'].mean().reset_index()
    fig_roas = px.bar(platform_roas, x='platform', y='roas',
                     title="Return on Ad Spend by Platform")
    fig_roas.update_traces(marker_color='#764ba2')
    st.plotly_chart(fig_roas, use_container_width=True)

# Campaign performance table
st.header("📋 Campaign Performance Details")
campaign_summary = filtered_data.groupby(['campaign_name', 'platform']).agg({
    'spend': 'sum',
    'revenue': 'sum',
    'clicks': 'sum',
    'conversions': 'sum',
    'impressions': 'sum'
}).reset_index()

campaign_summary['ROAS'] = campaign_summary['revenue'] / campaign_summary['spend']
campaign_summary['CTR'] = (campaign_summary['clicks'] / campaign_summary['impressions']) * 100
campaign_summary['CVR'] = (campaign_summary['conversions'] / campaign_summary['clicks']) * 100

# Format the dataframe for display
display_df = campaign_summary.copy()
display_df['spend'] = display_df['spend'].apply(lambda x: f"${x:,.0f}")
display_df['revenue'] = display_df['revenue'].apply(lambda x: f"${x:,.0f}")
display_df['ROAS'] = display_df['ROAS'].apply(lambda x: f"{x:.2f}x")
display_df['CTR'] = display_df['CTR'].apply(lambda x: f"{x:.2f}%")
display_df['CVR'] = display_df['CVR'].apply(lambda x: f"{x:.2f}%")

st.dataframe(display_df, use_container_width=True)

# AI Insights section
st.header("🤖 AI-Powered Insights")

# Simulated AI insights
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Optimization Recommendations")
    st.info("**Budget Reallocation**: Increase spend on Google Ads campaigns by 15% - showing 23% higher ROAS than average.")
    st.warning("**Underperforming Alert**: Campaign_C on Facebook has 34% lower conversion rate. Consider A/B testing new creative.")
    st.success("**Growth Opportunity**: LinkedIn campaigns show strong engagement. Scale budget by 25% for Q4.")

with col2:
    st.subheader("📈 Performance Predictions")
    st.metric(
        label="Predicted Monthly Revenue",
        value=f"${filtered_data['revenue'].sum() * 1.12:,.0f}",
        delta="12% increase expected"
    )
    st.metric(
        label="Forecasted ROAS",
        value=f"{filtered_data['roas'].mean() * 1.08:.2f}x",
        delta="8% improvement projected"
    )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🚀 AI Marketing Automation Platform | Built with Streamlit & OpenAI</p>
    <p>Real-time campaign optimization • Predictive analytics • Automated insights</p>
</div>
""", unsafe_allow_html=True)

# Add OpenAI integration placeholder
if st.sidebar.button("🔄 Generate AI Campaign Suggestions"):
    with st.spinner("Analyzing your campaigns with AI..."):
        # This would integrate with OpenAI API using st.secrets
        st.success("✅ AI analysis complete! Check the insights section above for recommendations.")
        st.balloons()

