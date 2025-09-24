import streamlit as st
import sys
import os
import pandas as pd

# Add src to Python path
sys.path.append('/app/src')

st.set_page_config(
    page_title="AI Marketing Automation",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 AI Marketing Automation Dashboard")

try:
    from apis.api_manager import UnifiedMarketingAPI
    
    # Initialize API manager
    api_manager = UnifiedMarketingAPI()
    
    st.success("✅ System initialized successfully!")
    
    # Fetch data
    with st.spinner("Loading campaign data..."):
        data = api_manager.fetch_all_platform_data()
        
        if data['status'] == 'success':
            combined_df = data['combined_data']
            meta_df = data['meta_data']
            google_df = data['google_data']
            
            # Show API status
            api_status = api_manager.get_api_status()
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Meta API:** {api_status['meta_api']}")
            with col2:
                st.write(f"**Google API:** {api_status['google_api']}")
            
            # Show metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_impressions = combined_df['impressions'].sum()
                st.metric("Total Impressions", f"{total_impressions:,.0f}")
            
            with col2:
                total_clicks = combined_df['clicks'].sum()
                st.metric("Total Clicks", f"{total_clicks:,.0f}")
            
            with col3:
                avg_ctr = combined_df['ctr'].mean()
                st.metric("Avg CTR", f"{avg_ctr:.2f}%")
            
            with col4:
                avg_roas = combined_df['roas'].mean()
                st.metric("Avg ROAS", f"{avg_roas:.2f}x")
            
            # Platform comparison
            st.subheader("Platform Performance Comparison")
            platform_summary = combined_df.groupby('platform').agg({
                'impressions': 'sum',
                'clicks': 'sum',
                'spend': 'sum',
                'conversions': 'sum',
                'roas': 'mean',
                'ctr': 'mean'
            }).round(2)
            
            st.dataframe(platform_summary, use_container_width=True)
            
            # Show recent campaign performance
            st.subheader("Recent Campaign Performance")
            # Get latest 10 campaigns
            latest_data = combined_df.sort_values('date', ascending=False).head(10)
            st.dataframe(latest_data[['date', 'platform', 'campaign_name', 'impressions', 'clicks', 'spend', 'conversions', 'roas']], use_container_width=True)
            
            # Data sources info
            st.subheader("Data Sources")
            sources = data['data_sources']
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Meta Data Source:** {sources['meta']}")
                st.write(f"**Meta Campaigns:** {len(meta_df)}")
            with col2:
                st.write(f"**Google Data Source:** {sources['google']}")
                st.write(f"**Google Campaigns:** {len(google_df)}")
            
        else:
            st.error("Failed to load data")
            
except ImportError as e:
    st.error(f"Import Error: {str(e)}")
    st.write("Check that all API manager files are in correct locations")
    
except Exception as e:
    st.error(f"System Error: {str(e)}")
    st.write("Please check the logs for more details")

st.write("---")
st.write(f"**System Status:** Running with enhanced unified API manager")
st.write(f"**Last Updated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")