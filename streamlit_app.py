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
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    margin: 0.5rem 0;
}
.success-metric {
    background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
}
.warning-metric {
    background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
}
.danger-metric {
    background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'campaigns_data' not in st.session_state:
    st.session_state.campaigns_data = None
if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = None

class StreamlitAIMarketing:
    def __init__(self):
        self.initialize_demo_data()
    
    def initialize_demo_data(self):
        """Initialize demo data for the platform"""
        np.random.seed(42)
        
        # Campaign data
        self.campaigns = pd.DataFrame({
            'campaign_id': [f'CAMP_{i:03d}' for i in range(1, 21)],
            'campaign_name': [f'Campaign {i}' for i in range(1, 21)],
            'platform': np.random.choice(['Google Ads', 'Facebook', 'Instagram', 'LinkedIn', 'Twitter'], 20),
            'budget': np.random.uniform(1000, 10000, 20).round(2),
            'spend': np.random.uniform(500, 8000, 20).round(2),
            'impressions': np.random.randint(10000, 1000000, 20),
            'clicks': np.random.randint(100, 50000, 20),
            'conversions': np.random.randint(10, 2000, 20),
            'revenue': np.random.uniform(1000, 50000, 20).round(2),
            'status': np.random.choice(['Active', 'Paused', 'Draft'], 20, p=[0.7, 0.2, 0.1])
        })
        
        # Calculate metrics
        self.campaigns['ctr'] = (self.campaigns['clicks'] / self.campaigns['impressions'] * 100).round(2)
        self.campaigns['cpc'] = (self.campaigns['spend'] / self.campaigns['clicks']).round(2)
        self.campaigns['roas'] = (self.campaigns['revenue'] / self.campaigns['spend']).round(2)
        self.campaigns['conversion_rate'] = (self.campaigns['conversions'] / self.campaigns['clicks'] * 100).round(2)
        
        # Time series data
        dates = pd.date_range(start='2024-01-01', end='2024-09-24', freq='D')
        self.time_series = pd.DataFrame({
            'date': dates,
            'spend': np.random.uniform(500, 2000, len(dates)) + np.sin(np.arange(len(dates)) * 2 * np.pi / 30) * 200,
            'revenue': np.random.uniform(1000, 5000, len(dates)) + np.sin(np.arange(len(dates)) * 2 * np.pi / 30) * 500,
            'conversions': np.random.randint(10, 100, len(dates)),
            'impressions': np.random.randint(5000, 50000, len(dates))
        })
        
        self.time_series['roas'] = (self.time_series['revenue'] / self.time_series['spend']).round(2)

    def render_dashboard(self):
        st.title("🚀 AI Marketing Automation Platform")
        st.markdown("---")
        
        # Sidebar
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox("Choose a page", [
            "📊 Dashboard Overview",
            "🎯 Campaign Management", 
            "🤖 AI Optimization",
            "📈 Performance Analytics",
            "⚙️ Settings"
        ])
        
        if page == "📊 Dashboard Overview":
            self.render_overview()
        elif page == "🎯 Campaign Management":
            self.render_campaigns()
        elif page == "🤖 AI Optimization":
            self.render_ai_optimization()
        elif page == "📈 Performance Analytics":
            self.render_analytics()
        elif page == "⚙️ Settings":
            self.render_settings()

    def render_overview(self):
        st.header("📊 Dashboard Overview")
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_spend = self.campaigns['spend'].sum()
        total_revenue = self.campaigns['revenue'].sum()
        total_conversions = self.campaigns['conversions'].sum()
        avg_roas = (total_revenue / total_spend)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>💰 Total Spend</h3>
                <h2>${total_spend:,.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="success-metric">
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
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    margin: 0.5rem 0;
}
.success-metric {
    background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
}
.warning-metric {
    background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
}
.danger-metric {
    background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'campaigns_data' not in st.session_state:
    st.session_state.campaigns_data = None
if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = None

class StreamlitAIMarketing:
    def __init__(self):
        self.initialize_demo_data()
    
    def initialize_demo_data(self):
        """Initialize demo data for the platform"""
        np.random.seed(42)
        
        # Campaign data
        self.campaigns = pd.DataFrame({
            'campaign_id': [f'CAMP_{i:03d}' for i in range(1, 21)],
            'campaign_name': [f'Campaign {i}' for i in range(1, 21)],
            'platform': np.random.choice(['Google Ads', 'Facebook', 'Instagram', 'LinkedIn', 'Twitter'], 20),
            'budget': np.random.uniform(1000, 10000, 20).round(2),
            'spend': np.random.uniform(500, 8000, 20).round(2),
            'impressions': np.random.randint(10000, 1000000, 20),
            'clicks': np.random.randint(100, 50000, 20),
            'conversions': np.random.randint(10, 2000, 20),
            'revenue': np.random.uniform(1000, 50000, 20).round(2),
            'status': np.random.choice(['Active', 'Paused', 'Draft'], 20, p=[0.7, 0.2, 0.1])
        })
        
        # Calculate metrics
        self.campaigns['ctr'] = (self.campaigns['clicks'] / self.campaigns['impressions'] * 100).round(2)
        self.campaigns['cpc'] = (self.campaigns['spend'] / self.campaigns['clicks']).round(2)
        self.campaigns['roas'] = (self.campaigns['revenue'] / self.campaigns['spend']).round(2)
        self.campaigns['conversion_rate'] = (self.campaigns['conversions'] / self.campaigns['clicks'] * 100).round(2)
        
        # Time series data
        dates = pd.date_range(start='2024-01-01', end='2024-09-24', freq='D')
        self.time_series = pd.DataFrame({
            'date': dates,
            'spend': np.random.uniform(500, 2000, len(dates)) + np.sin(np.arange(len(dates)) * 2 * np.pi / 30) * 200,
            'revenue': np.random.uniform(1000, 5000, len(dates)) + np.sin(np.arange(len(dates)) * 2 * np.pi / 30) * 500,
            'conversions': np.random.randint(10, 100, len(dates)),
            'impressions': np.random.randint(5000, 50000, len(dates))
        })
        
        self.time_series['roas'] = (self.time_series['revenue'] / self.time_series['spend']).round(2)

    def render_dashboard(self):
        st.title("🚀 AI Marketing Automation Platform")
        st.markdown("---")
        
        # Sidebar
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox("Choose a page", [
            "📊 Dashboard Overview",
            "🎯 Campaign Management", 
            "🤖 AI Optimization",
            "📈 Performance Analytics",
            "⚙️ Settings"
        ])
        
            self.render_analytics()
        elif page == "⚙️ Settings":
            self.render_settings()

    def render_overview(self):
        st.header("📊 Dashboard Overview")
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_spend = self.campaigns['spend'].sum()
        total_revenue = self.campaigns['revenue'].sum()
        total_conversions = self.campaigns['conversions'].sum()
        avg_roas = (total_revenue / total_spend)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>💰 Total Spend</h3>
                <h2>${total_spend:,.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="success-metric">
                <h3>📈 Total Revenue</h3>
                <h2>${total_revenue:,.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="warning-metric">
                <h3>🎯 Conversions</h3>
                <h2>{total_conversions:,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            roas_color = "success-metric" if avg_roas > 3 else "warning-metric" if avg_roas > 2 else "danger-metric"
            st.markdown(f"""
            <div class="{roas_color}">
                <h3>⚡ Avg ROAS</h3>
                <h2>{avg_roas:.2f}x</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💸 Spend vs Revenue Over Time")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=self.time_series['date'], 
                y=self.time_series['spend'],
                name='Spend',
                line=dict(color='#ff6b6b')
            ))
            fig.add_trace(go.Scatter(
                x=self.time_series['date'], 
                y=self.time_series['revenue'],
                name='Revenue',
                line=dict(color='#4ecdc4')
            ))
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Campaign Performance by Platform")
            platform_perf = self.campaigns.groupby('platform').agg({
                'spend': 'sum',
                'revenue': 'sum',
                'conversions': 'sum'
            }).reset_index()
            platform_perf['roas'] = platform_perf['revenue'] / platform_perf['spend']
            
            fig = px.bar(platform_perf, x='platform', y='roas', 
                        title="ROAS by Platform",
                        color='roas',
                        color_continuous_scale='viridis')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    def render_campaigns(self):
        st.header("🎯 Campaign Management")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            platform_filter = st.selectbox("Filter by Platform", 
                                         ['All'] + list(self.campaigns['platform'].unique()))
        with col2:
            status_filter = st.selectbox("Filter by Status",
                                       ['All'] + list(self.campaigns['status'].unique()))
        with col3:
            min_roas = st.slider("Minimum ROAS", 0.0, 10.0, 0.0)
        
        # Apply filters
        filtered_campaigns = self.campaigns.copy()
        if platform_filter != 'All':
            filtered_campaigns = filtered_campaigns[filtered_campaigns['platform'] == platform_filter]
        if status_filter != 'All':
            filtered_campaigns = filtered_campaigns[filtered_campaigns['status'] == status_filter]
        filtered_campaigns = filtered_campaigns[filtered_campaigns['roas'] >= min_roas]
        
        # Campaign table
        st.subheader(f"📋 Campaigns ({len(filtered_campaigns)} found)")
        
        # Format the dataframe for display
        display_df = filtered_campaigns[['campaign_name', 'platform', 'status', 'budget', 
                                       'spend', 'revenue', 'conversions', 'ctr', 'roas']].copy()
        display_df['budget'] = display_df['budget'].apply(lambda x: f"${x:,.2f}")
        display_df['spend'] = display_df['spend'].apply(lambda x: f"${x:,.2f}")
        display_df['revenue'] = display_df['revenue'].apply(lambda x: f"${x:,.2f}")
        display_df['ctr'] = display_df['ctr'].apply(lambda x: f"{x}%")
        display_df['roas'] = display_df['roas'].apply(lambda x: f"{x:.2f}x")
        
        st.dataframe(display_df, use_container_width=True)
        
        # Campaign actions
        st.subheader("⚡ Quick Actions")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🚀 Optimize All Campaigns"):
                st.success("AI optimization started for all campaigns!")
        with col2:
            if st.button("⏸️ Pause Low Performers"):
                low_performers = len(filtered_campaigns[filtered_campaigns['roas'] < 2])
                st.warning(f"Paused {low_performers} campaigns with ROAS < 2.0")
        with col3:
            if st.button("📊 Generate Report"):
                st.info("Comprehensive report generated and saved!")

    def render_ai_optimization(self):
        st.header("🤖 AI Optimization Engine")
        
        st.markdown("""
        Our AI engine continuously analyzes your campaigns and provides intelligent recommendations
        to maximize your ROAS and reduce wasted spend.
        """)
        
        # Optimization options
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Optimization Settings")
            
            target_roas = st.slider("Target ROAS", 1.0, 10.0, 4.0, 0.1)
            max_budget = st.number_input("Maximum Daily Budget", 1000, 50000, 10000, 500)
            
            optimization_type = st.selectbox("Optimization Strategy", [
                "Maximize ROAS",
                "Maximize Conversions", 
                "Minimize Cost per Acquisition",
                "Balanced Growth"
            ])
            
            if st.button("🚀 Run AI Optimization"):
                with st.spinner("AI is analyzing your campaigns..."):
                    import time
                    time.sleep(3)  # Simulate processing
                    
                    st.session_state.optimization_results = {
                        'recommendations': [
                            {"campaign": "Campaign 1", "action": "Increase budget by 25%", "impact": "+15% ROAS"},
                            {"campaign": "Campaign 5", "action": "Pause keywords with CTR < 2%", "impact": "+8% efficiency"},
                            {"campaign": "Campaign 12", "action": "Shift budget to mobile devices", "impact": "+12% conversions"},
                            {"campaign": "Campaign 8", "action": "Update ad creative", "impact": "+20% CTR"},
                            {"campaign": "Campaign 3", "action": "Expand to similar audiences", "impact": "+30% reach"}
                        ],
                        'predicted_improvements': {
                            'roas_increase': 18.5,
                            'cost_savings': 2340.50,
                            'conversion_increase': 156
                        }
                    }
                    
                st.success("✅ AI Optimization Complete!")
        
        with col2:
            st.subheader("📊 Current Performance")
            
            # Performance metrics
            current_roas = self.campaigns['roas'].mean()
            st.metric("Current Avg ROAS", f"{current_roas:.2f}x", 
                     delta=f"{current_roas - 3:.2f}x vs target")
            
            underperforming = len(self.campaigns[self.campaigns['roas'] < target_roas])
            st.metric("Underperforming Campaigns", underperforming, 
                     delta=f"{underperforming}/{len(self.campaigns)} campaigns")
            
            total_waste = self.campaigns[self.campaigns['roas'] < 2]['spend'].sum()
            st.metric("Estimated Waste", f"${total_waste:,.2f}", 
                     delta=f"{(total_waste/self.campaigns['spend'].sum()*100):.1f}% of spend")
        
        # Display optimization results
        if st.session_state.optimization_results:
            st.markdown("---")
            st.subheader("🎯 AI Recommendations")
            
            for i, rec in enumerate(st.session_state.optimization_results['recommendations']):
                col1, col2, col3, col4 = st.columns([2, 3, 2, 1])
                with col1:
                    st.write(f"**{rec['campaign']}**")
                with col2:
                    st.write(rec['action'])
                with col3:
                    st.write(f"🎯 {rec['impact']}")
                with col4:
                    if st.button("Apply", key=f"apply_{i}"):
                        st.success(f"Applied to {rec['campaign']}")
            
            st.subheader("📈 Predicted Impact")
            col1, col2, col3 = st.columns(3)
            results = st.session_state.optimization_results['predicted_improvements']
            
            with col1:
                st.metric("ROAS Increase", f"+{results['roas_increase']}%")
            with col2:
                st.metric("Cost Savings", f"${results['cost_savings']:,.2f}")
            with col3:
                st.metric("Additional Conversions", f"+{results['conversion_increase']}")

    def render_analytics(self):
        st.header("📈 Performance Analytics")
        
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
        with col2:
            end_date = st.date_input("End Date", datetime.now())
        
        # Filter time series data
        mask = (self.time_series['date'] >= pd.Timestamp(start_date)) & (self.time_series['date'] <= pd.Timestamp(end_date))
        filtered_ts = self.time_series.loc[mask]
        
        # Key metrics over time
        st.subheader("📊 Performance Trends")
        
        metric_choice = st.selectbox("Select Metric", ["ROAS", "Spend", "Revenue", "Conversions"])
        
        fig = go.Figure()
        if metric_choice == "ROAS":
            fig.add_trace(go.Scatter(x=filtered_ts['date'], y=filtered_ts['roas'], 
                                   name='ROAS', line=dict(color='#4ecdc4')))
        elif metric_choice == "Spend":
            fig.add_trace(go.Scatter(x=filtered_ts['date'], y=filtered_ts['spend'], 
                                   name='Spend', line=dict(color='#ff6b6b')))
        elif metric_choice == "Revenue":
            fig.add_trace(go.Scatter(x=filtered_ts['date'], y=filtered_ts['revenue'], 
                                   name='Revenue', line=dict(color='#4ecdc4')))
        elif metric_choice == "Conversions":
            fig.add_trace(go.Scatter(x=filtered_ts['date'], y=filtered_ts['conversions'], 
                                   name='Conversions', line=dict(color='#45b7d1')))
        
        fig.update_layout(height=400, title=f"{metric_choice} Over Time")
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Top Performing Campaigns")
            top_campaigns = self.campaigns.nlargest(5, 'roas')[['campaign_name', 'platform', 'roas', 'revenue']]
            st.dataframe(top_campaigns, use_container_width=True)
        
        with col2:
            st.subheader("⚠️ Campaigns Needing Attention")
            bottom_campaigns = self.campaigns.nsmallest(5, 'roas')[['campaign_name', 'platform', 'roas', 'spend']]
            st.dataframe(bottom_campaigns, use_container_width=True)

    def render_settings(self):
        st.header("⚙️ Settings & Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔐 API Configuration")
            st.info("Configure your API keys in Streamlit Cloud secrets")
            
            apis_configured = {
                "OpenAI": "✅ Configured" if st.secrets.get("OPENAI_API_KEY", "") else "❌ Not configured",
                "Google Ads": "✅ Configured" if st.secrets.get("GOOGLE_ADS_API_KEY", "") else "❌ Not configured", 
                "Facebook": "✅ Configured" if st.secrets.get("FACEBOOK_API_KEY", "") else "❌ Not configured",
                "LinkedIn": "✅ Configured" if st.secrets.get("LINKEDIN_API_KEY", "") else "❌ Not configured"
            }
            
            for api, status in apis_configured.items():
                st.write(f"{api}: {status}")
        
        with col2:
            st.subheader("🎛️ Platform Settings")
            
            auto_optimization = st.checkbox("Enable Auto-Optimization", value=True)
            notification_email = st.text_input("Notification Email", "boyculet1@gmail.com")
            optimization_frequency = st.selectbox("Optimization Frequency", 
                                                 ["Daily", "Weekly", "Bi-weekly", "Monthly"])
            
            if st.button("💾 Save Settings"):
                st.success("Settings saved successfully!")
        
        st.markdown("---")
        st.subheader("📋 System Information")
        st.info(f"""
        **Platform Version**: 2.0.0  
        **Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
        **Active Campaigns**: {len(self.campaigns)}  
        **Total Spend**: ${self.campaigns['spend'].sum():,.2f}
        """)

def main():
    # Check if running on Streamlit Cloud
    if 'STREAMLIT_SHARING' in os.environ:
        st.sidebar.success("🚀 Running on Streamlit Cloud")
    
    # Initialize and run the app
    app = StreamlitAIMarketing()
    app.render_dashboard()

if __name__ == "__main__":
    main()        if page == "📊 Dashboard Overview":
            self.render_ai_optimization()
        elif page == "📈 Performance Analytics":
        elif page == "🎯 Campaign Management":
            self.render_campaigns()
        elif page == "🤖 AI Optimization":


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
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    margin: 0.5rem 0;
}
.success-metric {
    background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
}
.warning-metric {
    background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
}
.danger-metric {
    background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'campaigns_data' not in st.session_state:
    st.session_state.campaigns_data = None
if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = None

class StreamlitAIMarketing:
    def __init__(self):
        self.initialize_demo_data()
    
    def initialize_demo_data(self):
        """Initialize demo data for the platform"""
        np.random.seed(42)
        
        # Campaign data
        self.campaigns = pd.DataFrame({
            'campaign_id': [f'CAMP_{i:03d}' for i in range(1, 21)],
            'campaign_name': [f'Campaign {i}' for i in range(1, 21)],
            'platform': np.random.choice(['Google Ads', 'Facebook', 'Instagram', 'LinkedIn', 'Twitter'], 20),
            'budget': np.random.uniform(1000, 10000, 20).round(2),
            'spend': np.random.uniform(500, 8000, 20).round(2),
            'impressions': np.random.randint(10000, 1000000, 20),
            'clicks': np.random.randint(100, 50000, 20),
            'conversions': np.random.randint(10, 2000, 20),
            'revenue': np.random.uniform(1000, 50000, 20).round(2),
            'status': np.random.choice(['Active', 'Paused', 'Draft'], 20, p=[0.7, 0.2, 0.1])
        })
        
        self.campaigns['ctr'] = (self.campaigns['clicks'] / self.campaigns['impressions'] * 100).round(2)

        self.campaigns['cpc'] = (self.campaigns['spend'] / self.campaigns['clicks']).round(2)
        self.campaigns['roas'] = (self.campaigns['revenue'] / self.campaigns['spend']).round(2)
        self.campaigns['conversion_rate'] = (self.campaigns['conversions'] / self.campaigns['clicks'] * 100).round(2)
        

        # Time series data
        dates = pd.date_range(start='2024-01-01', end='2024-09-24', freq='D')
        self.time_series = pd.DataFrame({

            'date': dates,
            'spend': np.random.uniform(500, 2000, len(dates)) + np.sin(np.arange(len(dates)) * 2 * np.pi / 30) * 200,
            'revenue': np.random.uniform(1000, 5000, len(dates)) + np.sin(np.arange(len(dates)) * 2 * np.pi / 30) * 500,
            'conversions': np.random.randint(10, 100, len(dates)),
            'impressions': np.random.randint(5000, 50000, len(dates))
        })
        
        self.time_series['roas'] = (self.time_series['revenue'] / self.time_series['spend']).round(2)

    def render_dashboard(self):
        st.title("🚀 AI Marketing Automation Platform")
        st.markdown("---")
        
        # Sidebar
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox("Choose a page", [
            "📊 Dashboard Overview",
            "🎯 Campaign Management", 
            "🤖 AI Optimization",
            "📈 Performance Analytics",
            "⚙️ Settings"
        ])
        
        if page == "📊 Dashboard Overview":
            self.render_overview()
        elif page == "🎯 Campaign Management":
            self.render_campaigns()
        elif page == "🤖 AI Optimization":
            self.render_ai_optimization()
        elif page == "📈 Performance Analytics":
            self.render_analytics()
        elif page == "⚙️ Settings":
            self.render_settings()

    def render_overview(self):
        st.header("📊 Dashboard Overview")
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_spend = self.campaigns['spend'].sum()
        total_revenue = self.campaigns['revenue'].sum()
        total_conversions = self.campaigns['conversions'].sum()
        avg_roas = (total_revenue / total_spend)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>💰 Total Spend</h3>
                <h2>${total_spend:,.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="success-metric">
                <h3>📈 Total Revenue</h3>
                <h2>${total_revenue:,.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="warning-metric">
                <h3>🎯 Conversions</h3>
                <h2>{total_conversions:,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            roas_color = "success-metric" if avg_roas > 3 else "warning-metric" if avg_roas > 2 else "danger-metric"
            st.markdown(f"""
            <div class="{roas_color}">
                <h3>⚡ Avg ROAS</h3>
                <h2>{avg_roas:.2f}x</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💸 Spend vs Revenue Over Time")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=self.time_series['date'], 
                y=self.time_series['spend'],
                name='Spend',
                line=dict(color='#ff6b6b')
            ))
            fig.add_trace(go.Scatter(
                x=self.time_series['date'], 
                y=self.time_series['revenue'],
                name='Revenue',
                line=dict(color='#4ecdc4')
            ))
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Campaign Performance by Platform")
            platform_perf = self.campaigns.groupby('platform').agg({
                'spend': 'sum',
                'revenue': 'sum',
                'conversions': 'sum'
            }).reset_index()
            platform_perf['roas'] = platform_perf['revenue'] / platform_perf['spend']
            
            fig = px.bar(platform_perf, x='platform', y='roas', 
                        title="ROAS by Platform",
                        color='roas',
                        color_continuous_scale='viridis')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    def render_campaigns(self):
        st.header("🎯 Campaign Management")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            platform_filter = st.selectbox("Filter by Platform", 
                                         ['All'] + list(self.campaigns['platform'].unique()))
        with col2:
            status_filter = st.selectbox("Filter by Status",
                                       ['All'] + list(self.campaigns['status'].unique()))
        with col3:
            min_roas = st.slider("Minimum ROAS", 0.0, 10.0, 0.0)
        
        # Apply filters
        filtered_campaigns = self.campaigns.copy()
        if platform_filter != 'All':
            filtered_campaigns = filtered_campaigns[filtered_campaigns['platform'] == platform_filter]
        if status_filter != 'All':
            filtered_campaigns = filtered_campaigns[filtered_campaigns['status'] == status_filter]
        filtered_campaigns = filtered_campaigns[filtered_campaigns['roas'] >= min_roas]
        
        # Campaign table
        st.subheader(f"📋 Campaigns ({len(filtered_campaigns)} found)")
        
        # Format the dataframe for display
        display_df = filtered_campaigns[['campaign_name', 'platform', 'status', 'budget', 
                                       'spend', 'revenue', 'conversions', 'ctr', 'roas']].copy()
        display_df['budget'] = display_df['budget'].apply(lambda x: f"${x:,.2f}")
        display_df['spend'] = display_df['spend'].apply(lambda x: f"${x:,.2f}")
        display_df['revenue'] = display_df['revenue'].apply(lambda x: f"${x:,.2f}")
        display_df['ctr'] = display_df['ctr'].apply(lambda x: f"{x}%")
        display_df['roas'] = display_df['roas'].apply(lambda x: f"{x:.2f}x")
        
        st.dataframe(display_df, use_container_width=True)
        
        # Campaign actions
        st.subheader("⚡ Quick Actions")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🚀 Optimize All Campaigns"):
                st.success("AI optimization started for all campaigns!")
        with col2:
            if st.button("⏸️ Pause Low Performers"):
                low_performers = len(filtered_campaigns[filtered_campaigns['roas'] < 2])
                st.warning(f"Paused {low_performers} campaigns with ROAS < 2.0")
        with col3:
            if st.button("📊 Generate Report"):
                st.info("Comprehensive report generated and saved!")

    def render_ai_optimization(self):
        st.header("🤖 AI Optimization Engine")
        
        st.markdown("""
        Our AI engine continuously analyzes your campaigns and provides intelligent recommendations
        to maximize your ROAS and reduce wasted spend.
        """)
        
        # Optimization options
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Optimization Settings")
            
            target_roas = st.slider("Target ROAS", 1.0, 10.0, 4.0, 0.1)
            max_budget = st.number_input("Maximum Daily Budget", 1000, 50000, 10000, 500)
            
            optimization_type = st.selectbox("Optimization Strategy", [
                "Maximize ROAS",
                "Maximize Conversions", 
                "Minimize Cost per Acquisition",
                "Balanced Growth"
            ])
            
            if st.button("🚀 Run AI Optimization"):
                with st.spinner("AI is analyzing your campaigns..."):
                    import time
                    time.sleep(3)  # Simulate processing
                    
                    st.session_state.optimization_results = {
                        'recommendations': [
                            {"campaign": "Campaign 1", "action": "Increase budget by 25%", "impact": "+15% ROAS"},
                            {"campaign": "Campaign 5", "action": "Pause keywords with CTR < 2%", "impact": "+8% efficiency"},
                            {"campaign": "Campaign 12", "action": "Shift budget to mobile devices", "impact": "+12% conversions"},
                            {"campaign": "Campaign 8", "action": "Update ad creative", "impact": "+20% CTR"},
                            {"campaign": "Campaign 3", "action": "Expand to similar audiences", "impact": "+30% reach"}
                        ],
                        'predicted_improvements': {
                            'roas_increase': 18.5,
                            'cost_savings': 2340.50,
                            'conversion_increase': 156
                        }
                    }
                    
                st.success("✅ AI Optimization Complete!")
        
        with col2:
            st.subheader("📊 Current Performance")
            
            # Performance metrics
            current_roas = self.campaigns['roas'].mean()
            st.metric("Current Avg ROAS", f"{current_roas:.2f}x", 
                     delta=f"{current_roas - 3:.2f}x vs target")
            
            underperforming = len(self.campaigns[self.campaigns['roas'] < target_roas])
            st.metric("Underperforming Campaigns", underperforming, 
                     delta=f"{underperforming}/{len(self.campaigns)} campaigns")
            
            total_waste = self.campaigns[self.campaigns['roas'] < 2]['spend'].sum()
            st.metric("Estimated Waste", f"${total_waste:,.2f}", 
                     delta=f"{(total_waste/self.campaigns['spend'].sum()*100):.1f}% of spend")
        
        # Display optimization results
        if st.session_state.optimization_results:
            st.markdown("---")
            st.subheader("🎯 AI Recommendations")
            
            for i, rec in enumerate(st.session_state.optimization_results['recommendations']):
                col1, col2, col3, col4 = st.columns([2, 3, 2, 1])
                with col1:
                    st.write(f"**{rec['campaign']}**")
                with col2:
                    st.write(rec['action'])
                with col3:
                    st.write(f"🎯 {rec['impact']}")
                with col4:
                    if st.button("Apply", key=f"apply_{i}"):
                        st.success(f"Applied to {rec['campaign']}")
            
            st.subheader("📈 Predicted Impact")
            col1, col2, col3 = st.columns(3)
            results = st.session_state.optimization_results['predicted_improvements']
            
            with col1:
                st.metric("ROAS Increase", f"+{results['roas_increase']}%")
            with col2:
                st.metric("Cost Savings", f"${results['cost_savings']:,.2f}")
            with col3:
                st.metric("Additional Conversions", f"+{results['conversion_increase']}")

    def render_analytics(self):
        st.header("📈 Performance Analytics")
        
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
        with col2:
            end_date = st.date_input("End Date", datetime.now())
        
        # Filter time series data
        mask = (self.time_series['date'] >= pd.Timestamp(start_date)) & (self.time_series['date'] <= pd.Timestamp(end_date))
        filtered_ts = self.time_series.loc[mask]
        
        # Key metrics over time
        st.subheader("📊 Performance Trends")
        
        metric_choice = st.selectbox("Select Metric", ["ROAS", "Spend", "Revenue", "Conversions"])
        
        fig = go.Figure()
        if metric_choice == "ROAS":
            fig.add_trace(go.Scatter(x=filtered_ts['date'], y=filtered_ts['roas'], 
                                   name='ROAS', line=dict(color='#4ecdc4')))
        elif metric_choice == "Spend":
            fig.add_trace(go.Scatter(x=filtered_ts['date'], y=filtered_ts['spend'], 
                                   name='Spend', line=dict(color='#ff6b6b')))
        elif metric_choice == "Revenue":
            fig.add_trace(go.Scatter(x=filtered_ts['date'], y=filtered_ts['revenue'], 
                                   name='Revenue', line=dict(color='#4ecdc4')))
        elif metric_choice == "Conversions":
            fig.add_trace(go.Scatter(x=filtered_ts['date'], y=filtered_ts['conversions'], 
                                   name='Conversions', line=dict(color='#45b7d1')))
        
        fig.update_layout(height=400, title=f"{metric_choice} Over Time")
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Top Performing Campaigns")
            top_campaigns = self.campaigns.nlargest(5, 'roas')[['campaign_name', 'platform', 'roas', 'revenue']]
            st.dataframe(top_campaigns, use_container_width=True)
        
        with col2:
            st.subheader("⚠️ Campaigns Needing Attention")
            bottom_campaigns = self.campaigns.nsmallest(5, 'roas')[['campaign_name', 'platform', 'roas', 'spend']]
            st.dataframe(bottom_campaigns, use_container_width=True)

    def render_settings(self):
        st.header("⚙️ Settings & Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔐 API Configuration")
            st.info("Configure your API keys in Streamlit Cloud secrets")
            
            apis_configured = {
                "OpenAI": "✅ Configured" if st.secrets.get("OPENAI_API_KEY", "") else "❌ Not configured",
                "Google Ads": "✅ Configured" if st.secrets.get("GOOGLE_ADS_API_KEY", "") else "❌ Not configured", 
                "Facebook": "✅ Configured" if st.secrets.get("FACEBOOK_API_KEY", "") else "❌ Not configured",
                "LinkedIn": "✅ Configured" if st.secrets.get("LINKEDIN_API_KEY", "") else "❌ Not configured"
            }
            
            for api, status in apis_configured.items():
                st.write(f"{api}: {status}")
        
        with col2:
            st.subheader("🎛️ Platform Settings")
            
            auto_optimization = st.checkbox("Enable Auto-Optimization", value=True)
            notification_email = st.text_input("Notification Email", "boyculet1@gmail.com")
            optimization_frequency = st.selectbox("Optimization Frequency", 
                                                 ["Daily", "Weekly", "Bi-weekly", "Monthly"])
            
            if st.button("💾 Save Settings"):
                st.success("Settings saved successfully!")
        
        st.markdown("---")
        st.subheader("📋 System Information")
        st.info(f"""
        **Platform Version**: 2.0.0  
        **Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
        **Active Campaigns**: {len(self.campaigns)}  
        **Total Spend**: ${self.campaigns['spend'].sum():,.2f}
        """)

def main():
    # Check if running on Streamlit Cloud
    if 'STREAMLIT_SHARING' in os.environ:
        st.sidebar.success("🚀 Running on Streamlit Cloud")
    
    # Initialize and run the app
    app = StreamlitAIMarketing()
    app.render_dashboard()

if __name__ == "__main__":
    main()


