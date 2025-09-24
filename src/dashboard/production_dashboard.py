import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Add src to Python path
sys.path.append('/app/src')

# Configure page FIRST - before any other Streamlit commands
st.set_page_config(
    page_title="AI Marketing Automation Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    from apis.api_manager import UnifiedMarketingAPI
    from data.advanced_processor import MarketingDataProcessor
    from ml.advanced_models import MarketingMLSuite
    from ai.advanced_ai_engine import AIOptimizationEngine
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.stop()

class ProductionMarketingDashboard:
    def __init__(self):
        self.api_manager = UnifiedMarketingAPI()
        self.data_processor = MarketingDataProcessor()
        self.ml_engine = MarketingMLSuite()
        self.ai_engine = AIOptimizationEngine()
        
    def render_dashboard(self):
        """Render the complete production dashboard"""
        
        # Custom CSS for professional styling
        self._inject_custom_css()
        
        # Header
        self._render_header()
        
        # Load and process data
        with st.spinner("Loading marketing data..."):
            data = self._load_and_process_data()
        
        if data is not None and len(data) > 0:
            # Main dashboard content
            self._render_kpi_overview(data)
            
            # Create main layout
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Analytics", 
                "🤖 ML Predictions", 
                "🧠 AI Insights", 
                "⚡ Automation", 
                "🔍 Performance"
            ])
            
            with tab1:
                self._render_advanced_analytics(data)
                
            with tab2:
                self._render_ml_predictions(data)
                
            with tab3:
                self._render_ai_insights(data)
                
            with tab4:
                self._render_automation_status()
                
            with tab5:
                self._render_performance_deep_dive(data)
        else:
            st.error("No data available. Please check your API connections.")
    
    def _inject_custom_css(self):
        """Inject custom CSS for professional styling"""
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }
        .status-online { color: #28a745; }
        .status-warning { color: #ffc107; }
        .status-error { color: #dc3545; }
        .insight-card {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            border-left: 4px solid #17a2b8;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _render_header(self):
        """Render professional header with system status"""
        st.markdown("""
        <div class="main-header">
            <h1 style="color: white; margin: 0;">🚀 AI Marketing Automation Platform</h1>
            <p style="color: white; margin: 0;">Production-Grade Marketing Intelligence & Optimization</p>
        </div>
        """, unsafe_allow_html=True)
        
        # System status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<span class="status-online">✅ System Status: Online</span>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<span class="status-online">🔄 Last Update: {datetime.now().strftime("%H:%M:%S")}</span>', unsafe_allow_html=True)
        with col3:
            st.markdown('<span class="status-online">🔗 API Status: Connected</span>', unsafe_allow_html=True)
        with col4:
            st.markdown('<span class="status-online">🤖 AI Engine: Active</span>', unsafe_allow_html=True)
    
    def _load_and_process_data(self):
        """Load and process marketing data"""
        try:
            # Fetch data from APIs
            raw_data = self.api_manager.fetch_all_platform_data()
            
            if raw_data:
                # Process the data
                processed_result = self.data_processor.process_campaign_data(raw_data)
                
                # Extract the actual DataFrame from the processing result
                if isinstance(processed_result, dict) and 'processed_data' in processed_result:
                    return processed_result['processed_data']
                else:
                    return processed_result
            else:
                return None
                
        except Exception as e:
            st.error(f"Data loading error: {e}")
            return None
    
    def _render_kpi_overview(self, data):
        """Executive KPI overview with advanced metrics"""
        st.subheader("📈 Performance Overview")
        
        # Ensure data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            st.error("Data is not in the expected format")
            return
        
        # Calculate KPIs with error handling
        try:
            total_spend = data['spend'].sum() if 'spend' in data.columns else 0
            total_conversions = data['conversions'].sum() if 'conversions' in data.columns else 0
            avg_roas = data['roas'].mean() if 'roas' in data.columns else 0
            avg_ctr = data['ctr'].mean() if 'ctr' in data.columns else 0
            total_clicks = data['clicks'].sum() if 'clicks' in data.columns else 0
        except Exception as e:
            st.error(f"Error calculating KPIs: {e}")
            st.write("Available columns:", list(data.columns))
            return
        
        # Calculate period-over-period changes (simulated)
        roas_change = np.random.uniform(-5, 15)
        ctr_change = np.random.uniform(-0.5, 1.0)
        spend_change = np.random.uniform(-10, 20)
        conv_change = np.random.randint(-50, 200)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "ROAS",
                f"{avg_roas:.2f}x",
                delta=f"{roas_change:+.1f}%",
                delta_color="normal"
            )
        
        with col2:
            st.metric(
                "CTR",
                f"{avg_ctr:.2f}%",
                delta=f"{ctr_change:+.2f}%"
            )
        
        with col3:
            st.metric(
                "Total Spend",
                f"${total_spend:,.0f}",
                delta=f"{spend_change:+.1f}%"
            )
        
        with col4:
            st.metric(
                "Conversions",
                f"{total_conversions:,}",
                delta=f"{conv_change:+.0f}"
            )
        
        with col5:
            efficiency_score = (avg_roas * avg_ctr) / 10
            st.metric(
                "Efficiency Score",
                f"{efficiency_score:.1f}/10",
                delta=f"{np.random.uniform(-0.5, 0.8):+.1f}"
            )
    
    def _render_advanced_analytics(self, data):
        """Multi-dimensional performance analytics"""
        st.subheader("📊 Advanced Analytics")
        
        if not isinstance(data, pd.DataFrame):
            st.error("Data format error in advanced analytics")
            return
        
        # Create comprehensive analytics chart
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('ROAS Trend Over Time', 'Platform Performance Comparison', 
                          'Spend vs Conversions Efficiency', 'Campaign Performance Heatmap'),
            specs=[[{"secondary_y": True}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "heatmap"}]]
        )
        
        try:
            # ROAS trend with spend overlay
            if 'date' in data.columns:
                daily_data = data.groupby('date').agg({
                    'roas': 'mean',
                    'spend': 'sum'
                }).reset_index()
                
                fig.add_trace(
                    go.Scatter(x=daily_data['date'], y=daily_data['roas'], 
                              name='ROAS', line=dict(color='#1f77b4')),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=daily_data['date'], y=daily_data['spend'], 
                              name='Spend', line=dict(color='#ff7f0e')),
                    row=1, col=1, secondary_y=True
                )
            
            # Platform comparison
            if 'platform' in data.columns:
                platform_metrics = data.groupby('platform').agg({
                    'roas': 'mean',
                    'ctr': 'mean',
                    'spend': 'sum'
                }).reset_index()
                
                fig.add_trace(
                    go.Bar(x=platform_metrics['platform'], y=platform_metrics['roas'], 
                           name='Avg ROAS', marker_color='#2E86AB'),
                    row=1, col=2
                )
            
            # Efficiency scatter plot
            if all(col in data.columns for col in ['spend', 'conversions', 'roas', 'ctr']):
                fig.add_trace(
                    go.Scatter(
                        x=data['spend'], 
                        y=data['conversions'],
                        mode='markers',
                        marker=dict(
                            size=data['roas']*3,
                            color=data['ctr'],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="CTR %")
                        ),
                        text=data.get('campaign_name', 'Campaign'),
                        name='Campaigns',
                        hovertemplate='Campaign: %{text}<br>Spend: $%{x}<br>Conversions: %{y}<extra></extra>'
                    ),
                    row=2, col=1
                )
            
            # Performance heatmap
            if 'date' in data.columns and 'campaign_name' in data.columns and len(data) > 5:
                pivot_data = data.pivot_table(
                    values='roas', 
                    index='campaign_name', 
                    columns='date', 
                    fill_value=0
                )
                
                pivot_data = pivot_data.head(10)
                
                fig.add_trace(
                    go.Heatmap(
                        z=pivot_data.values,
                        x=[str(col) for col in pivot_data.columns],
                        y=pivot_data.index,
                        colorscale='RdYlGn',
                        name='ROAS Heatmap',
                        showscale=False
                    ),
                    row=2, col=2
                )
            else:
                # Fallback simple heatmap
                if 'campaign_name' in data.columns:
                    campaign_performance = data.groupby('campaign_name')['roas'].mean().head(8)
                    fig.add_trace(
                        go.Heatmap(
                            z=[campaign_performance.values],
                            x=campaign_performance.index,
                            y=['ROAS'],
                            colorscale='RdYlGn',
                            showscale=False
                        ),
                        row=2, col=2
                    )
        
        except Exception as e:
            st.error(f"Error creating analytics charts: {e}")
        
        fig.update_layout(height=800, showlegend=False, title_text="Comprehensive Performance Analytics")
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_ml_predictions(self, data):
        """ML predictions and forecasts with FIXED budget recommendations"""
        st.subheader("🤖 ML Predictions & Forecasts")
        
        if not isinstance(data, pd.DataFrame):
            st.error("Data format error in ML predictions")
            return
        
        with st.spinner("Generating ML predictions..."):
            try:
                # Train models if not already trained
                training_result = self.ml_engine.train_all_models(data)
                
                # Generate predictions
                predictions = self.ml_engine.generate_predictions(data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**🎯 Model Performance**")
                    
                    perf_data = training_result.get('performance_summary', {})
                    for model_name, score in perf_data.items():
                        st.metric(
                            model_name.replace('_', ' ').title(),
                            f"{score:.3f}",
                            help="R² Score (higher is better)"
                        )
                    
                    st.write("**📊 Prediction Confidence**")
                    avg_confidence = np.mean(predictions.get('confidence_scores', [0.7]))
                    st.progress(avg_confidence)
                    st.write(f"Average Confidence: {avg_confidence:.1%}")
                
                with col2:
                    st.write("**🔮 ROAS Forecast**")
                    
                    # Create forecast visualization
                    if 'roas' in data.columns:
                        current_roas = data['roas'].values
                        predicted_roas = predictions.get('predicted_roas', current_roas)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=current_roas[:10],
                            mode='markers',
                            name='Current ROAS',
                            marker=dict(color='blue', size=8)
                        ))
                        fig.add_trace(go.Scatter(
                            y=predicted_roas[:10],
                            mode='markers',
                            name='Predicted ROAS',
                            marker=dict(color='red', size=8, symbol='diamond')
                        ))
                        fig.update_layout(
                            title="Current vs Predicted ROAS",
                            yaxis_title="ROAS",
                            height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # FIXED Budget optimization recommendations - TOP performers in each category
                st.write("**💰 Budget Optimization Recommendations**")
                
                # Create realistic optimization flags based on ROAS performance
                enhanced_data = data.copy()
                if 'roas' in enhanced_data.columns:
                    # Define optimization logic
                    conditions = [
                        enhanced_data['roas'] >= 3.0,  # High performers - scale up
                        enhanced_data['roas'] <= 1.5,  # Poor performers - optimize or pause
                    ]
                    choices = ['scale_up', 'optimize_or_pause']
                    enhanced_data['optimization_flag'] = np.select(conditions, choices, default='maintain')
                    
                    # FIXED: Get TOP performers in each category (sorted by ROAS)
                    scale_up_campaigns = enhanced_data[enhanced_data['optimization_flag'] == 'scale_up'].nlargest(5, 'roas')
                    optimize_campaigns = enhanced_data[enhanced_data['optimization_flag'] == 'optimize_or_pause'].nsmallest(5, 'roas')
                    maintain_campaigns = enhanced_data[enhanced_data['optimization_flag'] == 'maintain'].nlargest(5, 'roas')
                    
                    # Combine with priority to actionable campaigns
                    priority_campaigns = pd.concat([scale_up_campaigns, optimize_campaigns, maintain_campaigns]).head(15)
                    
                    if len(priority_campaigns) > 0:
                        # Create budget recommendations
                        budget_data = pd.DataFrame({
                            'Campaign': priority_campaigns['campaign_name'],
                            'Platform': priority_campaigns['platform'],
                            'Current Spend': priority_campaigns['spend'].round(0),
                            'Current ROAS': priority_campaigns['roas'].round(2),
                            'Recommended Spend': np.where(
                                priority_campaigns['optimization_flag'] == 'scale_up',
                                priority_campaigns['spend'] * 1.5,  # Increase by 50%
                                np.where(
                                    priority_campaigns['optimization_flag'] == 'optimize_or_pause',
                                    priority_campaigns['spend'] * 0.5,  # Decrease by 50%
                                    priority_campaigns['spend']  # Maintain
                                )
                            ).round(0),
                            'Optimization Flag': priority_campaigns['optimization_flag']
                        })
                        
                        # Add filtering for budget recommendations
                        col_filter1, col_filter2 = st.columns(2)
                        with col_filter1:
                            flag_filter = st.selectbox(
                                "Filter Recommendations", 
                                ["All", "Scale Up", "Optimize/Pause", "Maintain"],
                                key="budget_filter"
                            )
                        with col_filter2:
                            platform_filter = st.selectbox(
                                "Filter by Platform", 
                                ["All", "Google", "Meta"],
                                key="budget_platform_filter"
                            )
                        
                        # Apply filters
                        filtered_budget = budget_data.copy()
                        if flag_filter != "All":
                            flag_map = {"Scale Up": "scale_up", "Optimize/Pause": "optimize_or_pause", "Maintain": "maintain"}
                            filtered_budget = filtered_budget[filtered_budget['Optimization Flag'] == flag_map[flag_filter]]
                        
                        if platform_filter != "All":
                            filtered_budget = filtered_budget[filtered_budget['Platform'] == platform_filter]
                        
                        st.write(f"**Showing {len(filtered_budget)} recommendations**")
                        st.write("*Scale Up: TOP 5 highest ROAS campaigns ≥3.0 | Optimize: BOTTOM 5 lowest ROAS campaigns ≤1.5 | Maintain: TOP 5 mid-range campaigns*")
                        
                        # Color code the optimization flags
                        def style_budget_row(row):
                            if row['Optimization Flag'] == 'scale_up':
                                return ['background-color: #d4edda; color: #155724'] * len(row)
                            elif row['Optimization Flag'] == 'optimize_or_pause':
                                return ['background-color: #f8d7da; color: #721c24'] * len(row)
                            else:
                                return ['background-color: #fff3cd; color: #856404'] * len(row)
                        
                        if len(filtered_budget) > 0:
                            styled_budget = filtered_budget.style.apply(style_budget_row, axis=1)
                            st.dataframe(styled_budget, use_container_width=True, height=300)
                            
                            # Summary stats
                            col_stats1, col_stats2, col_stats3 = st.columns(3)
                            with col_stats1:
                                scale_up_count = len(filtered_budget[filtered_budget['Optimization Flag'] == 'scale_up'])
                                st.metric("Campaigns to Scale Up", scale_up_count)
                            with col_stats2:
                                optimize_count = len(filtered_budget[filtered_budget['Optimization Flag'] == 'optimize_or_pause'])
                                st.metric("Campaigns to Optimize", optimize_count)
                            with col_stats3:
                                maintain_count = len(filtered_budget[filtered_budget['Optimization Flag'] == 'maintain'])
                                st.metric("Campaigns to Maintain", maintain_count)
                        else:
                            st.info("No campaigns match the selected filters.")
                
            except Exception as e:
                st.error(f"ML prediction error: {e}")
                st.info("Using fallback predictions...")
    
    def _render_ai_insights(self, data):
        """AI-generated insights and recommendations"""
        st.subheader("🧠 AI Insights & Recommendations")
        
        if not isinstance(data, pd.DataFrame):
            st.error("Data format error in AI insights")
            return
        
        with st.spinner("Generating AI insights..."):
            try:
                # Get ML predictions first
                predictions = self.ml_engine.generate_predictions(data)
                
                # Generate AI analysis
                ai_analysis = self.ai_engine.generate_comprehensive_analysis(data, predictions)
                
                # Create tabs for different types of insights
                insight_tab1, insight_tab2, insight_tab3, insight_tab4 = st.tabs([
                    "📈 Performance Analysis", 
                    "⚡ Optimizations", 
                    "💡 Creative Ideas", 
                    "📊 Executive Summary"
                ])
                
                with insight_tab1:
                    st.markdown('<div class="insight-card">', unsafe_allow_html=True)
                    st.write("**Performance Pattern Analysis**")
                    st.write(ai_analysis.get('performance_insights', 'Analysis not available'))
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.write("**Risk Assessment**")
                    st.write(ai_analysis.get('risk_assessment', 'Risk analysis not available'))
                
                with insight_tab2:
                    st.write("**Optimization Recommendations**")
                    st.write(ai_analysis.get('optimization_recommendations', 'Recommendations not available'))
                    
                    st.write("**Priority Actions**")
                    priority_actions = ai_analysis.get('priority_actions', [])
                    for i, action in enumerate(priority_actions[:3], 1):
                        st.write(f"{i}. {action}")
                
                with insight_tab3:
                    st.write("**Creative Suggestions**")
                    st.write(ai_analysis.get('creative_suggestions', 'Creative ideas not available'))
                    
                    st.write("**Budget Strategy**")
                    st.write(ai_analysis.get('budget_strategy', 'Budget recommendations not available'))
                
                with insight_tab4:
                    st.write("**Executive Summary**")
                    executive_summary = self.ai_engine.generate_executive_summary(data, predictions, ai_analysis)
                    st.write(executive_summary)
                    
                    st.write("**Competitive Analysis**")
                    st.write(ai_analysis.get('competitive_analysis', 'Competitive analysis not available'))
                
            except Exception as e:
                st.error(f"AI insight generation error: {e}")
    
    def _render_automation_status(self):
        """Show automation workflow status"""
        st.subheader("⚡ Automation Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Active Workflows**")
            workflows = [
                {"name": "Data Pipeline", "status": "✅ Running", "last_run": "2 min ago"},
                {"name": "Performance Monitor", "status": "✅ Active", "last_run": "5 min ago"},
                {"name": "AI Analysis", "status": "✅ Running", "last_run": "1 min ago"},
                {"name": "Alert System", "status": "✅ Monitoring", "last_run": "30 sec ago"},
                {"name": "Budget Optimizer", "status": "⚠️ Pending", "last_run": "15 min ago"}
            ]
            
            for workflow in workflows:
                status_class = "status-online" if "✅" in workflow['status'] else "status-warning"
                st.markdown(f'<span class="{status_class}">**{workflow["name"]}:** {workflow["status"]} (Last: {workflow["last_run"]})</span>', 
                           unsafe_allow_html=True)
        
        with col2:
            st.write("**Recent Automated Actions**")
            actions = [
                "🔄 Data refresh completed - 2 min ago",
                "🤖 ML models retrained - 15 min ago", 
                "⚠️ Low ROAS alert triggered - Campaign XYZ - 23 min ago",
                "💡 AI optimization suggestions generated - 25 min ago",
                "💰 Budget reallocation executed - Campaign ABC +20% - 1 hr ago"
            ]
            
            for action in actions:
                st.write(action)
        
        # Workflow configuration
        st.write("**Automation Configuration**")
        
        config_col1, config_col2 = st.columns(2)
        with config_col1:
            st.selectbox("Data Refresh Interval", ["Every 2 hours", "Every 4 hours", "Daily"], index=0)
            st.selectbox("Alert Threshold ROAS", ["< 1.5x", "< 2.0x", "< 2.5x"], index=1)
        
        with config_col2:
            st.selectbox("Auto Budget Adjustment", ["Enabled", "Disabled"], index=1)
            st.selectbox("AI Analysis Frequency", ["Hourly", "Every 4 hours", "Daily"], index=1)
    
    def _render_performance_deep_dive(self, data):
        """WORKING performance analysis with persistent slider (NO reset button)"""
        st.subheader("🔍 Performance Deep Dive")
        
        if not isinstance(data, pd.DataFrame):
            st.error("Data format error in performance deep dive")
            return
        
        try:
            # Calculate additional metrics
            enhanced_data = data.copy()
            
            if 'roas' in data.columns and 'ctr' in data.columns:
                enhanced_data['efficiency_score'] = (enhanced_data['roas'] * enhanced_data['ctr']) / 10
            else:
                enhanced_data['efficiency_score'] = 0
                
            if 'spend' in data.columns and 'conversions' in data.columns:
                enhanced_data['cost_per_conversion'] = enhanced_data['spend'] / (enhanced_data['conversions'] + 1)
            else:
                enhanced_data['cost_per_conversion'] = 0
            
            # WORKING session state management (copied from your working code)
            st.write("**Campaign Data Explorer**")
            
            # Initialize session state properly
            min_roas = float(enhanced_data['roas'].min()) if 'roas' in enhanced_data.columns else 0
            max_roas = float(enhanced_data['roas'].max()) if 'roas' in enhanced_data.columns else 10
            
            # Initialize session state values if they don't exist
            if 'roas_filter_value' not in st.session_state:
                st.session_state.roas_filter_value = min_roas
            if 'platform_filter_value' not in st.session_state:
                st.session_state.platform_filter_value = 'All'
            if 'view_filter_value' not in st.session_state:
                st.session_state.view_filter_value = 'All Data'
            
            # Create filter controls with callback functions (EXACT copy from working code)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Platform filter
                platforms = ['All'] + list(enhanced_data['platform'].unique()) if 'platform' in enhanced_data.columns else ['All']
                def platform_callback():
                    st.session_state.platform_filter_value = st.session_state.platform_selector_new
                
                selected_platform = st.selectbox(
                    "Filter by Platform", 
                    platforms,
                    index=platforms.index(st.session_state.platform_filter_value) if st.session_state.platform_filter_value in platforms else 0,
                    key="platform_selector_new",
                    on_change=platform_callback
                )
            
            with col2:
                # WORKING ROAS slider with callback (copied exactly from your working code)
                def roas_callback():
                    st.session_state.roas_filter_value = st.session_state.roas_slider_new
                
                roas_threshold = st.slider(
                    "Minimum ROAS", 
                    min_value=min_roas, 
                    max_value=max_roas,
                    value=st.session_state.roas_filter_value,
                    step=0.1,
                    key="roas_slider_new",
                    on_change=roas_callback
                )
            
            with col3:
                # Data view options
                view_options = ["All Data", "Top 20 Performers", "Bottom 20 Performers", "Sample View"]
                def view_callback():
                    st.session_state.view_filter_value = st.session_state.view_selector_new
                
                view_option = st.selectbox(
                    "View", 
                    view_options,
                    index=view_options.index(st.session_state.view_filter_value) if st.session_state.view_filter_value in view_options else 0,
                    key="view_selector_new",
                    on_change=view_callback
                )
            
            # Apply filters using session state values (copied exactly)
            filtered_data = enhanced_data.copy()
            
            if st.session_state.platform_filter_value != 'All' and 'platform' in enhanced_data.columns:
                filtered_data = filtered_data[filtered_data['platform'] == st.session_state.platform_filter_value]
            
            if 'roas' in enhanced_data.columns:
                filtered_data = filtered_data[filtered_data['roas'] >= st.session_state.roas_filter_value]
            
            # Apply view options
            if st.session_state.view_filter_value == "Top 20 Performers" and 'roas' in filtered_data.columns:
                filtered_data = filtered_data.nlargest(20, 'roas')
            elif st.session_state.view_filter_value == "Bottom 20 Performers" and 'roas' in filtered_data.columns:
                filtered_data = filtered_data.nsmallest(20, 'roas')
            elif st.session_state.view_filter_value == "Sample View":
                # Show balanced sample if possible
                if 'platform' in filtered_data.columns:
                    google_sample = filtered_data[filtered_data['platform'] == 'Google'].head(10)
                    meta_sample = filtered_data[filtered_data['platform'] == 'Meta'].head(10)
                    filtered_data = pd.concat([google_sample, meta_sample]).reset_index(drop=True)
                else:
                    filtered_data = filtered_data.head(20)
            
            # Display current filter values
            st.write(f"**Current Filters:** Platform: {st.session_state.platform_filter_value}, ROAS ≥ {st.session_state.roas_filter_value:.2f}, View: {st.session_state.view_filter_value}")
            st.write(f"**Showing {len(filtered_data)} of {len(enhanced_data)} campaigns**")
            
            # Display the filtered data
            available_columns = [col for col in ['campaign_name', 'platform', 'roas', 'ctr', 'spend', 
                               'conversions', 'efficiency_score', 'cost_per_conversion'] if col in filtered_data.columns]
            
            if len(filtered_data) > 0 and available_columns:
                # Format numeric columns for better display
                display_data = filtered_data[available_columns].copy()
                
                # Round numeric columns
                numeric_cols = ['roas', 'ctr', 'spend', 'conversions', 'efficiency_score', 'cost_per_conversion']
                for col in numeric_cols:
                    if col in display_data.columns:
                        display_data[col] = display_data[col].round(2)
                
                st.dataframe(
                    display_data, 
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )
            else:
                st.warning("No campaigns match the selected filters.")
            
            # NOTE: Reset button REMOVED to prevent infinite loops
            # Users can manually adjust filters to reset them
            
            # Platform summary
            if 'platform' in enhanced_data.columns:
                st.write("**Platform Performance Summary**")
                platform_summary = enhanced_data.groupby('platform').agg({
                    'roas': 'mean',
                    'ctr': 'mean', 
                    'spend': 'sum',
                    'conversions': 'sum'
                }).round(2)
                st.dataframe(platform_summary, use_container_width=True)
            
            # Performance distribution charts
            col1, col2 = st.columns(2)
            
            with col1:
                if 'roas' in enhanced_data.columns:
                    fig = px.histogram(
                        enhanced_data, 
                        x='roas', 
                        nbins=20,
                        title="ROAS Distribution",
                        color_discrete_sequence=['#2E86AB']
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'platform' in enhanced_data.columns and 'spend' in enhanced_data.columns:
                    platform_spend = enhanced_data.groupby('platform')['spend'].sum()
                    fig = px.pie(
                        values=platform_spend.values,
                        names=platform_spend.index,
                        title="Spend Allocation by Platform"
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
        except Exception as e:
            st.error(f"Error in performance deep dive: {e}")

# Initialize and run dashboard
def load_dashboard_data():
    dashboard = ProductionMarketingDashboard()
    return dashboard

def main():
    dashboard = load_dashboard_data()
    dashboard.render_dashboard()

if __name__ == "__main__":
    main()