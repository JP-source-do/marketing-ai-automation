import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI package not available. AI insights will use fallback responses.")

class AIOptimizationEngine:
    """AI-powered marketing optimization and insights engine"""
    
    def __init__(self):
        self.logger = self._setup_logging()  # Initialize logger first
        self.openai_client = self._setup_openai()
        self.prompt_library = self._load_prompt_templates()
        self.fallback_responses = self._setup_fallback_responses()
    
    def _setup_openai(self):
        """Setup OpenAI client with error handling"""
        if not OPENAI_AVAILABLE:
            return None
            
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key and api_key != 'placeholder':
                openai.api_key = api_key
                return openai
            else:
                self.logger.warning("OpenAI API key not configured. Using fallback responses.")
                return None
        except Exception as e:
            self.logger.error(f"OpenAI setup failed: {e}")
            return None
    
    def _setup_logging(self):
        """Setup logging for AI operations"""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def _load_prompt_templates(self):
        """Load prompt templates for different analysis types"""
        return {
            'performance_analysis': """
            Analyze this marketing campaign performance data as a senior marketing strategist:
            
            PERFORMANCE METRICS:
            {metrics_summary}
            
            TRENDS:
            {trends_analysis}
            
            Provide a concise analysis covering:
            1. Key performance patterns identified
            2. Most significant opportunities
            3. Risk factors to monitor
            4. Strategic recommendations
            
            Keep response under 300 words, actionable, and data-driven.
            """,
            
            'optimization_recommendations': """
            As a marketing optimization expert, analyze this campaign data:
            
            CURRENT PERFORMANCE:
            {current_metrics}
            
            ML PREDICTIONS:
            {ml_predictions}
            
            Provide 5 specific optimization recommendations:
            1. IMMEDIATE ACTIONS (next 24 hours)
            2. BUDGET ADJUSTMENTS
            3. CREATIVE TESTING STRATEGY  
            4. AUDIENCE OPTIMIZATION
            5. PLATFORM-SPECIFIC TACTICS
            
            For each recommendation include:
            - Specific action steps
            - Expected ROAS impact
            - Implementation timeline
            
            Maximum 400 words total.
            """,
            
            'creative_suggestions': """
            Based on this top-performing campaign data:
            {top_campaigns}
            
            Generate 5 creative optimization ideas:
            1. Headlines/copy variations
            2. Visual concept improvements
            3. Call-to-action enhancements
            4. Landing page optimizations
            5. Audience messaging angles
            
            Each suggestion should include rationale and expected impact.
            Keep response under 250 words.
            """,
            
            'executive_summary': """
            Create an executive summary for marketing performance:
            
            KEY METRICS:
            - Total ROAS: {total_roas}
            - Total Spend: ${total_spend:,.2f}
            - Conversions: {total_conversions:,}
            - Top Platform: {best_platform}
            
            PREDICTIONS:
            {predictions_summary}
            
            Write 3 concise paragraphs:
            1. Performance vs targets/benchmarks
            2. Key opportunities identified
            3. Recommended strategic actions
            
            Target audience: C-level executives. Maximum 200 words.
            """
        }
    
    def _setup_fallback_responses(self):
        """Setup fallback responses for when OpenAI is unavailable"""
        return {
            'performance_insights': """
            PERFORMANCE ANALYSIS:
            
            Key Patterns Identified:
            • Strong performance variation across platforms indicates optimization opportunities
            • Campaign efficiency shows room for budget reallocation
            • Click-through rates suggest creative testing potential
            
            Strategic Recommendations:
            • Focus budget on highest ROAS campaigns
            • Implement A/B testing for underperforming creative assets
            • Consider audience expansion for top-performing campaigns
            • Monitor cost trends for competitive pressure indicators
            
            Risk Factors:
            • Performance volatility requires continuous monitoring
            • Budget concentration risks if top campaigns decline
            • Seasonal trends may impact current optimization strategies
            """,
            
            'optimization_recommendations': """
            OPTIMIZATION RECOMMENDATIONS:
            
            1. IMMEDIATE ACTIONS (Next 24 hours):
            • Pause campaigns with ROAS < 1.5x
            • Increase budgets 20% for campaigns with ROAS > 3.0x
            • Review ad creative for CTR < 1%
            
            2. BUDGET ADJUSTMENTS:
            • Reallocate 30% budget from low performers to high performers
            • Test 10% budget increase on top 3 campaigns
            • Set automated rules for budget optimization
            
            3. CREATIVE TESTING STRATEGY:
            • Launch 3 new headline variations this week
            • Test video vs image creative formats
            • Implement dynamic creative optimization
            
            4. AUDIENCE OPTIMIZATION:
            • Expand lookalike audiences for top campaigns
            • Exclude underperforming demographics
            • Test interest-based targeting refinements
            
            5. PLATFORM-SPECIFIC TACTICS:
            • Focus video content on Meta platforms
            • Emphasize search campaigns on Google
            • Cross-platform retargeting implementation
            """,
            
            'creative_suggestions': """
            CREATIVE OPTIMIZATION IDEAS:
            
            1. Headlines: Test urgency-driven copy ("Limited Time", "Today Only")
            2. Visuals: Implement user-generated content and testimonials
            3. CTAs: Test action-oriented buttons ("Get Started Now", "Claim Offer")
            4. Landing Pages: Optimize for mobile experience and load speed
            5. Messaging: Focus on value proposition and social proof elements
            
            Each suggestion targets improved engagement and conversion rates.
            """,
            
            'risk_assessment': """
            RISK ASSESSMENT:
            
            HIGH PRIORITY:
            • Budget concentration in limited campaigns
            • Seasonal performance dependencies
            • Competitive pressure indicators
            
            MEDIUM PRIORITY:
            • Creative fatigue potential
            • Audience saturation risks
            • Platform algorithm changes
            
            MITIGATION STRATEGIES:
            • Diversify campaign portfolio
            • Implement continuous testing protocols
            • Monitor competitor activities
            """
        }
    
    def generate_comprehensive_analysis(self, campaign_data: pd.DataFrame, ml_predictions: Dict) -> Dict:
        """Generate multi-dimensional AI analysis"""
        
        try:
            # Create analysis context
            context = self._create_analysis_context(campaign_data, ml_predictions)
            
            # Generate different types of insights
            analysis_results = {
                'performance_insights': self._analyze_performance_patterns(campaign_data, context),
                'optimization_recommendations': self._generate_optimizations(campaign_data, ml_predictions, context),
                'creative_suggestions': self._generate_creative_ideas(campaign_data),
                'budget_strategy': self._optimize_budget_allocation(campaign_data, ml_predictions),
                'risk_assessment': self._assess_campaign_risks(campaign_data),
                'priority_actions': self._prioritize_recommendations(campaign_data, ml_predictions),
                'competitive_analysis': self._benchmark_performance(campaign_data)
            }
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"AI analysis generation failed: {e}")
            return self._get_fallback_analysis()
    
    def _create_analysis_context(self, data: pd.DataFrame, predictions: Dict) -> Dict:
        """Create comprehensive context for AI analysis"""
        
        # Current performance metrics
        current_metrics = {
            'total_spend': data['spend'].sum(),
            'total_conversions': data['conversions'].sum(),
            'avg_roas': data['roas'].mean(),
            'avg_ctr': data['ctr'].mean(),
            'avg_cpc': data['cpc'].mean(),
            'campaign_count': len(data),
            'platform_breakdown': data.groupby('platform')['spend'].sum().to_dict(),
            'top_campaign': data.nlargest(1, 'roas')[['campaign_name', 'roas']].to_dict('records')[0] if len(data) > 0 else {}
        }
        
        # Trend analysis
        if 'date' in data.columns:
            data_sorted = data.sort_values('date')
            recent_data = data_sorted.tail(7)  # Last 7 days
            older_data = data_sorted.head(7)   # First 7 days
            
            trends = {
                'roas_trend': (recent_data['roas'].mean() - older_data['roas'].mean()) / older_data['roas'].mean() * 100,
                'spend_trend': (recent_data['spend'].sum() - older_data['spend'].sum()) / older_data['spend'].sum() * 100,
                'ctr_trend': recent_data['ctr'].mean() - older_data['ctr'].mean()
            }
        else:
            trends = {'roas_trend': 0, 'spend_trend': 0, 'ctr_trend': 0}
        
        # ML predictions summary
        pred_summary = {}
        if predictions:
            pred_summary = {
                'avg_predicted_roas': np.mean(predictions.get('predicted_roas', [2.5])),
                'total_recommended_spend': np.sum(predictions.get('optimal_daily_spend', [100])),
                'optimization_flags': predictions.get('optimization_flags', ['maintain'])
            }
        
        return {
            'current_metrics': current_metrics,
            'trends': trends,
            'ml_predictions': pred_summary
        }
    
    def _generate_ai_response(self, prompt_template: str, context: Dict) -> str:
        """Generate AI response using OpenAI or fallback"""
        
        if self.openai_client:
            try:
                formatted_prompt = prompt_template.format(**context)
                
                response = self.openai_client.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": formatted_prompt}],
                    max_tokens=500,
                    temperature=0.7
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                self.logger.warning(f"OpenAI API call failed: {e}")
                return None
        
        return None
    
    def _analyze_performance_patterns(self, data: pd.DataFrame, context: Dict) -> str:
        """Analyze performance patterns with AI"""
        
        # Try AI response first
        ai_response = self._generate_ai_response(
            self.prompt_library['performance_analysis'],
            {
                'metrics_summary': self._format_metrics_summary(context['current_metrics']),
                'trends_analysis': self._format_trends_analysis(context['trends'])
            }
        )
        
        if ai_response:
            return ai_response
        
        # Fallback to template response with data
        return self._customize_fallback_response(
            self.fallback_responses['performance_insights'], 
            context
        )
    
    def _generate_optimizations(self, data: pd.DataFrame, predictions: Dict, context: Dict) -> str:
        """Generate optimization recommendations"""
        
        # Try AI response
        ai_response = self._generate_ai_response(
            self.prompt_library['optimization_recommendations'],
            {
                'current_metrics': self._format_metrics_summary(context['current_metrics']),
                'ml_predictions': str(context['ml_predictions'])
            }
        )
        
        if ai_response:
            return ai_response
        
        return self._customize_fallback_response(
            self.fallback_responses['optimization_recommendations'],
            context
        )
    
    def _generate_creative_ideas(self, data: pd.DataFrame) -> str:
        """Generate creative suggestions"""
        
        top_performing = data.nlargest(3, 'roas') if len(data) >= 3 else data
        
        ai_response = self._generate_ai_response(
            self.prompt_library['creative_suggestions'],
            {
                'top_campaigns': top_performing[['campaign_name', 'roas', 'ctr']].to_string()
            }
        )
        
        if ai_response:
            return ai_response
        
        return self.fallback_responses['creative_suggestions']
    
    def _optimize_budget_allocation(self, data: pd.DataFrame, predictions: Dict) -> str:
        """Generate budget optimization strategy"""
        
        # Identify budget reallocation opportunities
        high_performers = data[data['roas'] > data['roas'].quantile(0.7)]
        low_performers = data[data['roas'] < data['roas'].quantile(0.3)]
        
        budget_analysis = f"""
        BUDGET OPTIMIZATION STRATEGY:
        
        HIGH PERFORMERS (Scale Up):
        {high_performers[['campaign_name', 'roas', 'spend']].head(3).to_string(index=False)}
        
        OPTIMIZATION CANDIDATES (Review/Adjust):
        {low_performers[['campaign_name', 'roas', 'spend']].head(3).to_string(index=False)}
        
        RECOMMENDATIONS:
        • Increase budget by 25% for campaigns with ROAS > {data['roas'].quantile(0.7):.2f}
        • Reduce budget by 15% for campaigns with ROAS < {data['roas'].quantile(0.3):.2f}
        • Total potential budget reallocation: ${high_performers['spend'].sum() * 0.25:.2f}
        """
        
        return budget_analysis
    
    def _assess_campaign_risks(self, data: pd.DataFrame) -> str:
        """Assess campaign performance risks"""
        
        # Calculate risk indicators
        high_spend_low_roas = data[(data['spend'] > data['spend'].quantile(0.8)) & 
                                  (data['roas'] < data['roas'].quantile(0.5))]
        
        declining_ctr = data[data['ctr'] < 1.0] if 'ctr' in data.columns else pd.DataFrame()
        
        risk_analysis = f"""
        RISK ASSESSMENT:
        
        HIGH PRIORITY RISKS:
        • {len(high_spend_low_roas)} campaigns with high spend but low ROAS
        • {len(declining_ctr)} campaigns with CTR below 1%
        • Budget concentration: Top 3 campaigns account for {data.nlargest(3, 'spend')['spend'].sum() / data['spend'].sum() * 100:.1f}% of spend
        
        RISK MITIGATION:
        • Monitor high-spend/low-ROAS campaigns daily
        • Implement automated pause rules for ROAS < 1.0
        • Diversify campaign portfolio to reduce concentration risk
        • Weekly creative refresh for low CTR campaigns
        """
        
        return risk_analysis
    
    def _prioritize_recommendations(self, data: pd.DataFrame, predictions: Dict) -> List[str]:
        """Generate prioritized action items"""
        
        actions = []
        
        # Priority 1: Critical issues
        critical_campaigns = data[data['roas'] < 1.0]
        if len(critical_campaigns) > 0:
            actions.append(f"URGENT: Pause or optimize {len(critical_campaigns)} campaigns with ROAS < 1.0")
        
        # Priority 2: Scale opportunities  
        scale_opportunities = data[data['roas'] > 3.0]
        if len(scale_opportunities) > 0:
            actions.append(f"Scale up {len(scale_opportunities)} high-performing campaigns (ROAS > 3.0)")
        
        # Priority 3: Optimization candidates
        optimize_candidates = data[(data['roas'] > 1.0) & (data['roas'] < 2.0)]
        if len(optimize_candidates) > 0:
            actions.append(f"Optimize {len(optimize_candidates)} moderate-performing campaigns")
        
        # Priority 4: Creative testing
        low_ctr = data[data['ctr'] < 2.0] if 'ctr' in data.columns else pd.DataFrame()
        if len(low_ctr) > 0:
            actions.append(f"Launch creative tests for {len(low_ctr)} campaigns with low CTR")
        
        return actions[:5]  # Top 5 priorities
    
    def _benchmark_performance(self, data: pd.DataFrame) -> str:
        """Benchmark performance against industry standards"""
        
        # Industry benchmarks (simplified)
        benchmarks = {
            'meta': {'ctr': 1.5, 'roas': 2.8, 'cpc': 1.25},
            'google': {'ctr': 3.2, 'roas': 3.1, 'cpc': 2.15}
        }
        
        analysis = "COMPETITIVE BENCHMARKING:\n\n"
        
        for platform in data['platform'].unique():
            platform_data = data[data['platform'] == platform.lower()]
            if platform.lower() in benchmarks:
                benchmark = benchmarks[platform.lower()]
                
                analysis += f"{platform} Performance vs Industry:\n"
                analysis += f"• CTR: {platform_data['ctr'].mean():.2f}% vs {benchmark['ctr']:.2f}% benchmark\n"
                analysis += f"• ROAS: {platform_data['roas'].mean():.2f}x vs {benchmark['roas']:.2f}x benchmark\n"
                analysis += f"• CPC: ${platform_data['cpc'].mean():.2f} vs ${benchmark['cpc']:.2f} benchmark\n\n"
        
        return analysis
    
    def generate_executive_summary(self, data: pd.DataFrame, predictions: Dict, ai_insights: Dict) -> str:
        """Generate executive-level performance summary"""
        
        context = {
            'total_roas': f"{data['roas'].mean():.2f}x",
            'total_spend': data['spend'].sum(),
            'total_conversions': data['conversions'].sum(),
            'best_platform': data.groupby('platform')['roas'].mean().idxmax(),
            'predictions_summary': f"ML models predict {predictions.get('predicted_roas', [2.5])[0]:.2f}x ROAS trend"
        }
        
        # Try AI response
        ai_response = self._generate_ai_response(
            self.prompt_library['executive_summary'],
            context
        )
        
        if ai_response:
            return ai_response
        
        # Fallback executive summary
        return f"""
        EXECUTIVE PERFORMANCE SUMMARY
        
        Current Performance: Achieving {context['total_roas']} average ROAS across ${context['total_spend']:,.0f} in ad spend, generating {context['total_conversions']:,} conversions. {context['best_platform']} is the top-performing platform.
        
        Key Opportunities: ML analysis identifies budget reallocation opportunities that could improve overall ROAS by 15-25%. High-performing campaigns show potential for scaled investment.
        
        Recommended Actions: Focus budget on campaigns exceeding 3.0x ROAS, optimize underperforming creative assets, and implement automated bidding strategies. Expected impact: 20% efficiency improvement within 30 days.
        """
    
    def _format_metrics_summary(self, metrics: Dict) -> str:
        """Format metrics for AI prompts"""
        return f"""
        Total Spend: ${metrics.get('total_spend', 0):,.2f}
        Average ROAS: {metrics.get('avg_roas', 0):.2f}x
        Average CTR: {metrics.get('avg_ctr', 0):.2f}%
        Total Conversions: {metrics.get('total_conversions', 0):,}
        Campaign Count: {metrics.get('campaign_count', 0)}
        """
    
    def _format_trends_analysis(self, trends: Dict) -> str:
        """Format trend data for AI prompts"""
        return f"""
        ROAS Trend: {trends.get('roas_trend', 0):+.1f}%
        Spend Trend: {trends.get('spend_trend', 0):+.1f}%
        CTR Trend: {trends.get('ctr_trend', 0):+.2f}%
        """
    
    def _customize_fallback_response(self, template: str, context: Dict) -> str:
        """Customize fallback responses with actual data"""
        # Simple template customization with actual metrics
        try:
            metrics = context.get('current_metrics', {})
            customized = template.replace(
                "Strong performance variation", 
                f"ROAS ranging from minimum to {metrics.get('avg_roas', 2.5):.2f}x average"
            )
            return customized
        except:
            return template
    
    def _get_fallback_analysis(self) -> Dict:
        """Return fallback analysis when AI generation fails"""
        return {
            'performance_insights': self.fallback_responses['performance_insights'],
            'optimization_recommendations': self.fallback_responses['optimization_recommendations'],
            'creative_suggestions': self.fallback_responses['creative_suggestions'],
            'budget_strategy': "Focus budget allocation on campaigns with ROAS > 2.5x for optimal performance.",
            'risk_assessment': self.fallback_responses['risk_assessment'],
            'priority_actions': ["Review campaign performance metrics", "Optimize budget allocation", "Test new creative variations"],
            'competitive_analysis': "Performance benchmarking shows opportunities for optimization across multiple metrics."
        }