import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple

class RealisticMarketingDataGenerator:
    """Generate sophisticated marketing data with realistic patterns and correlations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.platform_benchmarks = self._load_platform_benchmarks()
        self.seasonal_patterns = self._load_seasonal_patterns()
        
    def _load_platform_benchmarks(self) -> Dict:
        """Industry benchmark data for different platforms"""
        return {
            'Meta': {
                'base_ctr': 2.5,
                'ctr_std': 0.8,
                'base_cpc': 1.20,
                'cpc_std': 0.35,
                'base_cvr': 8.5,
                'cvr_std': 2.5,
                'impression_multiplier': 1.0,
                'campaign_types': {
                    'conversion': {'ctr_boost': 1.2, 'cvr_boost': 1.4, 'cpc_penalty': 1.3},
                    'awareness': {'ctr_boost': 0.8, 'cvr_boost': 0.6, 'cpc_penalty': 0.7},
                    'retargeting': {'ctr_boost': 1.8, 'cvr_boost': 2.2, 'cpc_penalty': 1.6}
                }
            },
            'Google': {
                'base_ctr': 3.2,
                'ctr_std': 1.0,
                'base_cpc': 1.50,
                'cpc_std': 0.45,
                'base_cvr': 12.0,
                'cvr_std': 3.0,
                'impression_multiplier': 0.85,
                'campaign_types': {
                    'search': {'ctr_boost': 1.3, 'cvr_boost': 1.5, 'cpc_penalty': 1.4},
                    'display': {'ctr_boost': 0.6, 'cvr_boost': 0.8, 'cpc_penalty': 0.6},
                    'shopping': {'ctr_boost': 1.1, 'cvr_boost': 1.8, 'cpc_penalty': 1.2}
                }
            }
        }
    
    def _load_seasonal_patterns(self) -> Dict:
        """Seasonal and temporal performance patterns"""
        return {
            'day_of_week_factors': {
                0: 0.9,   # Monday - slower start
                1: 1.1,   # Tuesday - peak performance
                2: 1.15,  # Wednesday - highest performance
                3: 1.1,   # Thursday - strong performance
                4: 1.0,   # Friday - standard performance
                5: 0.8,   # Saturday - weekend dip
                6: 0.75   # Sunday - lowest performance
            },
            'hour_of_day_factors': {
                # Business hours see higher conversion rates
                6: 0.6, 7: 0.8, 8: 1.0, 9: 1.2, 10: 1.3, 11: 1.4,
                12: 1.2, 13: 1.1, 14: 1.3, 15: 1.4, 16: 1.3, 17: 1.2,
                18: 1.0, 19: 0.9, 20: 0.8, 21: 0.7, 22: 0.6, 23: 0.5,
                0: 0.4, 1: 0.3, 2: 0.3, 3: 0.3, 4: 0.4, 5: 0.5
            },
            'monthly_seasonality': {
                1: 0.85,  # January - post-holiday slowdown
                2: 0.90,  # February - recovery
                3: 1.05,  # March - spring uptick
                4: 1.1,   # April - strong performance
                5: 1.15,  # May - peak spring
                6: 1.0,   # June - summer start
                7: 0.95,  # July - summer dip
                8: 0.90,  # August - vacation impact
                9: 1.2,   # September - back to business
                10: 1.25, # October - pre-holiday boost
                11: 1.4,  # November - Black Friday effect
                12: 1.3   # December - holiday shopping
            }
        }
    
    def generate_comprehensive_marketing_data(self, platforms=['Meta', 'Google'], days=30, 
                                            campaigns_per_platform=3) -> pd.DataFrame:
        """Generate complete marketing dataset with realistic patterns"""
        
        self.logger.info(f"Generating {days} days of data for {len(platforms)} platforms")
        
        all_campaign_data = []
        
        for platform in platforms:
            platform_campaigns = self._generate_platform_campaigns(platform, campaigns_per_platform)
            
            for campaign in platform_campaigns:
                campaign_performance = self._generate_campaign_timeline(
                    campaign, platform, days
                )
                all_campaign_data.extend(campaign_performance)
        
        df = pd.DataFrame(all_campaign_data)
        
        # Add correlated noise and realistic constraints
        df = self._apply_realistic_constraints(df)
        df = self._add_correlated_metrics(df)
        
        # Sort by date for proper time series
        df = df.sort_values(['platform', 'campaign_id', 'date']).reset_index(drop=True)
        
        self.logger.info(f"Generated {len(df)} data points across {df['campaign_id'].nunique()} campaigns")
        
        return df
    
    def _generate_platform_campaigns(self, platform: str, count: int) -> List[Dict]:
        """Generate campaign configurations for a platform"""
        campaigns = []
        
        benchmark = self.platform_benchmarks[platform]
        campaign_types = list(benchmark['campaign_types'].keys())
        
        for i in range(count):
            campaign_type = campaign_types[i % len(campaign_types)]
            
            # Budget affects performance (higher budget = more competition)
            budget_tier = np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.5, 0.2])
            budget_map = {'low': 200, 'medium': 500, 'high': 1000}
            
            campaigns.append({
                'campaign_id': f'{platform}_{campaign_type}_{i+1:03d}',
                'campaign_name': f'{platform} {campaign_type.title()} {i+1:03d}',
                'campaign_type': campaign_type,
                'platform': platform,
                'daily_budget': budget_map[budget_tier],
                'budget_tier': budget_tier,
                'launch_date': datetime.now() - timedelta(days=np.random.randint(1, 180))
            })
        
        return campaigns
    
    def _generate_campaign_timeline(self, campaign: Dict, platform: str, days: int) -> List[Dict]:
        """Generate daily performance data for a campaign"""
        
        timeline_data = []
        benchmark = self.platform_benchmarks[platform]
        campaign_modifiers = benchmark['campaign_types'][campaign['campaign_type']]
        
        # Campaign lifecycle affects performance
        campaign_age_start = (datetime.now() - timedelta(days=days-1) - campaign['launch_date']).days
        
        for day_offset in range(days):
            current_date = datetime.now() - timedelta(days=days-1-day_offset)
            campaign_age = campaign_age_start + day_offset
            
            # Get temporal factors
            day_factor = self.seasonal_patterns['day_of_week_factors'][current_date.weekday()]
            month_factor = self.seasonal_patterns['monthly_seasonality'][current_date.month]
            lifecycle_factor = self._get_lifecycle_factor(campaign_age)
            
            # Budget competition factor (higher budget = more competitive auctions)
            budget_factor = {
                'low': 1.1,    # Less competition, better efficiency
                'medium': 1.0,  # Average competition
                'high': 0.9     # High competition, higher costs
            }[campaign['budget_tier']]
            
            # Generate base metrics with all factors applied
            total_factor = day_factor * month_factor * lifecycle_factor * budget_factor
            
            impressions = self._generate_impressions(
                campaign, benchmark, campaign_modifiers, total_factor
            )
            
            ctr = self._generate_ctr(
                benchmark, campaign_modifiers, total_factor, campaign_age
            )
            
            clicks = int(impressions * (ctr / 100))
            
            cpc = self._generate_cpc(
                benchmark, campaign_modifiers, total_factor, campaign['budget_tier']
            )
            
            spend = clicks * cpc
            
            cvr = self._generate_cvr(
                benchmark, campaign_modifiers, total_factor, clicks
            )
            
            conversions = int(clicks * (cvr / 100))
            
            timeline_data.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'platform': platform,
                'campaign_id': campaign['campaign_id'],
                'campaign_name': campaign['campaign_name'], 
                'campaign_type': campaign['campaign_type'],
                'budget_tier': campaign['budget_tier'],
                'campaign_age': campaign_age,
                'impressions': max(impressions, 50),  # Minimum threshold
                'clicks': max(clicks, 1),
                'spend': round(max(spend, 1.0), 2),
                'conversions': max(conversions, 0),
                'ctr': round(max(ctr, 0.1), 2),
                'cpc': round(max(cpc, 0.05), 2),
                'cvr': round(max(cvr, 0.1), 2),
                'day_factor': day_factor,
                'month_factor': month_factor,
                'lifecycle_factor': lifecycle_factor
            })
        
        return timeline_data
    
    def _generate_impressions(self, campaign: Dict, benchmark: Dict, 
                            modifiers: Dict, total_factor: float) -> int:
        """Generate realistic impression volumes"""
        base_impressions = campaign['daily_budget'] * 15  # Budget to impressions ratio
        
        # Apply platform and campaign type factors
        adjusted_impressions = (
            base_impressions * 
            benchmark['impression_multiplier'] * 
            total_factor *
            np.random.normal(1.0, 0.2)  # Random variation
        )
        
        return int(max(adjusted_impressions, 50))
    
    def _generate_ctr(self, benchmark: Dict, modifiers: Dict, 
                     total_factor: float, campaign_age: int) -> float:
        """Generate realistic CTR with learning effects"""
        
        base_ctr = benchmark['base_ctr'] * modifiers['ctr_boost']
        
        # Learning effect - CTR improves with campaign age up to a point
        learning_factor = min(1.0 + (campaign_age * 0.001), 1.2) if campaign_age > 0 else 0.8
        
        ctr = (
            base_ctr * 
            total_factor * 
            learning_factor *
            np.random.normal(1.0, benchmark['ctr_std'] / base_ctr)
        )
        
        return max(ctr, 0.1)
    
    def _generate_cpc(self, benchmark: Dict, modifiers: Dict, 
                     total_factor: float, budget_tier: str) -> float:
        """Generate realistic CPC with auction dynamics"""
        
        base_cpc = benchmark['base_cpc'] * modifiers['cpc_penalty']
        
        # Higher demand periods = higher CPC
        auction_pressure = total_factor if total_factor > 1.0 else 1.0
        
        # Budget tier affects CPC (higher budget competes in more expensive auctions)
        tier_multiplier = {'low': 0.8, 'medium': 1.0, 'high': 1.3}[budget_tier]
        
        cpc = (
            base_cpc * 
            auction_pressure * 
            tier_multiplier *
            np.random.normal(1.0, benchmark['cpc_std'] / base_cpc)
        )
        
        return max(cpc, 0.05)
    
    def _generate_cvr(self, benchmark: Dict, modifiers: Dict, 
                     total_factor: float, clicks: int) -> float:
        """Generate realistic conversion rates with volume effects"""
        
        base_cvr = benchmark['base_cvr'] * modifiers['cvr_boost']
        
        # Volume effect - very low click volumes can have unstable CVR
        volume_stability = min(1.0, clicks / 10) if clicks > 0 else 0.1
        
        cvr = (
            base_cvr * 
            total_factor * 
            np.random.normal(1.0, (benchmark['cvr_std'] / base_cvr) / volume_stability)
        )
        
        return max(cvr, 0.1)
    
    def _get_lifecycle_factor(self, campaign_age: int) -> float:
        """Campaign performance lifecycle curve"""
        if campaign_age <= 0:
            return 0.7  # Pre-launch/learning phase
        elif campaign_age <= 7:
            return 0.8 + (campaign_age * 0.04)  # Learning phase improvement
        elif campaign_age <= 30:
            return 1.1  # Optimal performance phase
        elif campaign_age <= 90:
            return 1.1 - ((campaign_age - 30) * 0.002)  # Gradual decline
        else:
            return 0.9  # Mature campaign plateau
    
    def _apply_realistic_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply realistic business constraints and correlations"""
        
        # Spend cannot exceed daily budget significantly
        df['theoretical_max_spend'] = df['budget_tier'].map({
            'low': 250, 'medium': 600, 'high': 1200  # 25% buffer over budget
        })
        df['spend'] = np.minimum(df['spend'], df['theoretical_max_spend'])
        
        # Recalculate dependent metrics after spend capping
        df['actual_cpc'] = df['spend'] / df['clicks']
        df['cpc'] = df['actual_cpc']
        
        # Remove temporary columns
        df = df.drop(['theoretical_max_spend', 'actual_cpc'], axis=1)
        
        return df
    
    def _add_correlated_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived metrics and correlations"""
        
        # Conversion value (varies by campaign type and platform)
        aov_map = {
            'Meta': {'conversion': 45, 'awareness': 35, 'retargeting': 55},
            'Google': {'search': 65, 'display': 40, 'shopping': 75}
        }
        
        df['avg_order_value'] = df.apply(
            lambda row: aov_map[row['platform']][row['campaign_type']] * 
                       np.random.normal(1.0, 0.15), 
            axis=1
        ).round(2)
        
        df['conversion_value'] = df['conversions'] * df['avg_order_value']
        
        # Calculate final KPIs
        df['roas'] = df['conversion_value'] / df['spend']
        df['cpm'] = (df['spend'] / df['impressions']) * 1000
        df['cpa'] = df['spend'] / df['conversions'].replace(0, np.nan)
        
        # Quality indicators
        df['click_quality'] = (df['cvr'] / df['ctr']) * 100  # Conversion efficiency
        
        return df

def generate_demo_dataset(days=30) -> pd.DataFrame:
    """Convenience function to generate demonstration dataset"""
    generator = RealisticMarketingDataGenerator()
    
    return generator.generate_comprehensive_marketing_data(
        platforms=['Meta', 'Google'],
        days=days,
        campaigns_per_platform=3
    )