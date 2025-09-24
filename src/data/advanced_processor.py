import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler

class DataValidator:
    """Validate marketing data structure and quality"""
    
    def __init__(self):
        self.required_columns = [
            'date', 'platform', 'campaign_id', 'campaign_name',
            'impressions', 'clicks', 'spend', 'conversions'
        ]
        self.quality_thresholds = {
            'min_impressions': 50,
            'max_ctr': 25.0,  # 25% CTR is unusual
            'min_ctr': 0.01,  # 0.01% CTR is too low
            'max_cpc': 50.0,  # $50 CPC is very high
            'min_conversions_for_roas': 1
        }
    
    def validate(self, data: pd.DataFrame) -> Dict:
        """Comprehensive data validation"""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'data_quality_score': 0
        }
        
        # Check required columns
        missing_columns = [col for col in self.required_columns if col not in data.columns]
        if missing_columns:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_columns}")
        
        # Data quality checks
        if 'impressions' in data.columns:
            low_impressions = (data['impressions'] < self.quality_thresholds['min_impressions']).sum()
            if low_impressions > 0:
                validation_results['warnings'].append(f"{low_impressions} campaigns with impressions < {self.quality_thresholds['min_impressions']}")
        
        if 'ctr' in data.columns:
            high_ctr = (data['ctr'] > self.quality_thresholds['max_ctr']).sum()
            low_ctr = (data['ctr'] < self.quality_thresholds['min_ctr']).sum()
            
            if high_ctr > 0:
                validation_results['warnings'].append(f"{high_ctr} campaigns with unusually high CTR (>{self.quality_thresholds['max_ctr']}%)")
            if low_ctr > 0:
                validation_results['warnings'].append(f"{low_ctr} campaigns with very low CTR (<{self.quality_thresholds['min_ctr']}%)")
        
        # Calculate overall quality score
        base_score = 100
        score_deductions = len(validation_results['errors']) * 25 + len(validation_results['warnings']) * 5
        validation_results['data_quality_score'] = max(0, base_score - score_deductions)
        
        return validation_results

class FeatureEngineer:
    """Create advanced features for marketing analysis"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive feature set"""
        df = data.copy()
        
        # Time-based features
        df = self._create_time_features(df)
        
        # Performance ratio features
        df = self._create_performance_ratios(df)
        
        # Rolling window features
        df = self._create_rolling_features(df)
        
        # Campaign lifecycle features
        df = self._create_lifecycle_features(df)
        
        # Efficiency metrics
        df = self._create_efficiency_metrics(df)
        
        # Platform benchmarking
        df = self._create_platform_benchmarks(df)
        
        return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['month'] = df['date'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        
        return df
    
    def _create_performance_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create performance ratio features"""
        # Efficiency ratios
        df['clicks_per_1k_impressions'] = (df['clicks'] / df['impressions']) * 1000
        df['spend_per_conversion'] = df['spend'] / df['conversions'].replace(0, np.nan)
        df['conversions_per_100_clicks'] = (df['conversions'] / df['clicks']) * 100
        
        # Value ratios
        df['revenue_per_impression'] = df['conversion_value'] / df['impressions']
        df['profit_margin'] = (df['conversion_value'] - df['spend']) / df['conversion_value']
        df['cost_efficiency'] = df['conversions'] / df['spend']
        
        return df
    
    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window features for trend analysis"""
        df = df.sort_values(['campaign_id', 'date'])
        
        # 7-day rolling averages
        for col in ['roas', 'ctr', 'cpc', 'cvr']:
            if col in df.columns:
                df[f'{col}_7d_avg'] = df.groupby('campaign_id')[col].transform(
                    lambda x: x.rolling(window=7, min_periods=1).mean()
                )
                df[f'{col}_7d_trend'] = df.groupby('campaign_id')[col].transform(
                    lambda x: x.pct_change(periods=7)
                )
        
        # Performance momentum (3-day vs 7-day comparison)
        df['roas_momentum'] = (
            df.groupby('campaign_id')['roas'].transform(lambda x: x.rolling(3).mean()) /
            df.groupby('campaign_id')['roas'].transform(lambda x: x.rolling(7).mean())
        )
        
        return df
    
    def _create_lifecycle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create campaign lifecycle features"""
        # Campaign age
        df['campaign_start_date'] = df.groupby('campaign_id')['date'].transform('min')
        df['campaign_age_days'] = (df['date'] - df['campaign_start_date']).dt.days
        
        # Performance lifecycle stage
        def get_lifecycle_stage(age_days):
            if age_days <= 7:
                return 'launch'
            elif age_days <= 30:
                return 'growth' 
            elif age_days <= 90:
                return 'mature'
            else:
                return 'legacy'
        
        df['lifecycle_stage'] = df['campaign_age_days'].apply(get_lifecycle_stage)
        
        # Campaign performance trend
        df['cumulative_spend'] = df.groupby('campaign_id')['spend'].cumsum()
        df['cumulative_conversions'] = df.groupby('campaign_id')['conversions'].cumsum()
        df['lifetime_roas'] = (
            df.groupby('campaign_id')['conversion_value'].cumsum() / 
            df['cumulative_spend']
        )
        
        return df
    
    def _create_efficiency_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced efficiency metrics"""
        # Multi-dimensional efficiency score
        df['efficiency_score'] = (
            df['roas'].rank(pct=True) * 0.4 +  # 40% weight on ROAS
            df['ctr'].rank(pct=True) * 0.3 +   # 30% weight on CTR
            (1 / df['cpc']).rank(pct=True) * 0.2 +  # 20% weight on low CPC
            df['cvr'].rank(pct=True) * 0.1     # 10% weight on CVR
        )
        
        # Performance consistency (coefficient of variation)
        df['roas_consistency'] = df.groupby('campaign_id')['roas'].transform(
            lambda x: 1 / (x.std() / x.mean()) if x.mean() != 0 else 0
        )
        
        # Spend efficiency relative to platform average
        platform_avg_cpc = df.groupby('platform')['cpc'].transform('mean')
        df['cpc_vs_platform_avg'] = df['cpc'] / platform_avg_cpc
        
        return df
    
    def _create_platform_benchmarks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create platform benchmark features"""
        # Performance vs platform median
        for metric in ['roas', 'ctr', 'cpc', 'cvr']:
            if metric in df.columns:
                platform_median = df.groupby('platform')[metric].transform('median')
                df[f'{metric}_vs_platform_median'] = df[metric] / platform_median
        
        # Platform-specific percentile rankings
        for metric in ['roas', 'ctr', 'cvr']:
            if metric in df.columns:
                df[f'{metric}_platform_percentile'] = df.groupby('platform')[metric].rank(pct=True)
        
        return df

class MarketingDataProcessor:
    """Complete marketing data processing pipeline"""
    
    def __init__(self):
        self.validator = DataValidator()
        self.feature_engineer = FeatureEngineer()
        self.logger = self._setup_logging()
    
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def process_campaign_data(self, raw_data: Dict) -> Dict:
        """Complete ETL pipeline for marketing data"""
        self.logger.info("Starting comprehensive data processing pipeline")
        
        # Extract data
        if isinstance(raw_data, dict) and 'combined_data' in raw_data:
            df = raw_data['combined_data']
        else:
            df = raw_data
        
        # Validate input data
        validation_results = self.validator.validate(df)
        self.logger.info(f"Data validation score: {validation_results['data_quality_score']}/100")
        
        if not validation_results['valid']:
            raise ValueError(f"Data validation failed: {validation_results['errors']}")
        
        # Clean and normalize data
        cleaned_df = self._clean_marketing_data(df)
        
        # Feature engineering
        enriched_df = self.feature_engineer.create_features(cleaned_df)
        
        # Calculate advanced KPIs
        final_df = self._calculate_advanced_kpis(enriched_df)
        
        # Generate processing summary
        processing_summary = self._generate_processing_summary(df, final_df, validation_results)
        
        return {
            'processed_data': final_df,
            'original_rows': len(df),
            'final_rows': len(final_df),
            'validation_results': validation_results,
            'processing_summary': processing_summary,
            'features_created': len([col for col in final_df.columns if col not in df.columns])
        }
    
    def _clean_marketing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize marketing data"""
        cleaned_df = df.copy()
        
        # Handle missing values
        cleaned_df['conversions'] = cleaned_df['conversions'].fillna(0)
        cleaned_df['conversion_value'] = cleaned_df['conversion_value'].fillna(0)
        
        # Remove impossible values
        cleaned_df = cleaned_df[cleaned_df['impressions'] > 0]
        cleaned_df = cleaned_df[cleaned_df['spend'] >= 0]
        
        # Cap extreme outliers (99th percentile)
        for col in ['ctr', 'cpc', 'roas']:
            if col in cleaned_df.columns:
                upper_bound = cleaned_df[col].quantile(0.99)
                cleaned_df[col] = cleaned_df[col].clip(upper=upper_bound)
        
        # Standardize text fields
        cleaned_df['platform'] = cleaned_df['platform'].str.title()
        cleaned_df['campaign_name'] = cleaned_df['campaign_name'].str.strip()
        
        return cleaned_df
    
    def _calculate_advanced_kpis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive marketing KPIs"""
        kpi_df = df.copy()
        
        # Core KPIs (recalculated for consistency)
        kpi_df['ctr'] = (kpi_df['clicks'] / kpi_df['impressions']) * 100
        kpi_df['cpc'] = kpi_df['spend'] / kpi_df['clicks']
        kpi_df['cpm'] = (kpi_df['spend'] / kpi_df['impressions']) * 1000
        kpi_df['cvr'] = (kpi_df['conversions'] / kpi_df['clicks']) * 100
        kpi_df['cpa'] = kpi_df['spend'] / kpi_df['conversions'].replace(0, np.nan)
        kpi_df['roas'] = kpi_df['conversion_value'] / kpi_df['spend']
        
        # Advanced KPIs
        kpi_df['impression_share'] = kpi_df.groupby(['platform', 'date'])['impressions'].transform(
            lambda x: x / x.sum()
        )
        
        kpi_df['click_share'] = kpi_df.groupby(['platform', 'date'])['clicks'].transform(
            lambda x: x / x.sum()
        )
        
        # Quality Score (Meta-inspired metric)
        kpi_df['quality_score'] = (
            kpi_df['ctr'].rank(pct=True) * 0.4 +
            kpi_df['cvr'].rank(pct=True) * 0.4 +
            (1 / kpi_df['cpc']).rank(pct=True) * 0.2
        ) * 10  # Scale to 1-10
        
        # Performance grade
        def grade_performance(row):
            if row['roas'] >= 4.0 and row['ctr'] >= 3.0:
                return 'A'
            elif row['roas'] >= 2.5 and row['ctr'] >= 2.0:
                return 'B'
            elif row['roas'] >= 1.5 and row['ctr'] >= 1.0:
                return 'C'
            else:
                return 'D'
        
        kpi_df['performance_grade'] = kpi_df.apply(grade_performance, axis=1)
        
        return kpi_df
    
    def _generate_processing_summary(self, original_df: pd.DataFrame, processed_df: pd.DataFrame, 
                                   validation_results: Dict) -> Dict:
        """Generate comprehensive processing summary"""
        return {
            'data_quality': {
                'original_columns': len(original_df.columns),
                'final_columns': len(processed_df.columns),
                'quality_score': validation_results['data_quality_score'],
                'validation_warnings': len(validation_results['warnings'])
            },
            'performance_metrics': {
                'avg_roas': processed_df['roas'].mean(),
                'avg_ctr': processed_df['ctr'].mean(), 
                'total_spend': processed_df['spend'].sum(),
                'total_conversions': processed_df['conversions'].sum(),
                'campaigns_analyzed': processed_df['campaign_id'].nunique(),
                'date_range_days': (processed_df['date'].max() - processed_df['date'].min()).days
            },
            'data_enrichment': {
                'features_added': len(processed_df.columns) - len(original_df.columns),
                'time_features': sum(1 for col in processed_df.columns if any(x in col for x in ['day_', 'week_', 'month'])),
                'rolling_features': sum(1 for col in processed_df.columns if '_7d_' in col),
                'efficiency_features': sum(1 for col in processed_df.columns if 'efficiency' in col)
            }
        }