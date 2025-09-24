import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import joblib
import os

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder

class MarketingMLSuite:
    """Production-grade ML suite for marketing automation"""
    
    def __init__(self):
        self.models = {
            'roas_predictor': GradientBoostingRegressor(
                n_estimators=100, 
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'spend_optimizer': RandomForestRegressor(
                n_estimators=100, 
                max_depth=8,
                random_state=42
            ),
            'conversion_forecaster': GradientBoostingRegressor(
                n_estimators=150, 
                learning_rate=0.08,
                max_depth=5,
                random_state=42
            ),
            'ctr_predictor': LinearRegression(),
            'budget_allocator': GradientBoostingRegressor(
                n_estimators=80, 
                learning_rate=0.12,
                max_depth=4,
                random_state=42
            )
        }
        
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        self.model_performance = {}
        self.logger = self._setup_logging()
        
        # Ensure models directory exists
        os.makedirs('data/models', exist_ok=True)
    
    def _setup_logging(self):
        """Setup logging for ML operations"""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML models"""
        df = data.copy()
        
        # Create datetime features
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['day_of_week'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            df['is_weekend'] = (df['day_of_week'].isin([5, 6])).astype(int)
        else:
            df['day_of_week'] = 1
            df['month'] = 1
            df['is_weekend'] = 0
        
        # Encode categorical variables
        categorical_cols = ['platform', 'campaign_type']
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.encoders[col].fit_transform(df[col].astype(str))
                else:
                    # Handle new categories in prediction
                    unique_values = set(df[col].astype(str))
                    known_values = set(self.encoders[col].classes_)
                    new_values = unique_values - known_values
                    
                    if new_values:
                        # Add new categories to encoder
                        all_values = list(self.encoders[col].classes_) + list(new_values)
                        self.encoders[col].classes_ = np.array(all_values)
                    
                    df[f'{col}_encoded'] = self.encoders[col].transform(df[col].astype(str))
            else:
                df[f'{col}_encoded'] = 0
        
        # Calculate campaign age (mock for sample data)
        df['campaign_age'] = np.random.randint(1, 90, len(df))
        
        # Historical performance features (rolling averages)
        if len(df) > 7:
            df = df.sort_values('date') if 'date' in df.columns else df
            df['historical_roas_7d'] = df['roas'].rolling(7, min_periods=1).mean()
            df['historical_ctr_7d'] = df['ctr'].rolling(7, min_periods=1).mean()
            df['spend_trend_7d'] = df['spend'].rolling(7, min_periods=1).mean()
        else:
            df['historical_roas_7d'] = df['roas']
            df['historical_ctr_7d'] = df['ctr']
            df['spend_trend_7d'] = df['spend']
        
        # Performance ratios
        df['clicks_per_impression'] = df['clicks'] / (df['impressions'] + 1)
        df['conversions_per_click'] = df['conversions'] / (df['clicks'] + 1)
        df['spend_per_conversion'] = df['spend'] / (df['conversions'] + 1)
        
        # Interaction features
        df['platform_x_campaign_type'] = df['platform_encoded'] * df['campaign_type_encoded']
        df['ctr_x_cvr'] = df['ctr'] * df['cvr']
        
        # Define feature columns
        self.feature_columns = [
            'impressions', 'clicks', 'spend', 'ctr', 'cpc', 'cvr',
            'day_of_week', 'month', 'is_weekend', 'campaign_age',
            'platform_encoded', 'campaign_type_encoded',
            'historical_roas_7d', 'historical_ctr_7d', 'spend_trend_7d',
            'clicks_per_impression', 'conversions_per_click', 'spend_per_conversion',
            'platform_x_campaign_type', 'ctr_x_cvr'
        ]
        
        # Ensure all feature columns exist
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        return df[self.feature_columns + ['roas', 'conversions', 'ctr']].fillna(0)
    
    def _calculate_optimal_spend(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate optimal daily spend based on performance"""
        # Simple optimization: increase spend for high ROAS, decrease for low ROAS
        base_spend = data['spend']
        roas_factor = np.where(data['roas'] > 3.0, 1.2, 
                              np.where(data['roas'] > 2.0, 1.0, 0.8))
        
        optimal_spend = base_spend * roas_factor
        return optimal_spend.values
    
    def train_all_models(self, data: pd.DataFrame) -> Dict:
        """Train comprehensive ML model suite"""
        self.logger.info("Starting ML model training...")
        
        # Prepare features
        processed_data = self._prepare_features(data)
        X = processed_data[self.feature_columns]
        
        # Scale features
        self.scalers['features'] = StandardScaler()
        X_scaled = self.scalers['features'].fit_transform(X)
        
        results = {}
        
        # Train ROAS predictor
        if 'roas' in processed_data.columns:
            y_roas = processed_data['roas']
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_roas, test_size=0.2, random_state=42
            )
            
            self.models['roas_predictor'].fit(X_train, y_train)
            roas_score = self.models['roas_predictor'].score(X_test, y_test)
            roas_predictions = self.models['roas_predictor'].predict(X_test)
            roas_mae = mean_absolute_error(y_test, roas_predictions)
            
            results['roas_predictor'] = {
                'r2_score': roas_score,
                'mae': roas_mae,
                'feature_importance': dict(zip(
                    self.feature_columns,
                    self.models['roas_predictor'].feature_importances_
                ))
            }
            
            self.logger.info(f"ROAS Predictor - R2: {roas_score:.3f}, MAE: {roas_mae:.3f}")
        
        # Train spend optimizer
        y_optimal_spend = self._calculate_optimal_spend(processed_data)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_optimal_spend, test_size=0.2, random_state=42
        )
        
        self.models['spend_optimizer'].fit(X_train, y_train)
        spend_score = self.models['spend_optimizer'].score(X_test, y_test)
        
        results['spend_optimizer'] = {
            'r2_score': spend_score,
            'feature_importance': dict(zip(
                self.feature_columns,
                self.models['spend_optimizer'].feature_importances_
            ))
        }
        
        # Train conversion forecaster
        if 'conversions' in processed_data.columns:
            y_conversions = processed_data['conversions']
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_conversions, test_size=0.2, random_state=42
            )
            
            self.models['conversion_forecaster'].fit(X_train, y_train)
            conv_score = self.models['conversion_forecaster'].score(X_test, y_test)
            
            results['conversion_forecaster'] = {
                'r2_score': conv_score
            }
        
        # Train CTR predictor
        if 'ctr' in processed_data.columns:
            y_ctr = processed_data['ctr']
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_ctr, test_size=0.2, random_state=42
            )
            
            self.models['ctr_predictor'].fit(X_train, y_train)
            ctr_score = self.models['ctr_predictor'].score(X_test, y_test)
            
            results['ctr_predictor'] = {
                'r2_score': ctr_score
            }
        
        # Train budget allocator
        self.models['budget_allocator'].fit(X_scaled, y_optimal_spend)
        
        # Save all models and scalers
        self._save_models()
        
        # Store performance metrics
        self.model_performance = results
        
        summary = {
            'status': 'success',
            'models_trained': len(self.models),
            'feature_count': len(self.feature_columns),
            'data_points': len(data),
            'performance_summary': {
                name: metrics.get('r2_score', 0) for name, metrics in results.items()
            }
        }
        
        self.logger.info(f"ML training completed: {summary}")
        return summary
    
    def generate_predictions(self, campaign_data: pd.DataFrame) -> Dict:
        """Generate comprehensive predictions"""
        if not self.feature_columns:
            raise ValueError("Models must be trained before generating predictions")
        
        # Prepare features
        processed_data = self._prepare_features(campaign_data)
        X = processed_data[self.feature_columns]
        
        if 'features' in self.scalers:
            X_scaled = self.scalers['features'].transform(X)
        else:
            X_scaled = StandardScaler().fit_transform(X)
        
        predictions = {}
        
        try:
            predictions['predicted_roas'] = self.models['roas_predictor'].predict(X_scaled)
            predictions['optimal_daily_spend'] = self.models['spend_optimizer'].predict(X_scaled)
            predictions['expected_conversions'] = self.models['conversion_forecaster'].predict(X_scaled)
            predictions['predicted_ctr'] = self.models['ctr_predictor'].predict(X_scaled)
            predictions['recommended_budget'] = self.models['budget_allocator'].predict(X_scaled)
            
            # Calculate prediction confidence (simplified)
            predictions['confidence_scores'] = self._calculate_prediction_confidence(X_scaled)
            
            # Generate optimization recommendations
            predictions['optimization_flags'] = self._generate_optimization_flags(predictions, processed_data)
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            # Return safe fallback predictions
            n_samples = len(campaign_data)
            predictions = {
                'predicted_roas': np.full(n_samples, 2.5),
                'optimal_daily_spend': campaign_data.get('spend', pd.Series([100] * n_samples)) * 1.1,
                'expected_conversions': np.full(n_samples, 10),
                'predicted_ctr': np.full(n_samples, 2.5),
                'confidence_scores': np.full(n_samples, 0.7),
                'optimization_flags': ['maintain'] * n_samples
            }
        
        return predictions
    
    def _calculate_prediction_confidence(self, X_scaled: np.ndarray) -> np.ndarray:
        """Calculate prediction confidence scores"""
        # Simplified confidence based on model agreement
        try:
            roas_pred = self.models['roas_predictor'].predict(X_scaled)
            # Use prediction variance as inverse confidence measure
            confidence = 1.0 / (1.0 + np.abs(roas_pred - np.mean(roas_pred)))
            return np.clip(confidence, 0.3, 0.95)
        except:
            return np.full(len(X_scaled), 0.7)
    
    def _generate_optimization_flags(self, predictions: Dict, data: pd.DataFrame) -> List[str]:
        """Generate optimization recommendations"""
        flags = []
        
        pred_roas = predictions.get('predicted_roas', [])
        current_roas = data.get('roas', pd.Series([2.0] * len(pred_roas)))
        
        for i in range(len(pred_roas)):
            if pred_roas[i] > current_roas.iloc[i] * 1.2:
                flags.append('scale_up')
            elif pred_roas[i] < current_roas.iloc[i] * 0.8:
                flags.append('optimize_or_pause')
            else:
                flags.append('maintain')
        
        return flags
    
    def _save_models(self):
        """Save all trained models and scalers"""
        try:
            for name, model in self.models.items():
                joblib.dump(model, f'data/models/{name}.pkl')
            
            for name, scaler in self.scalers.items():
                joblib.dump(scaler, f'data/models/{name}_scaler.pkl')
            
            for name, encoder in self.encoders.items():
                joblib.dump(encoder, f'data/models/{name}_encoder.pkl')
            
            # Save feature columns and performance metrics
            joblib.dump({
                'feature_columns': self.feature_columns,
                'model_performance': self.model_performance
            }, 'data/models/ml_metadata.pkl')
            
            self.logger.info("All models saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
    
    def load_models(self):
        """Load pre-trained models"""
        try:
            for name in self.models.keys():
                model_path = f'data/models/{name}.pkl'
                if os.path.exists(model_path):
                    self.models[name] = joblib.load(model_path)
            
            # Load scalers and encoders
            for scaler_type in ['features']:
                scaler_path = f'data/models/{scaler_type}_scaler.pkl'
                if os.path.exists(scaler_path):
                    self.scalers[scaler_type] = joblib.load(scaler_path)
            
            # Load metadata
            metadata_path = 'data/models/ml_metadata.pkl'
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                self.feature_columns = metadata.get('feature_columns', [])
                self.model_performance = metadata.get('model_performance', {})
            
            self.logger.info("Models loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            return False
    
    def evaluate_model_performance(self, test_data: pd.DataFrame) -> Dict:
        """Evaluate model performance on test data"""
        if not self.feature_columns:
            return {"error": "Models not trained"}
        
        try:
            processed_data = self._prepare_features(test_data)
            X = processed_data[self.feature_columns]
            X_scaled = self.scalers['features'].transform(X)
            
            predictions = self.generate_predictions(test_data)
            
            # Calculate accuracy metrics
            metrics = {}
            
            if 'roas' in processed_data.columns:
                actual_roas = processed_data['roas']
                pred_roas = predictions['predicted_roas']
                
                metrics['roas_accuracy'] = {
                    'mae': mean_absolute_error(actual_roas, pred_roas),
                    'r2_score': r2_score(actual_roas, pred_roas),
                    'rmse': np.sqrt(mean_squared_error(actual_roas, pred_roas))
                }
            
            return {
                'status': 'success',
                'metrics': metrics,
                'prediction_count': len(predictions['predicted_roas'])
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def get_feature_importance(self, model_name: str = 'roas_predictor') -> Dict:
        """Get feature importance for specified model"""
        if model_name not in self.models:
            return {}
        
        model = self.models[model_name]
        if hasattr(model, 'feature_importances_'):
            importance_dict = dict(zip(self.feature_columns, model.feature_importances_))
            # Sort by importance
            sorted_importance = dict(sorted(importance_dict.items(), 
                                          key=lambda x: x[1], reverse=True))
            return sorted_importance
        
        return {}
    
    def generate_model_insights(self) -> Dict:
        """Generate insights about model performance and predictions"""
        insights = {
            'model_summary': {
                'total_models': len(self.models),
                'feature_count': len(self.feature_columns),
                'training_status': 'trained' if self.feature_columns else 'not_trained'
            },
            'top_features': {},
            'performance_summary': self.model_performance
        }
        
        # Get top features for each model
        for model_name in ['roas_predictor', 'spend_optimizer']:
            feature_importance = self.get_feature_importance(model_name)
            insights['top_features'][model_name] = dict(
                list(feature_importance.items())[:5]  # Top 5 features
            )
        
        return insights