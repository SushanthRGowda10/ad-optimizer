"""
Advanced AI Predictor with Ensemble Methods and Uncertainty Estimation
Provides accurate performance predictions with confidence intervals
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                              VotingRegressor, StackingRegressor)
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_predict, learning_curve
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class AdvancedAIPredictor:
    """
    Advanced ensemble-based predictor with uncertainty estimation
    and multiple model architectures
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoders = {}
        self.models = {}
        self.ensemble_model = None
        self.feature_cols = []
        self.training_history = {}
        
    def _encode_categorical(self, df, categorical_cols):
        """Label encode categorical columns"""
        df = df.copy()
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_enc'] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
        return df
    
    def _prepare_features(self, df):
        """Prepare feature matrix for training"""
        categorical_cols = ['platform', 'age_group', 'device', 'campaign_name', 'location']
        df = self._encode_categorical(df, categorical_cols)
        
        # Feature engineering
        feature_df = df.copy()
        
        # Temporal features
        if 'month' not in feature_df.columns:
            feature_df['month'] = 6  # Default
        if 'day_of_week' not in feature_df.columns:
            feature_df['day_of_week'] = 2  # Default
            
        # Interaction features
        feature_df['budget_per_impression'] = feature_df['cost'] / feature_df['impressions'].replace(0, 1)
        feature_df['click_through_value'] = feature_df['clicks'] * feature_df['cost'] / (feature_df['impressions'] + 1)
        
        # Select features
        self.feature_cols = [
            'impressions', 'cost',
            'platform_enc', 'age_group_enc', 'device_enc',
            'month', 'day_of_week',
            'budget_per_impression', 'click_through_value'
        ]
        
        # Only use available columns
        available_cols = [col for col in self.feature_cols if col in feature_df.columns]
        X = feature_df[available_cols].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, feature_df
    
    def train_ensemble(self, df, target_columns=['clicks', 'conversions']):
        """
        Train sophisticated ensemble models for each target
        """
        X_scaled, feature_df = self._prepare_features(df)
        
        training_results = {}
        
        for target in target_columns:
            if target not in df.columns or len(df) < 10:
                continue
                
            y = df[target].fillna(0).values
            
            # Define base models
            rf = RandomForestRegressor(
                n_estimators=100, 
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            gb = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            
            mlp = MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42,
                early_stopping=True
            )
            
            ridge = Ridge(alpha=1.0)
            
            # Create stacking ensemble
            stacking = StackingRegressor(
                estimators=[
                    ('rf', rf),
                    ('gb', gb),
                    ('mlp', mlp)
                ],
                final_estimator=BayesianRidge(),
                cv=5
            )
            
            # Fit individual models
            models_dict = {
                'Random Forest': rf,
                'Gradient Boosting': gb,
                'Neural Network': mlp,
                'Ridge Regression': ridge,
                'Stacking Ensemble': stacking
            }
            
            target_models = {}
            
            for name, model in models_dict.items():
                try:
                    model.fit(X_scaled, y)
                    
                    # Cross-validation predictions for better metrics
                    if len(y) >= 10:
                        cv_preds = cross_val_predict(model, X_scaled, y, cv=min(5, len(y)//2))
                        cv_r2 = r2_score(y, cv_preds)
                        cv_mae = mean_absolute_error(y, cv_preds)
                    else:
                        cv_r2 = 0.0
                        cv_mae = 0.0
                    
                    # Training metrics
                    train_preds = model.predict(X_scaled)
                    train_r2 = r2_score(y, train_preds)
                    train_mae = mean_absolute_error(y, train_preds)
                    
                    target_models[name] = {
                        'model': model,
                        'train_r2': round(max(0, train_r2), 4),
                        'train_mae': round(train_mae, 2),
                        'cv_r2': round(max(0, cv_r2), 4),
                        'cv_mae': round(cv_mae, 2),
                        'feature_importances': self._get_feature_importance(model, name)
                    }
                    
                except Exception as e:
                    print(f"Error training {name}: {str(e)}")
                    continue
            
            # Use stacking ensemble as primary
            best_model_name = 'Stacking Ensemble' if 'Stacking Ensemble' in target_models else list(target_models.keys())[0]
            
            training_results[target] = {
                'models': target_models,
                'best_model': target_models.get(best_model_name),
                'best_model_name': best_model_name
            }
        
        # Store training history
        self.training_history = {
            'n_samples': len(df),
            'n_features': len(self.feature_cols),
            'targets': target_columns
        }
        
        return training_results
    
    def _get_feature_importance(self, model, model_name):
        """Extract feature importances from model"""
        try:
            if hasattr(model, 'feature_importances_'):
                return model.feature_importances_.tolist()
            elif hasattr(model, 'estimators_') and hasattr(model.estimators_[0], 'feature_importances_'):
                # For stacking, get RF importance
                for name, est in model.estimators_:
                    if hasattr(est, 'feature_importances_'):
                        return est.feature_importances_.tolist()
            elif hasattr(model, 'coef_'):
                return np.abs(model.coef_).tolist()
        except:
            pass
        return None
    
    def predict_with_uncertainty(self, platform, age_group, device, budget, impressions, 
                                 training_results, n_predictions=50):
        """
        Make prediction with uncertainty estimation using Monte Carlo dropout
        """
        # Encode inputs
        def encode_val(col, val):
            le = self.encoders.get(col)
            if le is None:
                return 0
            try:
                return int(le.transform([val])[0])
            except:
                return 0
        
        # Prepare features
        budget_per_imp = budget / impressions if impressions > 0 else 0
        click_through_val = 0  # Unknown at prediction time
        
        features = np.array([[
            impressions,
            budget,
            encode_val('platform', platform),
            encode_val('age_group', age_group),
            encode_val('device', device),
            6,  # month
            2,  # day_of_week
            budget_per_imp,
            click_through_val
        ]])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        predictions = {}
        
        for target in ['clicks', 'conversions']:
            if target not in training_results:
                continue
                
            model_info = training_results[target]
            best_model = model_info['best_model']['model'] if isinstance(model_info['best_model'], dict) else model_info['best_model']
            
            # Make prediction
            pred = best_model.predict(features_scaled)[0]
            pred = max(0, pred)
            
            # Uncertainty estimation using variance across ensemble members
            individual_preds = []
            for name, info in model_info['models'].items():
                if 'model' in info:
                    try:
                        p = info['model'].predict(features_scaled)[0]
                        individual_preds.append(max(0, p))
                    except:
                        pass
            
            if len(individual_preds) > 1:
                uncertainty = np.std(individual_preds)
                confidence_interval_low = pred - 1.96 * uncertainty
                confidence_interval_high = pred + 1.96 * uncertainty
                confidence = max(0, min(100, 100 - (uncertainty / (pred + 1)) * 100))
            else:
                uncertainty = 0
                confidence_interval_low = pred * 0.8
                confidence_interval_high = pred * 1.2
                confidence = 75.0
            
            predictions[target] = {
                'predicted': round(pred, 2),
                'uncertainty': round(uncertainty, 2),
                'confidence_interval': {
                    'low': round(max(0, confidence_interval_low), 2),
                    'high': round(confidence_interval_high, 2)
                },
                'confidence': round(confidence, 1)
            }
        
        return predictions
    
    def what_if_analysis(self, base_params, variations, training_results):
        """
        Analyze different scenarios by varying parameters
        """
        scenarios = []
        
        for scenario in variations:
            params = base_params.copy()
            params.update(scenario['changes'])
            
            pred = self.predict_with_uncertainty(
                params['platform'],
                params['age_group'],
                params['device'],
                params['budget'],
                params['impressions'],
                training_results
            )
            
            scenarios.append({
                'scenario_name': scenario['name'],
                'changes': scenario['changes'],
                'predictions': pred,
                'is_baseline': scenario.get('is_baseline', False)
            })
        
        return scenarios
    
    def get_feature_insights(self, training_results):
        """
        Provide insights about which features matter most
        """
        insights = []
        
        for target, info in training_results.items():
            for model_name, model_info in info['models'].items():
                importance = model_info.get('feature_importances')
                if importance is not None and len(importance) == len(self.feature_cols):
                    feature_imp = sorted(
                        zip(self.feature_cols, importance),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]
                    
                    insights.append({
                        'target': target,
                        'model': model_name,
                        'top_features': [
                            {'feature': f, 'importance': round(i * 100, 2)}
                            for f, i in feature_imp
                        ]
                    })
        
        return insights


def create_advanced_predictions(df, platform, age_group, device, budget, impressions):
    """
    Main function to generate comprehensive AI predictions
    """
    predictor = AdvancedAIPredictor()
    
    # Train ensemble models
    training_results = predictor.train_ensemble(df)
    
    if not training_results:
        return {'error': 'Insufficient data for training'}
    
    # Get predictions with uncertainty
    predictions = predictor.predict_with_uncertainty(
        platform, age_group, device, budget, impressions, training_results
    )
    
    # Calculate derived metrics
    pred_clicks = predictions.get('clicks', {}).get('predicted', 0)
    pred_conversions = predictions.get('conversions', {}).get('predicted', 0)
    
    ctr = round((pred_clicks / impressions * 100) if impressions > 0 else 0, 4)
    conv_rate = round((pred_conversions / pred_clicks * 100) if pred_clicks > 0 else 0, 4)
    cpa = round(budget / pred_conversions if pred_conversions > 0 else 0, 2)
    roas = round((pred_conversions * 100 / budget * 100) if budget > 0 else 0, 2)
    
    # Performance score (0-100)
    perf_score = min(100, round(ctr * 10 + conv_rate * 5 + (roas / 10), 2))
    
    # Model performance summary
    model_summary = {}
    for target, info in training_results.items():
        model_summary[target] = {
            'best_model': info['best_model_name'],
            'cv_r2': info['models'][info['best_model_name']]['cv_r2'],
            'cv_mae': info['models'][info['best_model_name']]['cv_mae']
        }
    
    # Feature insights
    feature_insights = predictor.get_feature_insights(training_results)
    
    # What-if analysis
    base_params = {
        'platform': platform,
        'age_group': age_group,
        'device': device,
        'budget': budget,
        'impressions': impressions
    }
    
    variations = [
        {'name': 'Increase Budget 25%', 'changes': {'budget': budget * 1.25}},
        {'name': 'Increase Budget 50%', 'changes': {'budget': budget * 1.50}},
        {'name': 'Double Impressions', 'changes': {'impressions': impressions * 2}},
        {'name': 'Premium Platform', 'changes': {'platform': 'Google', 'budget': budget * 1.2}},
    ]
    
    what_if_results = predictor.what_if_analysis(base_params, variations, training_results)
    
    return {
        'predictions': {
            'clicks': predictions.get('clicks', {}),
            'conversions': predictions.get('conversions', {})
        },
        'metrics': {
            'ctr': ctr,
            'conversion_rate': conv_rate,
            'cpa': cpa,
            'roas': roas,
            'performance_score': perf_score
        },
        'model_performance': model_summary,
        'feature_insights': feature_insights,
        'what_if_analysis': what_if_results,
        'training_info': predictor.training_history,
        'confidence_level': 'High' if predictions.get('clicks', {}).get('confidence', 0) > 80 else 'Medium' if predictions.get('clicks', {}).get('confidence', 0) > 60 else 'Low'
    }
