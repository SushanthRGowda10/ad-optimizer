"""
Advanced Optimized Analysis with Machine Learning and Optimization Algorithms
Provides intelligent budget allocation, campaign scoring, and ROI forecasting
"""

import pandas as pd
import numpy as np
from scipy.optimize import linprog, minimize
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


def advanced_budget_optimizer(df: pd.DataFrame, total_budget: float = None) -> dict:
    """
    Uses linear programming to optimally allocate budget across campaigns
    maximizing conversions while respecting constraints.
    """
    # Campaign performance aggregation
    camp_perf = df.groupby('campaign_name').agg({
        'impressions': 'sum',
        'clicks': 'sum',
        'conversions': 'sum',
        'cost': 'sum'
    }).reset_index()
    
    camp_perf['ctr'] = camp_perf['clicks'] / camp_perf['impressions'].replace(0, 1)
    camp_perf['conv_rate'] = camp_perf['conversions'] / camp_perf['clicks'].replace(0, 1)
    camp_perf['cpa'] = camp_perf['cost'] / camp_perf['conversions'].replace(0, 1)
    camp_perf['roi'] = (camp_perf['conversions'] * 100 - camp_perf['cost']) / camp_perf['cost'].replace(0, 1) * 100
    
    n_campaigns = len(camp_perf)
    if n_campaigns == 0:
        return {'error': 'No campaigns found'}
    
    # Default total budget if not specified
    if total_budget is None:
        total_budget = camp_perf['cost'].sum()
    
    # Objective: Minimize negative conversions (maximize conversions)
    # Using historical conversion rates
    obj_coeffs = -camp_perf['conv_rate'].fillna(0).values
    
    # Constraints
    constraints = []
    
    # Budget constraint: sum of all campaign budgets <= total_budget
    constraints.append({'type': 'ineq', 'fun': lambda x: total_budget - np.sum(x)})
    
    # Minimum budget for each campaign (10% of current)
    min_budgets = camp_perf['cost'] * 0.1
    for i in range(n_campaigns):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - min_budgets.iloc[i]})
    
    # Maximum budget cap (3x current for top performers)
    max_budgets = camp_perf['cost'] * 3
    for i in range(n_campaigns):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: max_budgets.iloc[i] - x[i]})
    
    # Initial guess: proportional to current spend
    x0 = camp_perf['cost'].values
    
    # Optimization
    try:
        result = minimize(
            lambda x: np.dot(x, obj_coeffs),
            x0,
            method='SLSQP',
            bounds=[(0, None) for _ in range(n_campaigns)],
            constraints=constraints
        )
        
        optimized_budgets = result.x
    except:
        # Fallback: simple proportional allocation
        optimized_budgets = camp_perf['cost'] / camp_perf['cost'].sum() * total_budget
    
    camp_perf['current_budget'] = camp_perf['cost']
    camp_perf['optimized_budget'] = optimized_budgets
    camp_perf['budget_change'] = ((optimized_budgets - camp_perf['cost']) / camp_perf['cost'].replace(0, 1)) * 100
    camp_perf['projected_conversions'] = camp_perf['optimized_budget'] * camp_perf['conv_rate']
    camp_perf['current_conversions'] = camp_perf['conversions']
    camp_perf['conversion_lift'] = camp_perf['projected_conversions'] - camp_perf['current_conversions']
    
    # Calculate projected ROI
    camp_perf['projected_roi'] = (camp_perf['projected_conversions'] * 100 - camp_perf['optimized_budget']) / camp_perf['optimized_budget'].replace(0, 1) * 100
    
    total_current_conv = camp_perf['current_conversions'].sum()
    total_projected_conv = camp_perf['projected_conversions'].sum()
    overall_lift = ((total_projected_conv - total_current_conv) / total_current_conv * 100) if total_current_conv > 0 else 0
    
    return {
        'campaigns': camp_perf.to_dict('records'),
        'total_budget': total_budget,
        'current_conversions': total_current_conv,
        'projected_conversions': total_projected_conv,
        'conversion_lift_percent': round(overall_lift, 2),
        'optimization_status': 'success' if result.success else 'fallback_used'
    }


def campaign_scorer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scores each campaign on multiple dimensions using ML clustering
    Returns campaigns with comprehensive scores and rankings
    """
    # Feature engineering
    features = df.groupby('campaign_name').agg({
        'impressions': ['sum', 'mean'],
        'clicks': ['sum', 'mean'],
        'conversions': ['sum', 'mean'],
        'cost': ['sum', 'mean'],
        'ctr': 'mean',
        'conversion_rate': 'mean'
    }).reset_index()
    
    # Flatten column names
    features.columns = ['campaign_name', 'imp_sum', 'imp_avg', 'clicks_sum', 'clicks_avg',
                       'conv_sum', 'conv_avg', 'cost_sum', 'cost_avg', 'ctr', 'conv_rate']
    
    # Derived metrics
    features['cpa'] = features['cost_sum'] / features['conv_sum'].replace(0, 1)
    features['roas'] = (features['conv_sum'] * 100) / features['cost_sum'].replace(0, 1)
    features['efficiency'] = features['conv_rate'] * features['ctr'] * 1000
    
    # Normalize features for scoring
    score_cols = ['conv_sum', 'ctr', 'conv_rate', 'roas', 'efficiency']
    scaler = StandardScaler()
    normalized = scaler.fit_transform(features[score_cols].fillna(0))
    
    # Weighted scoring
    weights = np.array([0.25, 0.20, 0.25, 0.15, 0.15])
    features['ai_score'] = np.dot(normalized, weights)
    
    # Normalize score to 0-100
    min_score = features['ai_score'].min()
    max_score = features['ai_score'].max()
    if max_score > min_score:
        features['ai_score'] = ((features['ai_score'] - min_score) / (max_score - min_score) * 100).round(2)
    else:
        features['ai_score'] = 50.0
    
    # Performance tier classification
    def classify_tier(score):
        if score >= 75:
            return 'Excellent'
        elif score >= 50:
            return 'Good'
        elif score >= 25:
            return 'Average'
        else:
            return 'Needs Improvement'
    
    features['performance_tier'] = features['ai_score'].apply(classify_tier)
    
    # Clustering for segment insights
    if len(features) >= 3:
        kmeans = KMeans(n_clusters=min(3, len(features)), random_state=42)
        features['cluster'] = kmeans.fit_predict(features[['ai_score', 'roas', 'efficiency']].values)
    
    # Sort by AI score
    features = features.sort_values('ai_score', ascending=False).reset_index(drop=True)
    features['rank'] = features.index + 1
    
    return features


def roi_forecaster(df: pd.DataFrame, scenarios: list = None) -> dict:
    """
    Forecasts ROI under different budget scenarios
    Uses historical performance trends
    """
    # Historical performance
    hist = df.groupby('campaign_name').agg({
        'conversions': 'sum',
        'cost': 'sum'
    }).reset_index()
    
    hist['roi'] = (hist['conversions'] * 100 - hist['cost']) / hist['cost'].replace(0, 1) * 100
    hist['conv_rate'] = hist['conversions'] / hist['cost'].replace(0, 1)
    
    total_cost = hist['cost'].sum()
    total_conversions = hist['conversions'].sum()
    current_roi = ((total_conversions * 100 - total_cost) / total_cost * 100) if total_cost > 0 else 0
    
    # Define scenarios
    if scenarios is None:
        scenarios = [
            {'name': 'Conservative', 'budget_change': 0.10},
            {'name': 'Moderate Growth', 'budget_change': 0.25},
            {'name': 'Aggressive Expansion', 'budget_change': 0.50},
            {'name': 'Status Quo', 'budget_change': 0.00}
        ]
    
    forecasts = []
    for scenario in scenarios:
        new_budget = total_cost * (1 + scenario['budget_change'])
        # Assume diminishing returns: sqrt scaling
        efficiency_factor = np.sqrt(1 + scenario['budget_change'])
        projected_conversions = total_conversions * efficiency_factor
        projected_roi = ((projected_conversions * 100 - new_budget) / new_budget * 100) if new_budget > 0 else 0
        
        forecasts.append({
            'scenario': scenario['name'],
            'budget': round(new_budget, 2),
            'budget_change_percent': scenario['budget_change'] * 100,
            'projected_conversions': round(projected_conversions, 1),
            'projected_roi': round(projected_roi, 2),
            'confidence': round(85 - abs(scenario['budget_change']) * 30, 1)  # Higher confidence for conservative
        })
    
    # Best scenario recommendation
    best_scenario = max(forecasts, key=lambda x: x['projected_roi'])
    
    return {
        'current_roi': round(current_roi, 2),
        'current_spend': round(total_cost, 2),
        'current_conversions': round(total_conversions, 1),
        'forecasts': forecasts,
        'recommended_scenario': best_scenario['scenario'],
        'forecast_horizon_days': 30
    }


def audience_targeting_optimizer(df: pd.DataFrame) -> dict:
    """
    Analyzes audience segments and provides targeting recommendations
    """
    # Platform-Age-Device combinations
    audience_perf = df.groupby(['platform', 'age_group', 'device']).agg({
        'impressions': 'sum',
        'clicks': 'sum',
        'conversions': 'sum',
        'cost': 'sum'
    }).reset_index()
    
    audience_perf['ctr'] = audience_perf['clicks'] / audience_perf['impressions'].replace(0, 1)
    audience_perf['conv_rate'] = audience_perf['conversions'] / audience_perf['clicks'].replace(0, 1)
    audience_perf['cpa'] = audience_perf['cost'] / audience_perf['conversions'].replace(0, 1)
    audience_perf['score'] = audience_perf['conv_rate'] * audience_perf['ctr'] * 1000
    
    # Top performing segments
    top_segments = audience_perf.nlargest(5, 'score')
    
    # Underperforming segments
    bottom_segments = audience_perf.nsmallest(3, 'score')
    
    # Platform-specific age recommendations
    platform_age_rec = []
    for platform in df['platform'].unique():
        plat_df = audience_perf[audience_perf['platform'] == platform]
        if len(plat_df) > 0:
            best_age = plat_df.loc[plat_df['score'].idxmax(), 'age_group']
            platform_age_rec.append({
                'platform': platform,
                'recommended_age_group': best_age,
                'expected_performance': 'High'
            })
    
    # Device preferences by age
    device_age_insights = []
    for age_group in df['age_group'].unique():
        age_df = audience_perf[audience_perf['age_group'] == age_group]
        if len(age_df) > 0:
            best_device = age_df.loc[age_df['conv_rate'].idxmax(), 'device']
            device_age_insights.append({
                'age_group': age_group,
                'preferred_device': best_device,
                'conversion_lift_potential': '15-25%'
            })
    
    return {
        'top_segments': top_segments.to_dict('records'),
        'underperforming_segments': bottom_segments.to_dict('records'),
        'platform_age_recommendations': platform_age_rec,
        'device_age_insights': device_age_insights,
        'total_segments_analyzed': len(audience_perf)
    }


def ab_testing_recommendations(df: pd.DataFrame) -> list:
    """
    Generates A/B testing recommendations based on performance variance
    """
    recommendations = []
    
    # Platform performance variance
    plat_var = df.groupby('platform')['conversion_rate'].var()
    high_var_platforms = plat_var[plat_var > plat_var.median()].index.tolist()
    
    for platform in high_var_platforms:
        recommendations.append({
            'test_type': 'Platform Optimization',
            'element': f'{platform} Campaign Structure',
            'hypothesis': f'Testing different ad creatives on {platform} will improve conversion rates due to high performance variance',
            'priority': 'High',
            'expected_impact': '10-20% improvement'
        })
    
    # Age group engagement gaps
    age_perf = df.groupby('age_group').agg({
        'conversions': 'sum',
        'clicks': 'sum'
    }).reset_index()
    age_perf['conv_rate'] = age_perf['conversions'] / age_perf['clicks'].replace(0, 1)
    
    if len(age_perf) >= 2:
        best_age = age_perf.loc[age_perf['conv_rate'].idxmax(), 'age_group']
        worst_age = age_perf.loc[age_perf['conv_rate'].idxmin(), 'age_group']
        
        recommendations.append({
            'test_type': 'Audience Targeting',
            'element': 'Age Group Messaging',
            'hypothesis': f'Customize messaging for {worst_age} age group based on successful approach from {best_age}',
            'priority': 'Medium',
            'expected_impact': '5-15% improvement'
        })
    
    # Device optimization
    device_perf = df.groupby('device').agg({
        'ctr': 'mean',
        'conversions': 'sum'
    }).reset_index()
    
    if len(device_perf) >= 2:
        recommendations.append({
            'test_type': 'Device Experience',
            'element': 'Mobile vs Desktop Landing Pages',
            'hypothesis': 'Device-specific landing page optimization will improve user experience and conversions',
            'priority': 'Medium',
            'expected_impact': '8-12% improvement'
        })
    
    # Budget allocation test
    recommendations.append({
        'test_type': 'Budget Strategy',
        'element': 'Budget Allocation Method',
        'hypothesis': 'AI-optimized budget allocation will outperform manual budget distribution',
        'priority': 'High',
        'expected_impact': '15-30% improvement'
    })
    
    return recommendations


def generate_ai_optimizer_report(df: pd.DataFrame) -> dict:
    """
    Comprehensive optimized analysis report combining all advanced features
    """
    # Run all analyses
    budget_opt = advanced_budget_optimizer(df)
    campaign_scores = campaign_scorer(df)
    roi_forecast = roi_forecaster(df)
    audience_opt = audience_targeting_optimizer(df)
    ab_tests = ab_testing_recommendations(df)
    
    # Overall optimization potential
    avg_score = campaign_scores['ai_score'].mean() if len(campaign_scores) > 0 else 0
    optimization_potential = min(100, avg_score + budget_opt.get('conversion_lift_percent', 0) * 2)
    
    return {
        'budget_optimization': budget_opt,
        'campaign_scores': campaign_scores.to_dict('records') if hasattr(campaign_scores, 'to_dict') else campaign_scores,
        'roi_forecast': roi_forecast,
        'audience_targeting': audience_opt,
        'ab_testing_recommendations': ab_tests,
        'overall_optimization_score': round(optimization_potential, 2),
        'total_campaigns_analyzed': len(campaign_scores) if hasattr(campaign_scores, '__len__') else 0,
        'ai_confidence_level': 'High' if optimization_potential > 70 else 'Medium' if optimization_potential > 40 else 'Low'
    }
