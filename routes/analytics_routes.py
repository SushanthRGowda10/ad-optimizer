import pandas as pd
from flask import Blueprint, render_template, current_app
from flask_login import login_required, current_user
from datetime import datetime

from routes.dataset_routes import get_active_df
from analytics.eda import (
    summary_stats,
    top_campaigns,
    platform_performance,
    age_group_performance,
    device_performance,
    location_performance,
    ctr_trend,
    conversion_trend
)
from models.optimization_engine import generate_recommendations
from models.ai_optimizer import generate_ai_optimizer_report
from models.ml_models import train_models

analytics_bp = Blueprint('analytics', __name__)


def _load(app, user_id):
    """Load dataset only for the logged-in user"""
    return get_active_df(app, user_id)


def generate_insights(df):
    """Generate simple smart insights from dataset"""
    insights = []

    try:
        if 'platform' in df.columns and 'clicks' in df.columns:
            plat = df.groupby('platform')['clicks'].mean().idxmax()
            insights.append(f"📈 {plat} platform shows highest engagement.")

        if 'age_group' in df.columns and 'clicks' in df.columns:
            age = df.groupby('age_group')['clicks'].mean().idxmax()
            insights.append(f"🎯 Age group {age} performs best.")

        if 'device' in df.columns and 'clicks' in df.columns:
            device = df.groupby('device')['clicks'].mean().idxmax()
            insights.append(f"📱 {device} users have highest interaction.")

        if not insights:
            insights.append("Not enough valid columns available for insights.")

    except Exception:
        insights.append("Not enough data for insights.")

    return insights


def get_model_comparison(trained):
    """Prepare lightweight model comparison from trained models"""
    comparison = {}

    if not trained or 'results' not in trained:
        return comparison

    for target, models in trained['results'].items():
        comparison[target] = {}
        for name, metrics in models.items():
            comparison[target][name] = {
                'r2': metrics.get('r2'),
                'mae': metrics.get('mae')
            }

    return comparison


@analytics_bp.route('/dashboard')
@login_required
def dashboard():
    df = _load(current_app, current_user.id)
    current_hour = datetime.now().hour

    if df is None:
        return render_template(
            'dashboard.html',
            no_data=True,
            has_data=False,
            current_hour=current_hour
        )

    stats = summary_stats(df)
    top_camp = top_campaigns(df)
    insights = generate_insights(df)

    try:
        trained = train_models(df)
        model_comparison = get_model_comparison(trained)
    except Exception:
        model_comparison = {}

    return render_template(
        'dashboard.html',
        stats=stats,
        top_campaigns=top_camp,
        insights=insights,
        model_comparison=model_comparison,
        no_data=False,
        has_data=True,
        current_hour=current_hour
    )


@analytics_bp.route('/analytics')
@login_required
def analytics():
    df = _load(current_app, current_user.id)

    if df is None:
        return render_template(
            'analytics.html',
            no_data=True,
            has_data=False
        )

    top_camp = top_campaigns(df, n=10)
    plat_perf = platform_performance(df)
    age_perf = age_group_performance(df)
    dev_perf = device_performance(df)
    loc_perf = location_performance(df)
    ctr_data = ctr_trend(df)
    conv_data = conversion_trend(df)
    insights = generate_insights(df)

    return render_template(
        'analytics.html',
        top_campaigns=top_camp,
        platform_perf=plat_perf,
        age_perf=age_perf,
        device_perf=dev_perf,
        loc_perf=loc_perf,
        ctr_trend=ctr_data,
        conv_trend=conv_data,
        insights=insights,
        no_data=False,
        has_data=True
    )


@analytics_bp.route('/optimization')
@login_required
def optimization():
    df = _load(current_app, current_user.id)

    if df is None:
        return render_template(
            'optimization.html',
            no_data=True,
            has_data=False
        )

    recs = generate_recommendations(df)
    ai_report = generate_ai_optimizer_report(df)

    return render_template(
        'optimization.html',
        recs=recs,
        ai_report=ai_report,
        no_data=False,
        has_data=True
    )


@analytics_bp.route('/reports')
@login_required
def reports():
    df = _load(current_app, current_user.id)

    if df is None:
        return render_template(
            'reports.html',
            no_data=True,
            has_data=False
        )

    stats = summary_stats(df)
    top_camp = top_campaigns(df, n=10)
    recs = generate_recommendations(df)
    insights = generate_insights(df)

    return render_template(
        'reports.html',
        stats=stats,
        top_campaigns=top_camp,
        recs=recs,
        insights=insights,
        no_data=False,
        has_data=True
    )