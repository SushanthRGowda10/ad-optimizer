from flask import Blueprint, render_template, request, jsonify, current_app, flash, redirect, url_for
from flask_login import login_required, current_user
from datetime import datetime

from models.ml_models import train_models, predict_performance, get_model_comparison
from routes.dataset_routes import get_active_df
from utils.recommender import generate_recommendation

# The variable is named prediction_bp
prediction_bp = Blueprint('prediction', __name__)

PLATFORMS = ['Facebook', 'Instagram', 'Google', 'Twitter', 'LinkedIn', 'TikTok', 'YouTube']
AGE_GROUPS = ['18-24', '25-34', '35-44', '45-54', '55+']
DEVICES = ['Mobile', 'Desktop', 'Tablet']


def _ensure_model(app, user_id):
    """Train model if not already in memory."""
    if not hasattr(app, 'trained_models'):
        app.trained_models = {}
        
    if user_id not in app.trained_models:
        df = get_active_df(app, user_id)
        if df is None:
            return None
        trained = train_models(df)
        app.trained_models[user_id] = {
            'basic': trained,
            'df': df,
            'last_trained': datetime.now()
        }

    model_data = app.trained_models.get(user_id)
    if model_data is None:
        return None

    if isinstance(model_data, dict):
        return model_data.get('basic')
    return model_data


@prediction_bp.route('/prediction', methods=['GET', 'POST'])
@login_required
def prediction():
    trained = _ensure_model(current_app, current_user.id)

    model_metrics = {}
    pred_result = None
    recommendation = None 
    best_model_info = None 

    if trained:
        model_metrics = get_model_comparison(trained)

    if request.method == 'POST':
        if trained is None:
            flash('Please upload a dataset first to enable predictions.', 'warning')
            return redirect(url_for('dataset.upload'))

        platform = request.form.get('platform', 'Facebook')
        age_group = request.form.get('age_group', '25-34')
        device = request.form.get('device', 'Mobile')
        
        try:
            budget = float(request.form.get('budget', 1000))
            impressions = float(request.form.get('impressions', 10000))
        except ValueError:
            flash('Invalid budget or impressions value.', 'danger')
            return redirect(url_for('prediction.prediction'))

        pred_result = predict_performance(
            platform=platform,
            age_group=age_group,
            device=device,
            budget=budget,
            impressions=impressions,
            trained=trained
        )

        ctr = pred_result.get('expected_ctr', 0)
        recommendation = generate_recommendation(ctr, budget)
        best_model_info = pred_result.get('model_used', 'Auto Selected')

    return render_template(
        'prediction.html',
        platforms=PLATFORMS,
        age_groups=AGE_GROUPS,
        devices=DEVICES,
        model_metrics=model_metrics,
        pred_result=pred_result,
        recommendation=recommendation,
        best_model_info=best_model_info
    )

# FIX: Changed @prediction.route to @prediction_bp.route

@login_required
def ab_test():
    if request.method == 'POST':
        # This is where you will handle the A/B test logic later
        # For now, let's just re-render the page to stop the 405 error
        pass
        
    return render_template('ab_test.html')