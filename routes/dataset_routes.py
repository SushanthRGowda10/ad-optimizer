import os
import uuid
import pandas as pd
from datetime import datetime
from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    current_app,
    send_file
)
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename

from models import db, Dataset
from utils.data_cleaner import clean_dataset
from analytics.feature_engineering import engineer_features

dataset_bp = Blueprint('dataset', __name__)

ALLOWED_EXTENSIONS = {'csv'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_active_df(app, user_id):
    """Load the latest active dataset for the logged-in user"""

    ds = (
        Dataset.query
        .filter_by(user_id=user_id, is_active=True)
        .order_by(Dataset.uploaded_at.desc())
        .first()
    )

    if ds is None:
        return None

    path = os.path.join(app.config['UPLOAD_FOLDER'], ds.filename)

    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    
    # Filter by user if the 'account' column exists
    if 'account' in df.columns:
        from flask_login import current_user
        if current_user.is_authenticated:
            df = df[df['account'] == current_user.username]
            if df.empty:
                return None

    return df


@dataset_bp.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():

    if request.method == 'POST':

        if 'file' not in request.files:
            flash('No file selected', 'danger')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No file selected', 'danger')
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash('Only CSV files allowed', 'danger')
            return redirect(request.url)

        original_name = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{original_name}"

        save_path = os.path.join(
            current_app.config['UPLOAD_FOLDER'],
            unique_name
        )

        file.save(save_path)

        try:

            df = pd.read_csv(save_path)

            result = clean_dataset(df)

            df_clean = result['data']

            df_clean = engineer_features(df_clean)
            
            # Associate this data uniquely to the logged in user
            df_clean['account'] = current_user.username

            df_clean.to_csv(save_path, index=False)

            # deactivate previous datasets of this user
            Dataset.query.filter_by(
                user_id=current_user.id,
                is_active=True
            ).update({'is_active': False})

            ds = Dataset(
                filename=unique_name,
                original_name=original_name,
                row_count=len(df_clean),
                column_count=len(df_clean.columns),
                user_id=current_user.id,
                is_active=True
            )

            db.session.add(ds)
            db.session.commit()

            from models.ml_models import train_models

            trained = train_models(df_clean)

            # Cache model in the correct format for prediction routes
            current_app.trained_models[current_user.id] = {
                'basic': trained,
                'df': df_clean,
                'last_trained': datetime.now()
            }

            flash("Dataset uploaded successfully!", "success")
            
            # Redirect to analytics page after successful upload
            return redirect(url_for('analytics.analytics'))

        except Exception as e:

            if os.path.exists(save_path):
                os.remove(save_path)

            flash(f"Processing failed: {e}", "danger")

            return redirect(request.url)

    return render_template('upload.html')


@dataset_bp.route('/view/<int:dataset_id>')
@login_required
def view_dataset(dataset_id):
    """View dataset details and reports"""
    
    ds = Dataset.query.filter_by(id=dataset_id, user_id=current_user.id).first()
    
    if ds is None:
        flash('Dataset not found.', 'warning')
        return redirect(url_for('auth.profile'))
    
    # Load the dataset file
    path = os.path.join(current_app.config['UPLOAD_FOLDER'], ds.filename)
    
    if not os.path.exists(path):
        flash('Dataset file not found.', 'warning')
        return redirect(url_for('auth.profile'))
    
    # Read the dataset
    try:
        df = pd.read_csv(path)
        
        # Filter by user if the 'account' column exists
        if 'account' in df.columns:
            df = df[df['account'] == current_user.username]
            if df.empty:
                flash('No data available for your account.', 'warning')
                return redirect(url_for('auth.profile'))
    except Exception as e:
        flash(f'Error loading dataset: {e}', 'danger')
        return redirect(url_for('auth.profile'))
    
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
    
    # Get all analytics data
    stats = summary_stats(df)
    top_camp = top_campaigns(df, n=10)
    plat_perf = platform_performance(df)
    age_perf = age_group_performance(df)
    dev_perf = device_performance(df)
    loc_perf = location_performance(df)
    ctr_data = ctr_trend(df)
    conv_data = conversion_trend(df)
    recs = generate_recommendations(df)
    
    return render_template(
        'reports.html',
        stats=stats,
        top_campaigns=top_camp,
        platform_perf=plat_perf,
        age_perf=age_perf,
        device_perf=dev_perf,
        loc_perf=loc_perf,
        ctr_trend=ctr_data,
        conv_trend=conv_data,
        recs=recs,
        no_data=False,
        has_data=True
    )


@dataset_bp.route('/download/csv')
@login_required
def download_csv():

    df = get_active_df(current_app, current_user.id)

    if df is None:
        flash('No dataset available.', 'warning')
        return redirect(url_for('analytics.dashboard'))

    import io

    buf = io.BytesIO()

    df.to_csv(buf, index=False)

    buf.seek(0)

    return send_file(
        buf,
        mimetype='text/csv',
        download_name='campaign_report.csv',
        as_attachment=True
    )


@dataset_bp.route('/download/excel')
@login_required
def download_excel():

    df = get_active_df(current_app, current_user.id)

    if df is None:
        flash('No dataset available.', 'warning')
        return redirect(url_for('analytics.dashboard'))

    import io

    buf = io.BytesIO()

    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)

    buf.seek(0)

    return send_file(
        buf,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        download_name='campaign_report.xlsx',
        as_attachment=True
    )