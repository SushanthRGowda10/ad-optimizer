from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user
from models import db, User

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():

    if current_user.is_authenticated:
        return redirect(url_for('analytics.dashboard'))

    if request.method == 'POST':

        email = request.form.get('email')
        password = request.form.get('password')
        remember = request.form.get('remember') == 'on'

        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):

            login_user(user, remember=remember)

            flash('Welcome back!', 'success')

            return redirect(url_for('analytics.dashboard'))

        flash('Invalid email or password.', 'danger')

    return render_template('login.html')


@auth_bp.route('/signup', methods=['GET', 'POST'])
def signup():

    if current_user.is_authenticated:
        return redirect(url_for('analytics.dashboard'))

    if request.method == 'POST':

        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm = request.form.get('confirm_password')

        if password != confirm:
            flash('Passwords do not match.', 'danger')
            return render_template('signup.html')

        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'danger')
            return render_template('signup.html')

        if User.query.filter_by(username=username).first():
            flash('Username already taken.', 'danger')
            return render_template('signup.html')

        user = User(username=username, email=email)

        user.set_password(password)

        db.session.add(user)

        db.session.commit()

        login_user(user)

        flash('Account created successfully!', 'success')

        return redirect(url_for('analytics.dashboard'))

    return render_template('signup.html')


@auth_bp.route('/logout')
@login_required
def logout():

    logout_user()

    flash('You have been logged out.', 'info')

    return redirect(url_for('auth.login'))


@auth_bp.route('/profile')
@login_required
def profile():
    from routes.dataset_routes import get_active_df
    from models import Dataset
    
    # Get all datasets for this user
    datasets = Dataset.query.filter_by(user_id=current_user.id).all()
    
    return render_template(
        'profile.html',
        datasets=datasets
    )