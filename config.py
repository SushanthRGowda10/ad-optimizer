import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'ad-optimizer-secret-2024'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///database.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'datasets')
    REPORT_FOLDER = os.path.join(os.path.dirname(__file__), 'reports')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload
