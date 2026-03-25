"""
Flask Application Configuration
==============================
This module contains configuration classes for different environments
(development, testing, production) for the PAI Flask application.
"""

import os
from datetime import timedelta

class Config:
    """
    Base configuration class with common settings.
    Other configuration classes inherit from this base class.
    """
    # Basic Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # Database Configuration
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_RECORD_QUERIES = True
    
    # Session Configuration
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    SESSION_COOKIE_SECURE = False  # Set to True in production with HTTPS
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Upload Configuration
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    
    # ML Model Configuration
    ML_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app', 'ml', 'models')
    
    # Application Configuration
    ITEMS_PER_PAGE = 20
    
    # Email Configuration (for future use)
    MAIL_SERVER = os.environ.get('MAIL_SERVER')
    MAIL_PORT = int(os.environ.get('MAIL_PORT') or 587)
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'true').lower() in ['true', 'on', '1']
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    
    @staticmethod
    def init_app(app):
        """Initialize application with configuration-specific setup"""
        pass


class DevelopmentConfig(Config):
    """
    Development environment configuration.
    Includes debugging features and development-specific settings.
    """
    DEBUG = True
    TESTING = False
    
    # Database Configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get('DEV_DATABASE_URL') or \
        'sqlite:///' + os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                   'instance', 'pai_dev.sqlite')
    
    # Development-specific settings
    SQLALCHEMY_ECHO = True  # Log all SQL queries
    WTF_CSRF_ENABLED = True
    
    @staticmethod
    def init_app(app):
        """Initialize development-specific configuration"""
        Config.init_app(app)
        
        # Create uploads directory if it doesn't exist
        if not os.path.exists(Config.UPLOAD_FOLDER):
            os.makedirs(Config.UPLOAD_FOLDER)
        
        # Create ML models directory if it doesn't exist
        if not os.path.exists(Config.ML_MODEL_PATH):
            os.makedirs(Config.ML_MODEL_PATH)


class TestingConfig(Config):
    """
    Testing environment configuration.
    Used for running unit tests and integration tests.
    """
    DEBUG = False
    TESTING = True
    
    # Use in-memory database for faster tests
    SQLALCHEMY_DATABASE_URI = os.environ.get('TEST_DATABASE_URL') or 'sqlite:///:memory:'
    
    # Disable CSRF protection in testing
    WTF_CSRF_ENABLED = False
    
    # Faster password hashing for tests
    BCRYPT_LOG_ROUNDS = 4
    
    # Disable request logging during tests
    SQLALCHEMY_RECORD_QUERIES = False
    
    @staticmethod
    def init_app(app):
        """Initialize testing-specific configuration"""
        Config.init_app(app)


class ProductionConfig(Config):
    """
    Production environment configuration.
    Includes security and performance optimizations for production deployment.
    """
    DEBUG = False
    TESTING = False
    
    # Database Configuration - Use environment variable or fallback to instance folder
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                   'instance', 'pai.sqlite')
    
    # Security Settings
    SESSION_COOKIE_SECURE = True  # Requires HTTPS
    WTF_CSRF_ENABLED = True
    
    # Performance Settings
    SQLALCHEMY_ECHO = False  # Don't log SQL queries in production
    SQLALCHEMY_RECORD_QUERIES = False
    
    @staticmethod
    def init_app(app):
        """Initialize production-specific configuration"""
        Config.init_app(app)
        
        # Import logging to set up production logging
        import logging
        from logging.handlers import RotatingFileHandler
        
        # Set up file logging
        if not app.debug:
            file_handler = RotatingFileHandler(
                'logs/pai.log', 
                maxBytes=10240, 
                backupCount=10
            )
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
            ))
            file_handler.setLevel(logging.INFO)
            app.logger.addHandler(file_handler)
            app.logger.setLevel(logging.INFO)
            app.logger.info('PAI startup')


# Configuration dictionary for easy access
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}