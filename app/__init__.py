"""
Flask Application Factory
=========================
This module contains the application factory function that creates and configures
the Flask application instance. It follows the application factory pattern for
better modularity and testing capabilities.
"""

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import os

# Initialize Flask extensions
db = SQLAlchemy()
migrate = Migrate()

def create_app(config_name=None):
    """
    Application factory function that creates and configures a Flask application.
    
    Args:
        config_name (str, optional): Configuration environment name.
                                   If None, uses FLASK_ENV environment variable.
    
    Returns:
        Flask: Configured Flask application instance
    """
    # Create Flask application instance
    app = Flask(__name__, instance_relative_config=True)
    
    # Determine configuration environment
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')
    
    # Load configuration
    if config_name == 'testing':
        app.config.from_object('config.TestingConfig')
    elif config_name == 'production':
        app.config.from_object('config.ProductionConfig')
    else:  # development
        app.config.from_object('config.DevelopmentConfig')
    
    # Ensure instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass  # Directory already exists
    
    # Initialize Flask extensions with app
    db.init_app(app)
    migrate.init_app(app, db)
    
    # Import models to ensure they are registered with SQLAlchemy
    from app import models
    
    # Register blueprints
    from app.routes import main
    app.register_blueprint(main)
    
    # Create database tables within application context
    with app.app_context():
        try:
            db.create_all()
            print("Database tables created successfully!")
        except Exception as e:
            print(f"Error creating database tables: {e}")
    
    # Add custom template filters or context processors if needed
    @app.context_processor
    def inject_user():
        """Inject common variables into all templates"""
        return dict(
            app_name="Plant AI Predictor",
            app_version="1.0.0"
        )
    
    # Error handlers
    @app.errorhandler(404)
    def not_found_error(error):
        """Handle 404 Not Found errors"""
        from flask import render_template
        return render_template('404.html'), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 Internal Server errors"""
        from flask import render_template
        db.session.rollback()
        return render_template('500.html'), 500
    
    # Log application startup
    app.logger.info(f'PAI Flask application started in {config_name} mode')
    
    return app