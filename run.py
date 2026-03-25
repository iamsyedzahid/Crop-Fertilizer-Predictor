#!/usr/bin/env python3
"""
Flask Application Entry Point
============================
This file serves as the entry point for the PAI (Plant AI) Flask application.
It creates and configures the Flask app instance and runs the development server.
"""

from app import create_app
import os

# Create the Flask application instance using the application factory pattern
app = create_app()

if __name__ == '__main__':
    # Get configuration from environment variables or use defaults
    host = os.environ.get('FLASK_HOST', '127.0.0.1')
    port = int(os.environ.get('FLASK_PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    print(f"Starting PAI Flask Application...")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Debug mode: {debug}")
    
    # Run the Flask development server
    app.run(
        host=host,
        port=port,
        debug=debug,
        threaded=True
    )