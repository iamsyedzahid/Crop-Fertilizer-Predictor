"""
Flask Routes Module
==================
This module contains all the route handlers for the PAI Flask application.
It handles web requests for crop prediction, fertilizer recommendation,
and general application routes.
"""

from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
from app.models import PredictionHistory, db
from app.ml.crop_model import CropPredictor
from app.ml.fertilizer_model import FertilizerRecommender
import logging
from datetime import datetime

# Create Blueprint for modular route organization
main = Blueprint('main', __name__)

# Initialize ML models
crop_predictor = CropPredictor()
fertilizer_recommender = FertilizerRecommender()

# Configure logging
logger = logging.getLogger(__name__)

@main.route('/')
def home():
    """
    Home page route that displays the main landing page.
    
    Returns:
        str: Rendered home.html template
    """
    try:
        # Get recent predictions count for display
        recent_predictions = PredictionHistory.query.count()
        return render_template('home.html', recent_predictions=recent_predictions)
    except Exception as e:
        logger.error(f"Error loading home page: {str(e)}")
        return render_template('home.html', recent_predictions=0)


@main.route('/predict')
def predict_form():
    """
    Display the prediction form page.
    
    Returns:
        str: Rendered predict.html template
    """
    return render_template('predict.html')


@main.route('/predict/crop', methods=['POST'])
def predict_crop():
    """
    Handle crop prediction requests.
    Accepts soil and environmental parameters and returns crop recommendations.
    
    Returns:
        JSON: Prediction results or error message
    """
    try:
        # Extract form data
        data = request.get_json() if request.is_json else request.form
        
        # Required parameters for crop prediction
        required_params = ['nitrogen', 'phosphorus', 'potassium', 'temperature', 
                          'humidity', 'ph', 'rainfall']
        
        # Validate input parameters
        prediction_data = {}
        for param in required_params:
            value = data.get(param)
            if value is None or value == '':
                return jsonify({
                    'success': False, 
                    'error': f'Missing required parameter: {param}'
                }), 400
            
            try:
                prediction_data[param] = float(value)
            except ValueError:
                return jsonify({
                    'success': False, 
                    'error': f'Invalid value for {param}. Must be a number.'
                }), 400
        
        # Validate parameter ranges
        if not (0 <= prediction_data['nitrogen'] <= 200):
            return jsonify({'success': False, 'error': 'Nitrogen must be between 0-200'}), 400
        if not (0 <= prediction_data['phosphorus'] <= 200):
            return jsonify({'success': False, 'error': 'Phosphorus must be between 0-200'}), 400
        if not (0 <= prediction_data['potassium'] <= 200):
            return jsonify({'success': False, 'error': 'Potassium must be between 0-200'}), 400
        if not (0 <= prediction_data['temperature'] <= 50):
            return jsonify({'success': False, 'error': 'Temperature must be between 0-50°C'}), 400
        if not (0 <= prediction_data['humidity'] <= 100):
            return jsonify({'success': False, 'error': 'Humidity must be between 0-100%'}), 400
        if not (0 <= prediction_data['ph'] <= 14):
            return jsonify({'success': False, 'error': 'pH must be between 0-14'}), 400
        if not (0 <= prediction_data['rainfall'] <= 500):
            return jsonify({'success': False, 'error': 'Rainfall must be between 0-500mm'}), 400
        
        # Make crop prediction
        prediction_result = crop_predictor.predict(prediction_data)
        
        # Save prediction to database
        try:
            prediction_record = PredictionHistory(
                prediction_type='crop',
                input_data=str(prediction_data),
                result=str(prediction_result),
                timestamp=datetime.utcnow()
            )
            db.session.add(prediction_record)
            db.session.commit()
            logger.info(f"Crop prediction saved: {prediction_result}")
        except Exception as db_error:
            logger.error(f"Database error: {str(db_error)}")
            db.session.rollback()
        
        return jsonify({
            'success': True,
            'prediction': prediction_result,
            'input_parameters': prediction_data,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        logger.error(f"Error in crop prediction: {str(e)}")
        return jsonify({
            'success': False, 
            'error': 'An error occurred during prediction. Please try again.'
        }), 500


@main.route('/predict/fertilizer', methods=['POST'])
def predict_fertilizer():
    """
    Handle fertilizer recommendation requests.
    Accepts soil parameters and crop type to recommend fertilizer.
    
    Returns:
        JSON: Fertilizer recommendation results or error message
    """
    try:
        # Extract form data
        data = request.get_json() if request.is_json else request.form
        
        # Required parameters for fertilizer recommendation
        required_params = ['nitrogen', 'phosphorus', 'potassium', 'crop_type']
        
        # Validate input parameters
        recommendation_data = {}
        for param in required_params:
            value = data.get(param)
            if value is None or value == '':
                return jsonify({
                    'success': False, 
                    'error': f'Missing required parameter: {param}'
                }), 400
            
            if param == 'crop_type':
                recommendation_data[param] = str(value).lower().strip()
            else:
                try:
                    recommendation_data[param] = float(value)
                except ValueError:
                    return jsonify({
                        'success': False, 
                        'error': f'Invalid value for {param}. Must be a number.'
                    }), 400
        
        # Validate numeric parameter ranges
        if not (0 <= recommendation_data['nitrogen'] <= 200):
            return jsonify({'success': False, 'error': 'Nitrogen must be between 0-200'}), 400
        if not (0 <= recommendation_data['phosphorus'] <= 200):
            return jsonify({'success': False, 'error': 'Phosphorus must be between 0-200'}), 400
        if not (0 <= recommendation_data['potassium'] <= 200):
            return jsonify({'success': False, 'error': 'Potassium must be between 0-200'}), 400
        
        # Make fertilizer recommendation
        recommendation_result = fertilizer_recommender.recommend(recommendation_data)
        
        # Save recommendation to database
        try:
            prediction_record = PredictionHistory(
                prediction_type='fertilizer',
                input_data=str(recommendation_data),
                result=str(recommendation_result),
                timestamp=datetime.utcnow()
            )
            db.session.add(prediction_record)
            db.session.commit()
            logger.info(f"Fertilizer recommendation saved: {recommendation_result}")
        except Exception as db_error:
            logger.error(f"Database error: {str(db_error)}")
            db.session.rollback()
        
        return jsonify({
            'success': True,
            'recommendation': recommendation_result,
            'input_parameters': recommendation_data,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        logger.error(f"Error in fertilizer recommendation: {str(e)}")
        return jsonify({
            'success': False, 
            'error': 'An error occurred during recommendation. Please try again.'
        }), 500


@main.route('/history')
def prediction_history():
    """
    Display prediction history page.
    Shows recent predictions made by users.
    
    Returns:
        str: Rendered history template with predictions
    """
    try:
        page = request.args.get('page', 1, type=int)
        per_page = 10
        
        predictions = PredictionHistory.query.order_by(
            PredictionHistory.timestamp.desc()
        ).paginate(
            page=page, 
            per_page=per_page, 
            error_out=False
        )
        
        return render_template('history.html', predictions=predictions)
        
    except Exception as e:
        logger.error(f"Error loading prediction history: {str(e)}")
        flash('Error loading prediction history.', 'error')
        return redirect(url_for('main.home'))


@main.route('/api/health')
def health_check():
    """
    Health check endpoint for monitoring application status.
    
    Returns:
        JSON: Application health status
    """
    try:
        # Check database connection
        db.session.execute('SELECT 1')
        
        # Check ML models status
        crop_status = crop_predictor.is_ready()
        fertilizer_status = fertilizer_recommender.is_ready()
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'database': 'connected',
            'crop_model': 'ready' if crop_status else 'not_ready',
            'fertilizer_model': 'ready' if fertilizer_status else 'not_ready'
        })
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@main.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    return render_template('404.html'), 404


@main.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    db.session.rollback()
    return render_template('500.html'), 500


# Context processor to inject common template variables
@main.app_context_processor
def inject_template_vars():
    """Inject common variables into all templates"""
    return {
        'current_year': datetime.now().year,
        'app_name': 'Plant AI Predictor'
    }