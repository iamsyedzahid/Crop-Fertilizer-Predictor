"""
Crop Prediction Model
====================
This module contains the CropPredictor class that handles crop recommendation
based on soil and environmental conditions. Currently implements rule-based
logic as a placeholder for machine learning models.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional
import json
import os
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class CropPredictor:
    """
    Crop prediction class that recommends suitable crops based on soil
    and environmental parameters using rule-based logic.
    
    This is a placeholder implementation that can be replaced with
    actual machine learning models (scikit-learn, TensorFlow, etc.)
    """
    
    def __init__(self, model_version: str = "1.0.0"):
        """
        Initialize the CropPredictor.
        
        Args:
            model_version (str): Version of the prediction model
        """
        self.model_version = model_version
        self.is_loaded = False
        self._crop_database = self._initialize_crop_database()
        self._load_model()
        
        logger.info(f"CropPredictor initialized with version {model_version}")
    
    def _initialize_crop_database(self) -> Dict[str, Dict]:
        """
        Initialize the crop database with optimal growing conditions.
        In a real implementation, this would be loaded from a database or file.
        
        Returns:
            Dict: Crop database with optimal growing conditions
        """
        return {
            "rice": {
                "nitrogen": {"min": 80, "max": 120, "optimal": 100},
                "phosphorus": {"min": 40, "max": 70, "optimal": 55},
                "potassium": {"min": 40, "max": 70, "optimal": 55},
                "temperature": {"min": 20, "max": 35, "optimal": 27},
                "humidity": {"min": 70, "max": 90, "optimal": 80},
                "ph": {"min": 5.5, "max": 7.0, "optimal": 6.2},
                "rainfall": {"min": 100, "max": 300, "optimal": 200},
                "season": ["kharif", "monsoon"],
                "description": "High-yield rice variety suitable for wet conditions"
            },
            "wheat": {
                "nitrogen": {"min": 60, "max": 100, "optimal": 80},
                "phosphorus": {"min": 30, "max": 60, "optimal": 45},
                "potassium": {"min": 30, "max": 60, "optimal": 45},
                "temperature": {"min": 15, "max": 25, "optimal": 20},
                "humidity": {"min": 50, "max": 70, "optimal": 60},
                "ph": {"min": 6.0, "max": 7.5, "optimal": 6.8},
                "rainfall": {"min": 50, "max": 150, "optimal": 100},
                "season": ["rabi", "winter"],
                "description": "High-protein wheat variety for moderate climates"
            },
            "maize": {
                "nitrogen": {"min": 70, "max": 110, "optimal": 90},
                "phosphorus": {"min": 35, "max": 65, "optimal": 50},
                "potassium": {"min": 35, "max": 65, "optimal": 50},
                "temperature": {"min": 18, "max": 30, "optimal": 24},
                "humidity": {"min": 60, "max": 80, "optimal": 70},
                "ph": {"min": 5.8, "max": 7.2, "optimal": 6.5},
                "rainfall": {"min": 60, "max": 200, "optimal": 130},
                "season": ["kharif", "summer"],
                "description": "Versatile corn variety with good yield potential"
            },
            "cotton": {
                "nitrogen": {"min": 50, "max": 90, "optimal": 70},
                "phosphorus": {"min": 25, "max": 55, "optimal": 40},
                "potassium": {"min": 45, "max": 75, "optimal": 60},
                "temperature": {"min": 20, "max": 35, "optimal": 28},
                "humidity": {"min": 50, "max": 80, "optimal": 65},
                "ph": {"min": 5.5, "max": 8.0, "optimal": 6.8},
                "rainfall": {"min": 50, "max": 150, "optimal": 100},
                "season": ["kharif", "summer"],
                "description": "High-quality cotton for textile production"
            },
            "sugarcane": {
                "nitrogen": {"min": 100, "max": 150, "optimal": 125},
                "phosphorus": {"min": 50, "max": 80, "optimal": 65},
                "potassium": {"min": 60, "max": 90, "optimal": 75},
                "temperature": {"min": 25, "max": 40, "optimal": 32},
                "humidity": {"min": 70, "max": 90, "optimal": 80},
                "ph": {"min": 6.0, "max": 8.0, "optimal": 7.0},
                "rainfall": {"min": 150, "max": 400, "optimal": 275},
                "season": ["kharif", "monsoon"],
                "description": "High-sugar content variety for processing"
            },
            "barley": {
                "nitrogen": {"min": 40, "max": 80, "optimal": 60},
                "phosphorus": {"min": 20, "max": 50, "optimal": 35},
                "potassium": {"min": 25, "max": 55, "optimal": 40},
                "temperature": {"min": 12, "max": 22, "optimal": 17},
                "humidity": {"min": 45, "max": 65, "optimal": 55},
                "ph": {"min": 6.0, "max": 7.8, "optimal": 6.9},
                "rainfall": {"min": 30, "max": 100, "optimal": 65},
                "season": ["rabi", "winter"],
                "description": "Hardy barley variety for cooler climates"
            },
            "soybean": {
                "nitrogen": {"min": 30, "max": 70, "optimal": 50},
                "phosphorus": {"min": 40, "max": 70, "optimal": 55},
                "potassium": {"min": 35, "max": 65, "optimal": 50},
                "temperature": {"min": 20, "max": 30, "optimal": 25},
                "humidity": {"min": 60, "max": 80, "optimal": 70},
                "ph": {"min": 6.0, "max": 7.2, "optimal": 6.6},
                "rainfall": {"min": 80, "max": 200, "optimal": 140},
                "season": ["kharif", "monsoon"],
                "description": "High-protein legume for oil and feed production"
            },
            "groundnut": {
                "nitrogen": {"min": 20, "max": 60, "optimal": 40},
                "phosphorus": {"min": 30, "max": 60, "optimal": 45},
                "potassium": {"min": 40, "max": 70, "optimal": 55},
                "temperature": {"min": 22, "max": 32, "optimal": 27},
                "humidity": {"min": 50, "max": 75, "optimal": 62},
                "ph": {"min": 5.5, "max": 7.5, "optimal": 6.5},
                "rainfall": {"min": 50, "max": 150, "optimal": 100},
                "season": ["kharif", "summer"],
                "description": "Nutritious groundnut variety for oil production"
            }
        }
    
    def _load_model(self) -> None:
        """
        Load the crop prediction model.
        In a real implementation, this would load a trained ML model.
        """
        try:
            # Placeholder for actual model loading
            # In practice, you would load a trained model here:
            # self.model = joblib.load('crop_model.pkl')
            # or
            # self.model = tf.keras.models.load_model('crop_model.h5')
            
            self.is_loaded = True
            logger.info("Crop prediction model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading crop prediction model: {str(e)}")
            self.is_loaded = False
    
    def is_ready(self) -> bool:
        """
        Check if the model is ready for predictions.
        
        Returns:
            bool: True if model is ready, False otherwise
        """
        return self.is_loaded
    
    def _calculate_crop_suitability(self, crop_data: Dict, input_params: Dict) -> float:
        """
        Calculate suitability score for a specific crop based on input parameters.
        
        Args:
            crop_data (Dict): Crop requirements and optimal conditions
            input_params (Dict): Input environmental parameters
            
        Returns:
            float: Suitability score (0-100)
        """
        score = 0.0
        total_weight = 0.0
        
        # Define parameter weights
        weights = {
            'nitrogen': 0.15,
            'phosphorus': 0.15,
            'potassium': 0.15,
            'temperature': 0.20,
            'humidity': 0.15,
            'ph': 0.10,
            'rainfall': 0.10
        }
        
        for param, weight in weights.items():
            if param in input_params and param in crop_data:
                value = input_params[param]
                requirements = crop_data[param]
                
                # Calculate parameter score based on proximity to optimal value
                if requirements['min'] <= value <= requirements['max']:
                    # Value is within acceptable range
                    optimal = requirements['optimal']
                    range_size = requirements['max'] - requirements['min']
                    
                    if range_size > 0:
                        distance_from_optimal = abs(value - optimal)
                        max_distance = max(optimal - requirements['min'], 
                                         requirements['max'] - optimal)
                        param_score = 100 * (1 - distance_from_optimal / max_distance)
                    else:
                        param_score = 100
                else:
                    # Value is outside acceptable range - penalize
                    if value < requirements['min']:
                        distance = requirements['min'] - value
                        penalty_factor = min(distance / requirements['min'], 1.0)
                    else:  # value > requirements['max']
                        distance = value - requirements['max']
                        penalty_factor = min(distance / requirements['max'], 1.0)
                    
                    param_score = max(0, 50 * (1 - penalty_factor))
                
                score += param_score * weight
                total_weight += weight
        
        # Normalize score
        return score / total_weight if total_weight > 0 else 0.0
    
    def predict(self, input_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict suitable crops based on input environmental parameters.
        
        Args:
            input_data (Dict[str, float]): Dictionary containing soil and environmental data
                Required keys: nitrogen, phosphorus, potassium, temperature, 
                             humidity, ph, rainfall
        
        Returns:
            Dict[str, Any]: Prediction results containing recommended crops and scores
        """
        try:
            start_time = datetime.now()
            
            if not self.is_ready():
                raise Exception("Crop prediction model is not ready")
            
            # Validate input data
            required_params = ['nitrogen', 'phosphorus', 'potassium', 'temperature',
                             'humidity', 'ph', 'rainfall']
            
            for param in required_params:
                if param not in input_data:
                    raise ValueError(f"Missing required parameter: {param}")
            
            # Calculate suitability scores for all crops
            crop_scores = []
            
            for crop_name, crop_data in self._crop_database.items():
                suitability_score = self._calculate_crop_suitability(crop_data, input_data)
                
                crop_scores.append({
                    'crop': crop_name,
                    'suitability_score': round(suitability_score, 2),
                    'confidence': 'High' if suitability_score >= 80 else 
                                'Medium' if suitability_score >= 60 else 'Low',
                    'description': crop_data.get('description', ''),
                    'optimal_conditions': {
                        param: crop_data[param]['optimal'] 
                        for param in required_params if param in crop_data
                    }
                })
            
            # Sort by suitability score
            crop_scores.sort(key=lambda x: x['suitability_score'], reverse=True)
            
            # Get top 3 recommendations
            top_crops = crop_scores[:3]
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare result
            result = {
                'success': True,
                'predictions': top_crops,
                'all_scores': crop_scores,
                'model_version': self.model_version,
                'processing_time': round(processing_time, 4),
                'timestamp': datetime.now().isoformat(),
                'recommendations': {
                    'primary': top_crops[0] if top_crops else None,
                    'alternatives': top_crops[1:3] if len(top_crops) > 1 else [],
                    'suitable_count': len([c for c in crop_scores if c['suitability_score'] >= 60])
                }
            }
            
            logger.info(f"Crop prediction completed for input: {input_data}")
            logger.info(f"Top recommendation: {top_crops[0]['crop'] if top_crops else 'None'}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in crop prediction: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'model_version': self.model_version,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_crop_info(self, crop_name: str) -> Optional[Dict]:
        """
        Get detailed information about a specific crop.
        
        Args:
            crop_name (str): Name of the crop
            
        Returns:
            Optional[Dict]: Crop information or None if not found
        """
        return self._crop_database.get(crop_name.lower())
    
    def get_all_crops(self) -> List[str]:
        """
        Get list of all available crops in the database.
        
        Returns:
            List[str]: List of crop names
        """
        return list(self._crop_database.keys())
    
    def validate_input(self, input_data: Dict) -> Dict[str, Any]:
        """
        Validate input data for crop prediction.
        
        Args:
            input_data (Dict): Input parameters to validate
            
        Returns:
            Dict[str, Any]: Validation result with errors if any
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        required_params = ['nitrogen', 'phosphorus', 'potassium', 'temperature',
                          'humidity', 'ph', 'rainfall']
        
        # Check for missing parameters
        for param in required_params:
            if param not in input_data:
                validation_result['errors'].append(f"Missing required parameter: {param}")
                validation_result['is_valid'] = False
        
        if validation_result['is_valid']:
            # Validate parameter ranges
            ranges = {
                'nitrogen': (0, 200),
                'phosphorus': (0, 200),
                'potassium': (0, 200),
                'temperature': (-10, 50),
                'humidity': (0, 100),
                'ph': (0, 14),
                'rainfall': (0, 500)
            }
            
            for param, (min_val, max_val) in ranges.items():
                if param in input_data:
                    value = input_data[param]
                    if not isinstance(value, (int, float)):
                        validation_result['errors'].append(f"{param} must be a number")
                        validation_result['is_valid'] = False
                    elif value < min_val or value > max_val:
                        validation_result['errors'].append(
                            f"{param} must be between {min_val} and {max_val}"
                        )
                        validation_result['is_valid'] = False
        
        return validation_result
    
    def export_model_info(self) -> Dict[str, Any]:
        """
        Export model information for monitoring and debugging.
        
        Returns:
            Dict[str, Any]: Model information
        """
        return {
            'model_type': 'crop_predictor',
            'version': self.model_version,
            'is_loaded': self.is_loaded,
            'available_crops': len(self._crop_database),
            'crop_list': list(self._crop_database.keys()),
            'last_updated': datetime.now().isoformat()
        }