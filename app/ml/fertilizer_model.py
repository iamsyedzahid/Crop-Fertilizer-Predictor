"""
Fertilizer Recommendation Model
==============================
This module contains the FertilizerRecommender class that provides fertilizer
recommendations based on soil nutrient levels and crop type. Currently implements
rule-based logic as a placeholder for machine learning models.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
import os
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class FertilizerRecommender:
    """
    Fertilizer recommendation class that suggests appropriate fertilizers
    based on soil nutrient levels and crop type using rule-based logic.
    
    This is a placeholder implementation that can be replaced with
    actual machine learning models (scikit-learn, TensorFlow, etc.)
    """
    
    def __init__(self, model_version: str = "1.0.0"):
        """
        Initialize the FertilizerRecommender.
        
        Args:
            model_version (str): Version of the recommendation model
        """
        self.model_version = model_version
        self.is_loaded = False
        self._fertilizer_database = self._initialize_fertilizer_database()
        self._crop_requirements = self._initialize_crop_requirements()
        self._load_model()
        
        logger.info(f"FertilizerRecommender initialized with version {model_version}")
    
    def _initialize_fertilizer_database(self) -> Dict[str, Dict]:
        """
        Initialize the fertilizer database with composition and application rates.
        In a real implementation, this would be loaded from a database or file.
        
        Returns:
            Dict: Fertilizer database with composition and usage information
        """
        return {
            "urea": {
                "composition": {"nitrogen": 46, "phosphorus": 0, "potassium": 0},
                "application_rate": {"min": 50, "max": 200, "unit": "kg/hectare"},
                "cost_per_kg": 25.0,
                "season": ["kharif", "rabi", "summer"],
                "soil_ph": {"min": 6.0, "max": 8.0},
                "description": "High nitrogen fertilizer for vegetative growth",
                "benefits": ["Promotes leafy growth", "Quick nitrogen source", "Water soluble"],
                "precautions": ["Avoid over-application", "Apply in split doses"]
            },
            "dap": {  # Diammonium Phosphate
                "composition": {"nitrogen": 18, "phosphorus": 46, "potassium": 0},
                "application_rate": {"min": 75, "max": 150, "unit": "kg/hectare"},
                "cost_per_kg": 35.0,
                "season": ["kharif", "rabi"],
                "soil_ph": {"min": 6.0, "max": 7.5},
                "description": "High phosphorus fertilizer for root development",
                "benefits": ["Promotes root growth", "Enhances flowering", "Good starter fertilizer"],
                "precautions": ["Apply at planting", "Don't mix with alkaline fertilizers"]
            },
            "mop": {  # Muriate of Potash
                "composition": {"nitrogen": 0, "phosphorus": 0, "potassium": 60},
                "application_rate": {"min": 40, "max": 120, "unit": "kg/hectare"},
                "cost_per_kg": 22.0,
                "season": ["kharif", "rabi", "summer"],
                "soil_ph": {"min": 5.5, "max": 8.5},
                "description": "High potassium fertilizer for fruit development",
                "benefits": ["Improves fruit quality", "Enhances disease resistance", "Improves water use efficiency"],
                "precautions": ["Avoid chloride-sensitive crops", "Apply before flowering"]
            },
            "npk_19_19_19": {
                "composition": {"nitrogen": 19, "phosphorus": 19, "potassium": 19},
                "application_rate": {"min": 100, "max": 250, "unit": "kg/hectare"},
                "cost_per_kg": 30.0,
                "season": ["kharif", "rabi", "summer"],
                "soil_ph": {"min": 6.0, "max": 7.5},
                "description": "Balanced NPK fertilizer for general nutrition",
                "benefits": ["Complete nutrition", "Easy application", "Suitable for most crops"],
                "precautions": ["Monitor soil nutrient levels", "Adjust based on soil test"]
            },
            "npk_20_20_0": {
                "composition": {"nitrogen": 20, "phosphorus": 20, "potassium": 0},
                "application_rate": {"min": 80, "max": 180, "unit": "kg/hectare"},
                "cost_per_kg": 28.0,
                "season": ["kharif", "rabi"],
                "soil_ph": {"min": 6.0, "max": 7.5},
                "description": "NP fertilizer for early growth stages",
                "benefits": ["Good for seedling growth", "Promotes early development", "Cost-effective"],
                "precautions": ["Supplement with potassium", "Use with K sources"]
            },
            "calcium_ammonium_nitrate": {
                "composition": {"nitrogen": 26, "phosphorus": 0, "potassium": 0, "calcium": 10},
                "application_rate": {"min": 60, "max": 180, "unit": "kg/hectare"},
                "cost_per_kg": 32.0,
                "season": ["kharif", "rabi", "summer"],
                "soil_ph": {"min": 5.5, "max": 7.5},
                "description": "Nitrogen fertilizer with calcium for acidic soils",
                "benefits": ["Provides nitrogen and calcium", "Reduces soil acidity", "Improves soil structure"],
                "precautions": ["Store in cool, dry place", "Avoid mixing with organic matter"]
            },
            "single_super_phosphate": {
                "composition": {"nitrogen": 0, "phosphorus": 16, "potassium": 0, "sulfur": 12},
                "application_rate": {"min": 100, "max": 200, "unit": "kg/hectare"},
                "cost_per_kg": 18.0,
                "season": ["kharif", "rabi"],
                "soil_ph": {"min": 6.0, "max": 8.0},
                "description": "Phosphorus fertilizer with sulfur",
                "benefits": ["Provides phosphorus and sulfur", "Cost-effective", "Good for legumes"],
                "precautions": ["Apply before sowing", "Mix well with soil"]
            },
            "potassium_sulfate": {
                "composition": {"nitrogen": 0, "phosphorus": 0, "potassium": 50, "sulfur": 18},
                "application_rate": {"min": 50, "max": 100, "unit": "kg/hectare"},
                "cost_per_kg": 40.0,
                "season": ["kharif", "rabi", "summer"],
                "soil_ph": {"min": 5.5, "max": 8.0},
                "description": "Premium potassium fertilizer with sulfur",
                "benefits": ["Chloride-free potassium", "Provides sulfur", "Suitable for sensitive crops"],
                "precautions": ["Higher cost", "Use for high-value crops"]
            },
            "organic_compost": {
                "composition": {"nitrogen": 2, "phosphorus": 1, "potassium": 2, "organic_matter": 30},
                "application_rate": {"min": 2000, "max": 5000, "unit": "kg/hectare"},
                "cost_per_kg": 5.0,
                "season": ["kharif", "rabi", "summer"],
                "soil_ph": {"min": 5.0, "max": 8.5},
                "description": "Organic fertilizer for soil health",
                "benefits": ["Improves soil structure", "Slow-release nutrients", "Enhances microbial activity"],
                "precautions": ["Apply well in advance", "Ensure proper decomposition"]
            }
        }
    
    def _initialize_crop_requirements(self) -> Dict[str, Dict]:
        """
        Initialize crop-specific nutrient requirements.
        
        Returns:
            Dict: Crop nutrient requirements database
        """
        return {
            "rice": {
                "nutrient_uptake": {"nitrogen": 120, "phosphorus": 60, "potassium": 80},
                "growth_stages": {
                    "vegetative": {"nitrogen": 50, "phosphorus": 30, "potassium": 20},
                    "reproductive": {"nitrogen": 40, "phosphorus": 20, "potassium": 35},
                    "maturity": {"nitrogen": 30, "phosphorus": 10, "potassium": 25}
                },
                "critical_nutrients": ["nitrogen", "phosphorus"],
                "preferred_fertilizers": ["urea", "dap", "npk_19_19_19"]
            },
            "wheat": {
                "nutrient_uptake": {"nitrogen": 100, "phosphorus": 50, "potassium": 60},
                "growth_stages": {
                    "vegetative": {"nitrogen": 40, "phosphorus": 25, "potassium": 20},
                    "reproductive": {"nitrogen": 35, "phosphorus": 15, "potassium": 25},
                    "maturity": {"nitrogen": 25, "phosphorus": 10, "potassium": 15}
                },
                "critical_nutrients": ["nitrogen", "phosphorus"],
                "preferred_fertilizers": ["urea", "dap", "npk_20_20_0"]
            },
            "maize": {
                "nutrient_uptake": {"nitrogen": 150, "phosphorus": 70, "potassium": 100},
                "growth_stages": {
                    "vegetative": {"nitrogen": 60, "phosphorus": 35, "potassium": 30},
                    "reproductive": {"nitrogen": 50, "phosphorus": 25, "potassium": 40},
                    "maturity": {"nitrogen": 40, "phosphorus": 10, "potassium": 30}
                },
                "critical_nutrients": ["nitrogen", "potassium"],
                "preferred_fertilizers": ["urea", "dap", "mop", "npk_19_19_19"]
            },
            "cotton": {
                "nutrient_uptake": {"nitrogen": 120, "phosphorus": 60, "potassium": 120},
                "growth_stages": {
                    "vegetative": {"nitrogen": 40, "phosphorus": 30, "potassium": 30},
                    "reproductive": {"nitrogen": 50, "phosphorus": 20, "potassium": 50},
                    "maturity": {"nitrogen": 30, "phosphorus": 10, "potassium": 40}
                },
                "critical_nutrients": ["potassium", "nitrogen"],
                "preferred_fertilizers": ["urea", "mop", "npk_19_19_19", "potassium_sulfate"]
            },
            "sugarcane": {
                "nutrient_uptake": {"nitrogen": 200, "phosphorus": 80, "potassium": 150},
                "growth_stages": {
                    "vegetative": {"nitrogen": 80, "phosphorus": 40, "potassium": 50},
                    "reproductive": {"nitrogen": 70, "phosphorus": 30, "potassium": 60},
                    "maturity": {"nitrogen": 50, "phosphorus": 10, "potassium": 40}
                },
                "critical_nutrients": ["nitrogen", "potassium"],
                "preferred_fertilizers": ["urea", "dap", "mop", "npk_19_19_19"]
            },
            "soybean": {
                "nutrient_uptake": {"nitrogen": 80, "phosphorus": 60, "potassium": 80},
                "growth_stages": {
                    "vegetative": {"nitrogen": 20, "phosphorus": 30, "potassium": 25},
                    "reproductive": {"nitrogen": 30, "phosphorus": 20, "potassium": 35},
                    "maturity": {"nitrogen": 30, "phosphorus": 10, "potassium": 20}
                },
                "critical_nutrients": ["phosphorus", "potassium"],
                "preferred_fertilizers": ["dap", "single_super_phosphate", "mop", "npk_19_19_19"]
            },
            "groundnut": {
                "nutrient_uptake": {"nitrogen": 60, "phosphorus": 50, "potassium": 70},
                "growth_stages": {
                    "vegetative": {"nitrogen": 15, "phosphorus": 25, "potassium": 20},
                    "reproductive": {"nitrogen": 25, "phosphorus": 15, "potassium": 30},
                    "maturity": {"nitrogen": 20, "phosphorus": 10, "potassium": 20}
                },
                "critical_nutrients": ["phosphorus", "potassium", "calcium"],
                "preferred_fertilizers": ["dap", "single_super_phosphate", "mop", "calcium_ammonium_nitrate"]
            },
            "barley": {
                "nutrient_uptake": {"nitrogen": 80, "phosphorus": 40, "potassium": 50},
                "growth_stages": {
                    "vegetative": {"nitrogen": 35, "phosphorus": 20, "potassium": 15},
                    "reproductive": {"nitrogen": 30, "phosphorus": 15, "potassium": 20},
                    "maturity": {"nitrogen": 15, "phosphorus": 5, "potassium": 15}
                },
                "critical_nutrients": ["nitrogen", "phosphorus"],
                "preferred_fertilizers": ["urea", "dap", "npk_20_20_0"]
            }
        }
    
    def _load_model(self) -> None:
        """
        Load the fertilizer recommendation model.
        In a real implementation, this would load a trained ML model.
        """
        try:
            # Placeholder for actual model loading
            # In practice, you would load a trained model here:
            # self.model = joblib.load('fertilizer_model.pkl')
            # or
            # self.model = tf.keras.models.load_model('fertilizer_model.h5')
            
            self.is_loaded = True
            logger.info("Fertilizer recommendation model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading fertilizer recommendation model: {str(e)}")
            self.is_loaded = False
    
    def is_ready(self) -> bool:
        """
        Check if the model is ready for recommendations.
        
        Returns:
            bool: True if model is ready, False otherwise
        """
        return self.is_loaded
    
    def _calculate_nutrient_deficiency(self, current_levels: Dict, crop_requirements: Dict) -> Dict[str, float]:
        """
        Calculate nutrient deficiency based on current soil levels and crop requirements.
        
        Args:
            current_levels (Dict): Current soil nutrient levels
            crop_requirements (Dict): Crop nutrient requirements
            
        Returns:
            Dict[str, float]: Deficiency levels for each nutrient
        """
        deficiency = {}
        required_nutrients = crop_requirements.get('nutrient_uptake', {})
        
        for nutrient in ['nitrogen', 'phosphorus', 'potassium']:
            current = current_levels.get(nutrient, 0)
            required = required_nutrients.get(nutrient, 0)
            
            # Calculate deficiency as percentage of required amount
            if required > 0:
                # Assume current levels represent available nutrients
                # Deficiency calculation: max(0, required - current * availability_factor)
                availability_factor = 0.7  # Assume 70% availability
                available = current * availability_factor
                deficiency[nutrient] = max(0, required - available)
            else:
                deficiency[nutrient] = 0
        
        return deficiency
    
    def _select_fertilizers(self, deficiency: Dict, crop_type: str) -> List[Dict]:
        """
        Select appropriate fertilizers based on nutrient deficiency and crop type.
        
        Args:
            deficiency (Dict): Nutrient deficiency levels
            crop_type (str): Type of crop
            
        Returns:
            List[Dict]: Selected fertilizers with application rates
        """
        recommendations = []
        crop_data = self._crop_requirements.get(crop_type, {})
        preferred_fertilizers = crop_data.get('preferred_fertilizers', [])
        
        # Sort nutrients by deficiency level (descending)
        sorted_deficiencies = sorted(deficiency.items(), key=lambda x: x[1], reverse=True)
        
        selected_fertilizers = set()
        
        for nutrient, deficit in sorted_deficiencies:
            if deficit <= 0:
                continue
            
            # Find best fertilizer for this nutrient
            best_fertilizer = None
            best_score = 0
            
            for fert_name, fert_data in self._fertilizer_database.items():
                if fert_name in selected_fertilizers:
                    continue
                
                composition = fert_data['composition']
                nutrient_content = composition.get(nutrient, 0)
                
                if nutrient_content > 0:
                    # Calculate score based on nutrient content and preference
                    score = nutrient_content
                    if fert_name in preferred_fertilizers:
                        score *= 1.2  # Preference bonus
                    
                    # Consider cost effectiveness
                    cost_per_unit = fert_data['cost_per_kg'] / nutrient_content
                    score /= (cost_per_unit / 10)  # Normalize cost factor
                    
                    if score > best_score:
                        best_fertilizer = fert_name
                        best_score = score
            
            if best_fertilizer:
                fert_data = self._fertilizer_database[best_fertilizer]
                composition = fert_data['composition']
                
                # Calculate application rate based on deficiency
                nutrient_content = composition.get(nutrient, 0)
                if nutrient_content > 0:
                    required_fertilizer = (deficit / nutrient_content) * 100  # kg/hectare
                    
                    # Constrain within application rate limits
                    min_rate = fert_data['application_rate']['min']
                    max_rate = fert_data['application_rate']['max']
                    application_rate = max(min_rate, min(required_fertilizer, max_rate))
                    
                    # Calculate actual nutrient supply
                    actual_supply = {}
                    for nut in ['nitrogen', 'phosphorus', 'potassium']:
                        if nut in composition:
                            actual_supply[nut] = (application_rate * composition[nut]) / 100
                    
                    recommendations.append({
                        'fertilizer': best_fertilizer,
                        'application_rate': round(application_rate, 1),
                        'unit': fert_data['application_rate']['unit'],
                        'cost_per_hectare': round(application_rate * fert_data['cost_per_kg'], 2),
                        'nutrient_supply': actual_supply,
                        'composition': composition.copy(),
                        'description': fert_data['description'],
                        'benefits': fert_data['benefits'],
                        'precautions': fert_data['precautions']
                    })
                    
                    selected_fertilizers.add(best_fertilizer)
                    
                    # Update deficiency levels
                    for nut, supply in actual_supply.items():
                        if nut in deficiency:
                            deficiency[nut] = max(0, deficiency[nut] - supply)
        
        # If no specific fertilizers selected, recommend balanced fertilizer
        if not recommendations and any(deficit > 0 for deficit in deficiency.values()):
            balanced_fert = 'npk_19_19_19'
            if balanced_fert in self._fertilizer_database:
                fert_data = self._fertilizer_database[balanced_fert]
                application_rate = (fert_data['application_rate']['min'] + 
                                  fert_data['application_rate']['max']) / 2
                
                composition = fert_data['composition']
                actual_supply = {}
                for nut in ['nitrogen', 'phosphorus', 'potassium']:
                    if nut in composition:
                        actual_supply[nut] = (application_rate * composition[nut]) / 100
                
                recommendations.append({
                    'fertilizer': balanced_fert,
                    'application_rate': round(application_rate, 1),
                    'unit': fert_data['application_rate']['unit'],
                    'cost_per_hectare': round(application_rate * fert_data['cost_per_kg'], 2),
                    'nutrient_supply': actual_supply,
                    'composition': composition.copy(),
                    'description': fert_data['description'],
                    'benefits': fert_data['benefits'],
                    'precautions': fert_data['precautions']
                })
        
        return recommendations
    
    def recommend(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recommend fertilizers based on soil nutrient levels and crop type.
        
        Args:
            input_data (Dict[str, Any]): Dictionary containing soil data and crop type
                Required keys: nitrogen, phosphorus, potassium, crop_type
        
        Returns:
            Dict[str, Any]: Fertilizer recommendations with application rates and costs
        """
        try:
            start_time = datetime.now()
            
            if not self.is_ready():
                raise Exception("Fertilizer recommendation model is not ready")
            
            # Validate input data
            required_params = ['nitrogen', 'phosphorus', 'potassium', 'crop_type']
            
            for param in required_params:
                if param not in input_data:
                    raise ValueError(f"Missing required parameter: {param}")
            
            crop_type = input_data['crop_type'].lower().strip()
            
            # Check if crop is supported
            if crop_type not in self._crop_requirements:
                available_crops = list(self._crop_requirements.keys())
                raise ValueError(f"Unsupported crop type: {crop_type}. Available crops: {available_crops}")
            
            # Extract nutrient levels
            current_levels = {
                'nitrogen': float(input_data['nitrogen']),
                'phosphorus': float(input_data['phosphorus']),
                'potassium': float(input_data['potassium'])
            }
            
            # Get crop requirements
            crop_requirements = self._crop_requirements[crop_type]
            
            # Calculate nutrient deficiency
            deficiency = self._calculate_nutrient_deficiency(current_levels, crop_requirements)
            
            # Select appropriate fertilizers
            fertilizer_recommendations = self._select_fertilizers(deficiency, crop_type)
            
            # Calculate total cost
            total_cost = sum(rec['cost_per_hectare'] for rec in fertilizer_recommendations)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare result
            result = {
                'success': True,
                'crop_type': crop_type,
                'current_soil_levels': current_levels,
                'crop_requirements': crop_requirements['nutrient_uptake'],
                'nutrient_deficiency': deficiency,
                'fertilizer_recommendations': fertilizer_recommendations,
                'total_estimated_cost': round(total_cost, 2),
                'currency': 'INR',
                'application_timing': self._get_application_timing(crop_type),
                'general_recommendations': self._get_general_recommendations(crop_type, deficiency),
                'model_version': self.model_version,
                'processing_time': round(processing_time, 4),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Fertilizer recommendation completed for crop: {crop_type}")
            logger.info(f"Number of fertilizers recommended: {len(fertilizer_recommendations)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in fertilizer recommendation: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'model_version': self.model_version,
                'timestamp': datetime.now().isoformat()
            }
    
    def _get_application_timing(self, crop_type: str) -> Dict[str, str]:
        """
        Get fertilizer application timing recommendations for a crop.
        
        Args:
            crop_type (str): Type of crop
            
        Returns:
            Dict[str, str]: Application timing recommendations
        """
        timing_recommendations = {
            'rice': {
                'basal': 'Apply 50% N, full P, and 50% K at transplanting',
                'vegetative': 'Apply 25% N and 25% K at tillering stage',
                'reproductive': 'Apply remaining 25% N and 25% K at panicle initiation'
            },
            'wheat': {
                'basal': 'Apply 50% N, full P, and full K at sowing',
                'vegetative': 'Apply 25% N at first irrigation (20-25 DAS)',
                'reproductive': 'Apply remaining 25% N at second irrigation (40-45 DAS)'
            },
            'maize': {
                'basal': 'Apply 25% N, full P, and 50% K at sowing',
                'vegetative': 'Apply 50% N and 25% K at knee-high stage',
                'reproductive': 'Apply remaining 25% N and 25% K at tasseling'
            }
        }
        
        return timing_recommendations.get(crop_type, {
            'basal': 'Apply 50% N, full P, and 50% K at planting',
            'top_dress': 'Apply remaining N and K in 2-3 splits during growing season'
        })
    
    def _get_general_recommendations(self, crop_type: str, deficiency: Dict) -> List[str]:
        """
        Get general fertilizer application recommendations.
        
        Args:
            crop_type (str): Type of crop
            deficiency (Dict): Nutrient deficiency levels
            
        Returns:
            List[str]: List of general recommendations
        """
        recommendations = []
        
        # Soil test recommendations
        recommendations.append("Conduct soil test every 2-3 years for precise fertilizer planning")
        
        # Organic matter
        recommendations.append("Apply 5-10 tons of well-decomposed organic matter per hectare annually")
        
        # Micronutrients
        if crop_type in ['rice', 'wheat', 'maize']:
            recommendations.append("Consider zinc sulfate application if deficiency symptoms appear")
        
        # Specific recommendations based on deficiency
        if deficiency.get('nitrogen', 0) > 50:
            recommendations.append("Split nitrogen application to reduce leaching losses")
        
        if deficiency.get('phosphorus', 0) > 30:
            recommendations.append("Apply phosphorus fertilizers at or before planting for better uptake")
        
        if deficiency.get('potassium', 0) > 40:
            recommendations.append("Ensure adequate potassium for improved drought tolerance and disease resistance")
        
        # Water management
        recommendations.append("Maintain optimal soil moisture for better nutrient uptake")
        
        return recommendations
    
    def get_fertilizer_info(self, fertilizer_name: str) -> Optional[Dict]:
        """
        Get detailed information about a specific fertilizer.
        
        Args:
            fertilizer_name (str): Name of the fertilizer
            
        Returns:
            Optional[Dict]: Fertilizer information or None if not found
        """
        return self._fertilizer_database.get(fertilizer_name.lower().replace(' ', '_'))
    
    def get_all_fertilizers(self) -> List[str]:
        """
        Get list of all available fertilizers in the database.
        
        Returns:
            List[str]: List of fertilizer names
        """
        return list(self._fertilizer_database.keys())
    
    def get_supported_crops(self) -> List[str]:
        """
        Get list of all supported crops.
        
        Returns:
            List[str]: List of supported crop names
        """
        return list(self._crop_requirements.keys())
    
    def validate_input(self, input_data: Dict) -> Dict[str, Any]:
        """
        Validate input data for fertilizer recommendation.
        
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
        
        required_params = ['nitrogen', 'phosphorus', 'potassium', 'crop_type']
        
        # Check for missing parameters
        for param in required_params:
            if param not in input_data:
                validation_result['errors'].append(f"Missing required parameter: {param}")
                validation_result['is_valid'] = False
        
        if validation_result['is_valid']:
            # Validate numeric parameter ranges
            ranges = {
                'nitrogen': (0, 200),
                'phosphorus': (0, 200),
                'potassium': (0, 200)
            }
            
            for param, (min_val, max_val) in ranges.items():
                if param in input_data:
                    try:
                        value = float(input_data[param])
                        if value < min_val or value > max_val:
                            validation_result['errors'].append(
                                f"{param} must be between {min_val} and {max_val}"
                            )
                            validation_result['is_valid'] = False
                    except (ValueError, TypeError):
                        validation_result['errors'].append(f"{param} must be a number")
                        validation_result['is_valid'] = False
            
            # Validate crop type
            crop_type = input_data.get('crop_type', '').lower().strip()
            if crop_type and crop_type not in self._crop_requirements:
                available_crops = list(self._crop_requirements.keys())
                validation_result['errors'].append(
                    f"Unsupported crop type: {crop_type}. Available: {', '.join(available_crops)}"
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
            'model_type': 'fertilizer_recommender',
            'version': self.model_version,
            'is_loaded': self.is_loaded,
            'available_fertilizers': len(self._fertilizer_database),
            'fertilizer_list': list(self._fertilizer_database.keys()),
            'supported_crops': len(self._crop_requirements),
            'crop_list': list(self._crop_requirements.keys()),
            'last_updated': datetime.now().isoformat()
        }