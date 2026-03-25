"""
Fertilizer Recommendation Model
==============================
Uses Google's Gemini API to recommend fertilizers.
"""

import logging
from typing import Dict, Any
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Configure logging
logger = logging.getLogger(__name__)

class FertilizerRecommender:
    def __init__(self, model_version: str = "gemini-pro"):
        self.model_version = model_version
        self.is_loaded = False
        self._load_model()
        
    def _load_model(self) -> None:
        try:
            load_dotenv()
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.error("GEMINI_API_KEY not found in (.env)")
                return
            
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel("gemini-pro")
            self.is_loaded = True
            logger.info("Gemini Fertilizer recommendation model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Gemini model: {str(e)}")
            self.is_loaded = False
    
    def is_ready(self) -> bool:
        return self.is_loaded
    
    def recommend(self, input_data: Dict[str, Any]) -> str:
        """Predict suitable fertilizer using Gemini API."""
        try:
            if not self.is_ready():
                return "Error: Gemini API not configured or loaded"
                
            prompt = (
                f"You are an agricultural expert AI. Based on the following soil "
                f"conditions and the crop type being grown, recommend the best type of fertilizer "
                f"and provide a short, 1 or 2 sentence reason why.\n"
                f"Conditions:\n"
                f"Nitrogen: {input_data.get('nitrogen')}\n"
                f"Phosphorus: {input_data.get('phosphorus')}\n"
                f"Potassium: {input_data.get('potassium')}\n"
                f"Crop Type: {input_data.get('crop_type')}\n"
            )
            
            response = self.model.generate_content(prompt)
            result_text = response.text.strip()
            
            logger.info(f"Gemini Fertilizer recommendation completed: {result_text}")
            return result_text
            
        except Exception as e:
            logger.error(f"Error in Gemini fertilizer recommendation: {str(e)}")
            return f"Error connecting to Gemini: {str(e)}"