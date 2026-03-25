"""
Crop Prediction Model
====================
Uses Google's Gemini API to recommend crops.
"""

import logging
from typing import Dict, Any
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Configure logging
logger = logging.getLogger(__name__)

class CropPredictor:
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
            self.model = genai.GenerativeModel("gemini-2.5-flash")
            self.is_loaded = True
            logger.info("Gemini Crop prediction model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Gemini model: {str(e)}")
            self.is_loaded = False
    
    def is_ready(self) -> bool:
        return self.is_loaded
    
    def predict(self, input_data: Dict[str, float]) -> str:
        """Predict suitable crops using Gemini API."""
        try:
            if not self.is_ready():
                return "Error: Gemini API not configured or loaded"
                
            prompt = (
                f"You are an agricultural expert AI. Based on the following soil and environmental "
                f"conditions, recommend the absolute best single crop to plant. Provide just the name of the crop,, or a very brief 1-sentence explanation if necessary.\n"
                f"Conditions:\n"
                f"Nitrogen: {input_data.get('nitrogen')}\n"
                f"Phosphorus: {input_data.get('phosphorus')}\n"
                f"Potassium: {input_data.get('potassium')}\n"
                f"Temperature: {input_data.get('temperature')}°C\n"
                f"Humidity: {input_data.get('humidity')}%\n"
                f"pH: {input_data.get('ph')}\n"
                f"Rainfall: {input_data.get('rainfall')}mm"
            )
            
            response = self.model.generate_content(prompt)
            result_text = response.text.strip()
            
            logger.info(f"Gemini Crop prediction completed: {result_text}")
            return result_text
            
        except Exception as e:
            logger.error(f"Error in Gemini crop prediction: {str(e)}")
            return f"Error connecting to Gemini: {str(e)}"