"""
SQLAlchemy Database Models
=========================
This module defines the database models for the PAI Flask application.
It includes models for storing prediction history and user data.
"""

from app import db
from datetime import datetime
from sqlalchemy import func
import json

class PredictionHistory(db.Model):
    """
    Model for storing prediction history.
    Tracks all crop predictions and fertilizer recommendations made by users.
    """
    __tablename__ = 'prediction_history'
    
    # Primary key
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    
    # Prediction metadata
    prediction_type = db.Column(db.String(20), nullable=False, index=True)  # 'crop' or 'fertilizer'
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Input and output data (stored as JSON strings)
    input_data = db.Column(db.Text, nullable=False)  # JSON string of input parameters
    result = db.Column(db.Text, nullable=False)      # JSON string of prediction/recommendation
    
    # Optional user tracking (for future enhancement)
    user_id = db.Column(db.String(100), nullable=True, index=True)  # Session ID or user identifier
    ip_address = db.Column(db.String(45), nullable=True)            # IPv4 or IPv6 address
    
    # Performance metrics
    processing_time = db.Column(db.Float, nullable=True)  # Time taken for prediction in seconds
    model_version = db.Column(db.String(20), nullable=True)  # Version of ML model used
    
    def __init__(self, prediction_type, input_data, result, user_id=None, 
                 ip_address=None, processing_time=None, model_version=None):
        """
        Initialize a new prediction history record.
        
        Args:
            prediction_type (str): Type of prediction ('crop' or 'fertilizer')
            input_data (str or dict): Input parameters used for prediction
            result (str or dict): Prediction or recommendation result
            user_id (str, optional): User or session identifier
            ip_address (str, optional): Client IP address
            processing_time (float, optional): Processing time in seconds
            model_version (str, optional): ML model version used
        """
        self.prediction_type = prediction_type
        self.input_data = input_data if isinstance(input_data, str) else json.dumps(input_data)
        self.result = result if isinstance(result, str) else json.dumps(result)
        self.user_id = user_id
        self.ip_address = ip_address
        self.processing_time = processing_time
        self.model_version = model_version
    
    def get_input_data_dict(self):
        """
        Parse input_data JSON string to dictionary.
        
        Returns:
            dict: Parsed input parameters
        """
        try:
            return json.loads(self.input_data)
        except (json.JSONDecodeError, TypeError):
            return {}
    
    def get_result_dict(self):
        """
        Parse result JSON string to dictionary.
        
        Returns:
            dict: Parsed prediction/recommendation result
        """
        try:
            return json.loads(self.result)
        except (json.JSONDecodeError, TypeError):
            return {}
    
    def __repr__(self):
        """String representation of PredictionHistory object"""
        return f'<PredictionHistory {self.id}: {self.prediction_type} at {self.timestamp}>'
    
    def to_dict(self):
        """
        Convert model instance to dictionary for JSON serialization.
        
        Returns:
            dict: Dictionary representation of the prediction record
        """
        return {
            'id': self.id,
            'prediction_type': self.prediction_type,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'input_data': self.get_input_data_dict(),
            'result': self.get_result_dict(),
            'user_id': self.user_id,
            'ip_address': self.ip_address,
            'processing_time': self.processing_time,
            'model_version': self.model_version
        }
    
    @staticmethod
    def get_recent_predictions(limit=10):
        """
        Get recent predictions ordered by timestamp.
        
        Args:
            limit (int): Maximum number of records to return
            
        Returns:
            list: List of recent PredictionHistory objects
        """
        return PredictionHistory.query.order_by(
            PredictionHistory.timestamp.desc()
        ).limit(limit).all()
    
    @staticmethod
    def get_predictions_by_type(prediction_type, limit=None):
        """
        Get predictions filtered by type.
        
        Args:
            prediction_type (str): Type of prediction ('crop' or 'fertilizer')
            limit (int, optional): Maximum number of records to return
            
        Returns:
            list: List of filtered PredictionHistory objects
        """
        query = PredictionHistory.query.filter_by(
            prediction_type=prediction_type
        ).order_by(PredictionHistory.timestamp.desc())
        
        if limit:
            query = query.limit(limit)
            
        return query.all()
    
    @staticmethod
    def get_daily_stats():
        """
        Get daily prediction statistics.
        
        Returns:
            dict: Statistics including total predictions and breakdown by type
        """
        today = datetime.utcnow().date()
        
        # Total predictions today
        total_today = PredictionHistory.query.filter(
            func.date(PredictionHistory.timestamp) == today
        ).count()
        
        # Crop predictions today
        crop_today = PredictionHistory.query.filter(
            func.date(PredictionHistory.timestamp) == today,
            PredictionHistory.prediction_type == 'crop'
        ).count()
        
        # Fertilizer predictions today
        fertilizer_today = PredictionHistory.query.filter(
            func.date(PredictionHistory.timestamp) == today,
            PredictionHistory.prediction_type == 'fertilizer'
        ).count()
        
        return {
            'total_today': total_today,
            'crop_today': crop_today,
            'fertilizer_today': fertilizer_today,
            'date': today.isoformat()
        }


class SystemMetrics(db.Model):
    """
    Model for storing system performance metrics and application statistics.
    Used for monitoring and analytics purposes.
    """
    __tablename__ = 'system_metrics'
    
    # Primary key
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    
    # Timestamp
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Metric information
    metric_name = db.Column(db.String(50), nullable=False, index=True)  # e.g., 'response_time', 'error_count'
    metric_value = db.Column(db.Float, nullable=False)                  # Numeric value of the metric
    metric_unit = db.Column(db.String(20), nullable=True)               # Unit of measurement (ms, count, etc.)
    
    # Additional context
    endpoint = db.Column(db.String(100), nullable=True, index=True)     # API endpoint if applicable
    user_id = db.Column(db.String(100), nullable=True, index=True)      # User or session identifier
    additional_data = db.Column(db.Text, nullable=True)                 # JSON string for extra data
    
    def __init__(self, metric_name, metric_value, metric_unit=None, 
                 endpoint=None, user_id=None, additional_data=None):
        """
        Initialize a new system metric record.
        
        Args:
            metric_name (str): Name of the metric
            metric_value (float): Value of the metric
            metric_unit (str, optional): Unit of measurement
            endpoint (str, optional): API endpoint if applicable
            user_id (str, optional): User or session identifier
            additional_data (str or dict, optional): Additional context data
        """
        self.metric_name = metric_name
        self.metric_value = metric_value
        self.metric_unit = metric_unit
        self.endpoint = endpoint
        self.user_id = user_id
        self.additional_data = additional_data if isinstance(additional_data, str) else json.dumps(additional_data)
    
    def get_additional_data_dict(self):
        """
        Parse additional_data JSON string to dictionary.
        
        Returns:
            dict: Parsed additional data
        """
        try:
            return json.loads(self.additional_data) if self.additional_data else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    
    def __repr__(self):
        """String representation of SystemMetrics object"""
        return f'<SystemMetrics {self.id}: {self.metric_name}={self.metric_value} at {self.timestamp}>'
    
    def to_dict(self):
        """
        Convert model instance to dictionary for JSON serialization.
        
        Returns:
            dict: Dictionary representation of the metric record
        """
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'metric_unit': self.metric_unit,
            'endpoint': self.endpoint,
            'user_id': self.user_id,
            'additional_data': self.get_additional_data_dict()
        }
    
    @staticmethod
    def record_metric(metric_name, metric_value, **kwargs):
        """
        Convenience method to record a new metric.
        
        Args:
            metric_name (str): Name of the metric
            metric_value (float): Value of the metric
            **kwargs: Additional arguments for the metric
            
        Returns:
            SystemMetrics: Created metric record
        """
        metric = SystemMetrics(
            metric_name=metric_name,
            metric_value=metric_value,
            **kwargs
        )
        
        try:
            db.session.add(metric)
            db.session.commit()
            return metric
        except Exception as e:
            db.session.rollback()
            raise e


class ModelInfo(db.Model):
    """
    Model for storing information about ML models used in the application.
    Tracks model versions, performance metrics, and deployment status.
    """
    __tablename__ = 'model_info'
    
    # Primary key
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    
    # Model identification
    model_type = db.Column(db.String(20), nullable=False, index=True)  # 'crop' or 'fertilizer'
    model_name = db.Column(db.String(100), nullable=False)             # Descriptive name
    version = db.Column(db.String(20), nullable=False)                 # Version string (e.g., "1.0.0")
    
    # Model metadata
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    deployed_at = db.Column(db.DateTime, nullable=True)
    is_active = db.Column(db.Boolean, nullable=False, default=False, index=True)
    
    # Performance metrics
    accuracy = db.Column(db.Float, nullable=True)           # Model accuracy (0-1)
    precision = db.Column(db.Float, nullable=True)          # Model precision (0-1)
    recall = db.Column(db.Float, nullable=True)             # Model recall (0-1)
    f1_score = db.Column(db.Float, nullable=True)           # F1 score (0-1)
    
    # Model description and notes
    description = db.Column(db.Text, nullable=True)
    training_notes = db.Column(db.Text, nullable=True)
    
    def __init__(self, model_type, model_name, version, description=None, **kwargs):
        """
        Initialize a new model info record.
        
        Args:
            model_type (str): Type of model ('crop' or 'fertilizer')
            model_name (str): Descriptive name of the model
            version (str): Version string
            description (str, optional): Model description
            **kwargs: Additional model attributes
        """
        self.model_type = model_type
        self.model_name = model_name
        self.version = version
        self.description = description
        
        # Set additional attributes from kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def __repr__(self):
        """String representation of ModelInfo object"""
        return f'<ModelInfo {self.id}: {self.model_name} v{self.version} ({self.model_type})>'
    
    def to_dict(self):
        """
        Convert model instance to dictionary for JSON serialization.
        
        Returns:
            dict: Dictionary representation of the model info
        """
        return {
            'id': self.id,
            'model_type': self.model_type,
            'model_name': self.model_name,
            'version': self.version,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'deployed_at': self.deployed_at.isoformat() if self.deployed_at else None,
            'is_active': self.is_active,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'description': self.description,
            'training_notes': self.training_notes
        }
    
    @staticmethod
    def get_active_model(model_type):
        """
        Get the currently active model for a given type.
        
        Args:
            model_type (str): Type of model to retrieve
            
        Returns:
            ModelInfo or None: Active model or None if not found
        """
        return ModelInfo.query.filter_by(
            model_type=model_type,
            is_active=True
        ).first()
    
    def activate(self):
        """
        Activate this model and deactivate others of the same type.
        """
        try:
            # Deactivate other models of the same type
            ModelInfo.query.filter_by(
                model_type=self.model_type,
                is_active=True
            ).update({'is_active': False})
            
            # Activate this model
            self.is_active = True
            self.deployed_at = datetime.utcnow()
            
            db.session.commit()
            
        except Exception as e:
            db.session.rollback()
            raise e