"""
Credit Risk Scoring System Package

A production-grade machine learning system for credit risk assessment
and loan default prediction with MySQL integration.
"""

__version__ = "1.0.0"
__author__ = "Rakshith Bopparathi"
__package_name__ = "credit-risk-scoring-system"

from .data_preprocessor import CreditRiskPreprocessor
from .model_trainer import ModelTrainer
from .credit_risk_model import CreditRiskModel
from .risk_scorer import RiskScorer

__all__ = [
    "CreditRiskPreprocessor",
    "ModelTrainer",
    "CreditRiskModel",
    "RiskScorer",
]
