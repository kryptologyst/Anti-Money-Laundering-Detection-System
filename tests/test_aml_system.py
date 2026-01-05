"""Tests for AML detection system."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.generator import AMLDataGenerator, AMLDataConfig
from features.engineering import AMLFeatureEngineer
from models.aml_models import DecisionTreeAML, RandomForestAML
from backtest.evaluation import AMLEvaluator
from utils.explainability import AMLExplainer


class TestAMLDataGenerator:
    """Test AML data generator."""
    
    def test_data_generator_initialization(self):
        """Test data generator initialization."""
        config = AMLDataConfig(n_customers=100, n_transactions=1000)
        generator = AMLDataGenerator(config)
        assert generator.config.n_customers == 100
        assert generator.config.n_transactions == 1000
    
    def test_generate_customers(self):
        """Test customer generation."""
        config = AMLDataConfig(n_customers=100)
        generator = AMLDataGenerator(config)
        customers = generator.generate_customers()
        
        assert len(customers) == 100
        assert 'customer_id' in customers.columns
        assert 'age' in customers.columns
        assert 'risk_score' in customers.columns
    
    def test_generate_transactions(self):
        """Test transaction generation."""
        config = AMLDataConfig(n_customers=10, n_transactions=100)
        generator = AMLDataGenerator(config)
        customers = generator.generate_customers()
        transactions = generator.generate_transactions(customers)
        
        assert len(transactions) > 0
        assert 'transaction_id' in transactions.columns
        assert 'amount' in transactions.columns
        assert 'is_suspicious' in transactions.columns


class TestAMLFeatureEngineer:
    """Test AML feature engineer."""
    
    def test_feature_engineer_initialization(self):
        """Test feature engineer initialization."""
        engineer = AMLFeatureEngineer()
        assert engineer is not None
    
    def test_create_customer_features(self):
        """Test customer feature creation."""
        engineer = AMLFeatureEngineer()
        
        customers = pd.DataFrame({
            'customer_id': ['C1', 'C2'],
            'age': [25, 45],
            'risk_score': [0.3, 0.8],
            'account_age_days': [100, 1000]
        })
        
        features = engineer.create_customer_features(customers)
        
        assert 'age_group' in features.columns
        assert 'risk_category' in features.columns
        assert 'account_age_group' in features.columns


class TestAMLModels:
    """Test AML models."""
    
    def test_decision_tree_initialization(self):
        """Test Decision Tree model initialization."""
        model = DecisionTreeAML()
        assert model is not None
        assert model.random_state == 42
    
    def test_random_forest_initialization(self):
        """Test Random Forest model initialization."""
        model = RandomForestAML()
        assert model is not None
        assert model.random_state == 42
    
    def test_model_training_and_prediction(self):
        """Test model training and prediction."""
        # Create sample data
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100)
        })
        y = pd.Series(np.random.randint(0, 2, 100))
        
        # Train model
        model = DecisionTreeAML()
        model.fit(X, y)
        
        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        assert len(predictions) == len(X)
        assert len(probabilities) == len(X)
        assert probabilities.shape[1] == 2  # Binary classification


class TestAMLEvaluator:
    """Test AML evaluator."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = AMLEvaluator()
        assert evaluator is not None
    
    def test_evaluate_model_performance(self):
        """Test model performance evaluation."""
        evaluator = AMLEvaluator()
        
        # Create sample data
        y_true = pd.Series([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0])
        y_proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
        
        metrics = evaluator.evaluate_model_performance(y_true, y_pred, y_proba[:, 1])
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'auc_roc' in metrics


class TestAMLExplainer:
    """Test AML explainer."""
    
    def test_explainer_initialization(self):
        """Test explainer initialization."""
        # Create mock model
        model = Mock()
        model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])
        
        X_train = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        
        explainer = AMLExplainer(model, X_train)
        assert explainer is not None
        assert explainer.model == model
        assert explainer.X_train.equals(X_train)


def test_data_quality():
    """Test data quality checks."""
    # Test that generated data has expected properties
    config = AMLDataConfig(n_customers=100, n_transactions=1000)
    generator = AMLDataGenerator(config)
    
    customers = generator.generate_customers()
    transactions = generator.generate_transactions(customers)
    
    # Check data types
    assert customers['age'].dtype in ['int64', 'int32']
    assert customers['risk_score'].dtype in ['float64', 'float32']
    assert transactions['amount'].dtype in ['float64', 'float32']
    
    # Check value ranges
    assert customers['age'].min() >= 18
    assert customers['age'].max() <= 80
    assert customers['risk_score'].min() >= 0
    assert customers['risk_score'].max() <= 1
    assert transactions['amount'].min() > 0


def test_model_reproducibility():
    """Test model reproducibility."""
    # Create sample data
    X = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100)
    })
    y = pd.Series(np.random.randint(0, 2, 100))
    
    # Train two models with same seed
    model1 = DecisionTreeAML(random_state=42)
    model2 = DecisionTreeAML(random_state=42)
    
    model1.fit(X, y)
    model2.fit(X, y)
    
    # Predictions should be identical
    pred1 = model1.predict(X)
    pred2 = model2.predict(X)
    
    assert np.array_equal(pred1, pred2)


if __name__ == "__main__":
    pytest.main([__file__])
