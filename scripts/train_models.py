#!/usr/bin/env python3
"""Main training script for AML detection system."""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import hydra
from omegaconf import DictConfig, OmegaConf
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.generator import AMLDataGenerator, AMLDataConfig
from features.engineering import AMLFeatureEngineer
from models.aml_models import (
    DecisionTreeAML, RandomForestAML, XGBoostAML, 
    LightGBMAML, IsolationForestAML, AutoEncoderAML, AMLEnsemble
)
from backtest.evaluation import AMLEvaluator, AMLBacktester
from utils.explainability import AMLExplainer


def setup_logging(log_level: str = "INFO", log_file: str = None) -> None:
    """Setup logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )


def ensure_directories(config: DictConfig) -> None:
    """Ensure required directories exist.
    
    Args:
        config: Configuration object
    """
    directories = [
        config.data_dir,
        config.output_dir,
        config.model_dir,
        "logs",
        "assets"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def load_or_generate_data(config: DictConfig) -> tuple:
    """Load existing data or generate new data.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (customers, transactions, relationships, features)
    """
    data_dir = Path(config.data_dir)
    
    # Check if data already exists
    if all((data_dir / f"{name}.csv").exists() for name in ["customers", "transactions", "relationships", "features"]):
        logging.info("Loading existing data...")
        
        customers = pd.read_csv(data_dir / "customers.csv")
        transactions = pd.read_csv(data_dir / "transactions.csv")
        relationships = pd.read_csv(data_dir / "relationships.csv")
        features = pd.read_csv(data_dir / "features.csv")
        
        return customers, transactions, relationships, features
    
    else:
        logging.info("Generating new data...")
        
        # Generate data
        data_config = AMLDataConfig(
            n_customers=config.data.generation.n_customers,
            n_transactions=config.data.generation.n_transactions,
            suspicious_ratio=config.data.generation.suspicious_ratio,
            time_range_days=config.data.generation.time_range_days,
            seed=config.data.generation.seed
        )
        
        generator = AMLDataGenerator(data_config)
        customers, transactions, relationships = generator.generate_all_data()
        
        # Save raw data
        customers.to_csv(data_dir / "customers.csv", index=False)
        transactions.to_csv(data_dir / "transactions.csv", index=False)
        relationships.to_csv(data_dir / "relationships.csv", index=False)
        
        # Generate features
        logging.info("Creating features...")
        engineer = AMLFeatureEngineer()
        features = engineer.create_all_features(customers, transactions, relationships)
        
        # Save features
        features.to_csv(data_dir / "features.csv", index=False)
        
        return customers, transactions, relationships, features


def prepare_training_data(features: pd.DataFrame, config: DictConfig) -> tuple:
    """Prepare data for training.
    
    Args:
        features: Feature DataFrame
        config: Configuration object
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # Prepare features and target
    target_cols = ['is_suspicious', 'transaction_id', 'customer_id']
    X = features.drop(columns=target_cols)
    y = features['is_suspicious']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.data.splits.test_size,
        random_state=config.data.splits.random_state,
        stratify=y
    )
    
    logging.info(f"Training set size: {len(X_train)}")
    logging.info(f"Test set size: {len(X_test)}")
    logging.info(f"Suspicious rate in training: {y_train.mean():.3f}")
    logging.info(f"Suspicious rate in test: {y_test.mean():.3f}")
    
    return X_train, X_test, y_train, y_test


def train_models(X_train: pd.DataFrame, y_train: pd.Series, config: DictConfig) -> Dict[str, Any]:
    """Train multiple AML models.
    
    Args:
        X_train: Training features
        y_train: Training labels
        config: Configuration object
        
    Returns:
        Dictionary of trained models
    """
    models = {}
    
    # Model configurations
    model_configs = {
        'decision_tree': DecisionTreeAML,
        'random_forest': RandomForestAML,
        'xgboost': XGBoostAML,
        'lightgbm': LightGBMAML,
        'isolation_forest': IsolationForestAML,
        'autoencoder': AutoEncoderAML
    }
    
    for model_name in config.models:
        if model_name in model_configs:
            logging.info(f"Training {model_name}...")
            
            try:
                model = model_configs[model_name](random_state=config.seed)
                model.fit(X_train, y_train)
                models[model_name] = model
                
                logging.info(f"Successfully trained {model_name}")
                
            except Exception as e:
                logging.error(f"Error training {model_name}: {e}")
                continue
    
    return models


def evaluate_models(
    models: Dict[str, Any], 
    X_test: pd.DataFrame, 
    y_test: pd.Series,
    config: DictConfig
) -> Dict[str, Dict[str, float]]:
    """Evaluate trained models.
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test labels
        config: Configuration object
        
    Returns:
        Dictionary of evaluation results
    """
    evaluator = AMLEvaluator()
    results = {}
    
    for model_name, model in models.items():
        logging.info(f"Evaluating {model_name}...")
        
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            
            # Evaluate
            metrics = evaluator.evaluate_model_performance(y_test, y_pred, y_proba[:, 1])
            results[model_name] = metrics
            
            logging.info(f"{model_name} - Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1_score']:.3f}")
            
        except Exception as e:
            logging.error(f"Error evaluating {model_name}: {e}")
            continue
    
    return results


def create_explanations(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: DictConfig
) -> None:
    """Create model explanations.
    
    Args:
        models: Dictionary of trained models
        X_train: Training features
        X_test: Test features
        y_test: Test labels
        config: Configuration object
    """
    if not config.explainability.create_plots:
        return
    
    logging.info("Creating model explanations...")
    
    # Create explanations for the best model
    if models:
        best_model_name = max(models.keys(), key=lambda k: results.get(k, {}).get('f1_score', 0))
        best_model = models[best_model_name]
        
        try:
            explainer = AMLExplainer(best_model, X_train)
            
            # Create plots
            explainer.plot_feature_importance(
                X_test, 
                f"{config.output_dir}/feature_importance_{best_model_name}.png"
            )
            
            explainer.plot_waterfall_plot(
                X_test, 
                0, 
                f"{config.output_dir}/waterfall_plot_{best_model_name}.png"
            )
            
            # Generate report
            y_pred = best_model.predict(X_test)
            report = explainer.generate_explanation_report(
                X_test, y_test, y_pred,
                f"{config.output_dir}/explanation_report_{best_model_name}.txt"
            )
            
            logging.info(f"Created explanations for {best_model_name}")
            
        except Exception as e:
            logging.error(f"Error creating explanations: {e}")


def save_results(
    results: Dict[str, Dict[str, float]],
    models: Dict[str, Any],
    config: DictConfig
) -> None:
    """Save training results and models.
    
    Args:
        results: Evaluation results
        models: Trained models
        config: Configuration object
    """
    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv(f"{config.output_dir}/model_results.csv")
    
    # Save best model
    if results:
        best_model_name = max(results.keys(), key=lambda k: results[k]['f1_score'])
        best_model = models[best_model_name]
        
        import pickle
        with open(f"{config.model_dir}/best_model_{best_model_name}.pkl", 'wb') as f:
            pickle.dump(best_model, f)
        
        logging.info(f"Saved best model: {best_model_name}")


def run_backtesting(
    models: Dict[str, Any],
    features: pd.DataFrame,
    config: DictConfig
) -> None:
    """Run backtesting on models.
    
    Args:
        models: Dictionary of trained models
        features: Feature DataFrame
        config: Configuration object
    """
    if not config.backtesting.enabled:
        return
    
    logging.info("Running backtesting...")
    
    backtester = AMLBacktester()
    
    # Prepare data for backtesting
    target_cols = ['is_suspicious', 'transaction_id', 'customer_id']
    X = features.drop(columns=target_cols)
    y = features['is_suspicious']
    
    # Run backtesting for each model
    for model_name, model in models.items():
        try:
            logging.info(f"Running backtesting for {model_name}...")
            
            results = backtester.walk_forward_validation(
                model, features, X.columns.tolist(), 'is_suspicious',
                config.backtesting.n_splits
            )
            
            # Plot results
            backtester.plot_backtest_results(
                results,
                f"{config.output_dir}/backtest_results_{model_name}.png"
            )
            
            logging.info(f"Completed backtesting for {model_name}")
            
        except Exception as e:
            logging.error(f"Error in backtesting for {model_name}: {e}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    """Main training function.
    
    Args:
        config: Configuration object
    """
    # Setup
    setup_logging(config.log_level, config.log_file)
    ensure_directories(config)
    
    logging.info("Starting AML model training...")
    logging.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")
    
    try:
        # Load or generate data
        customers, transactions, relationships, features = load_or_generate_data(config)
        
        # Prepare training data
        X_train, X_test, y_train, y_test = prepare_training_data(features, config)
        
        # Train models
        models = train_models(X_train, y_train, config)
        
        if not models:
            logging.error("No models were successfully trained")
            return
        
        # Evaluate models
        results = evaluate_models(models, X_test, y_test, config)
        
        # Create explanations
        create_explanations(models, X_train, X_test, y_test, config)
        
        # Save results
        save_results(results, models, config)
        
        # Run backtesting
        run_backtesting(models, features, config)
        
        logging.info("Training completed successfully!")
        
        # Print summary
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        
        for model_name, metrics in results.items():
            print(f"\n{model_name.upper()}:")
            print(f"  Accuracy:  {metrics['accuracy']:.3f}")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall:    {metrics['recall']:.3f}")
            print(f"  F1-Score:  {metrics['f1_score']:.3f}")
            print(f"  AUC-ROC:   {metrics['auc_roc']:.3f}")
        
        print(f"\nBest model: {max(results.keys(), key=lambda k: results[k]['f1_score'])}")
        print("="*50)
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
