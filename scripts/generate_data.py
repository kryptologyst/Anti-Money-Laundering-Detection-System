#!/usr/bin/env python3
"""Data generation script for AML detection system."""

import os
import sys
import logging
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.generator import AMLDataGenerator, AMLDataConfig
from features.engineering import AMLFeatureEngineer


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration.
    
    Args:
        log_level: Logging level
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def ensure_directories(data_dir: str) -> None:
    """Ensure data directory exists.
    
    Args:
        data_dir: Data directory path
    """
    Path(data_dir).mkdir(parents=True, exist_ok=True)


def generate_data(config: DictConfig) -> None:
    """Generate AML data.
    
    Args:
        config: Configuration object
    """
    logging.info("Starting data generation...")
    
    # Ensure data directory exists
    ensure_directories(config.data_dir)
    
    # Create data generator
    data_config = AMLDataConfig(
        n_customers=config.data.generation.n_customers,
        n_transactions=config.data.generation.n_transactions,
        suspicious_ratio=config.data.generation.suspicious_ratio,
        time_range_days=config.data.generation.time_range_days,
        seed=config.data.generation.seed
    )
    
    generator = AMLDataGenerator(data_config)
    
    # Generate data
    logging.info("Generating customer profiles...")
    customers = generator.generate_customers()
    
    logging.info("Generating transaction data...")
    transactions = generator.generate_transactions(customers)
    
    logging.info("Generating network relationships...")
    relationships = generator.generate_network_data(transactions)
    
    # Save raw data
    logging.info("Saving raw data...")
    customers.to_csv(f"{config.data_dir}/customers.csv", index=False)
    transactions.to_csv(f"{config.data_dir}/transactions.csv", index=False)
    relationships.to_csv(f"{config.data_dir}/relationships.csv", index=False)
    
    # Generate features
    logging.info("Creating features...")
    engineer = AMLFeatureEngineer()
    features = engineer.create_all_features(customers, transactions, relationships)
    
    # Save features
    features.to_csv(f"{config.data_dir}/features.csv", index=False)
    
    # Print summary
    print("\n" + "="*50)
    print("DATA GENERATION SUMMARY")
    print("="*50)
    print(f"Customers: {len(customers):,}")
    print(f"Transactions: {len(transactions):,}")
    print(f"Relationships: {len(relationships):,}")
    print(f"Features: {len(features):,}")
    print(f"Suspicious transaction rate: {transactions['is_suspicious'].mean():.3f}")
    print(f"Feature columns: {len(features.columns)}")
    print("="*50)
    
    logging.info("Data generation completed successfully!")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    """Main data generation function.
    
    Args:
        config: Configuration object
    """
    # Setup
    setup_logging(config.log_level)
    
    logging.info("Starting AML data generation...")
    logging.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")
    
    try:
        generate_data(config)
        
    except Exception as e:
        logging.error(f"Data generation failed: {e}")
        raise


if __name__ == "__main__":
    main()
