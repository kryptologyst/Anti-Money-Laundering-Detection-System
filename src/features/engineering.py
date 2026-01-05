"""Feature engineering for AML detection system."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class AMLFeatureEngineer:
    """Feature engineering for AML transaction data."""
    
    def __init__(self) -> None:
        """Initialize the feature engineer."""
        pass
    
    def create_customer_features(self, customers: pd.DataFrame) -> pd.DataFrame:
        """Create customer-level features.
        
        Args:
            customers: Customer profile data
            
        Returns:
            DataFrame with customer features
        """
        features = customers.copy()
        
        # Age groups
        features['age_group'] = pd.cut(
            features['age'], 
            bins=[0, 25, 35, 50, 65, 100], 
            labels=['young', 'adult', 'middle', 'senior', 'elderly']
        )
        
        # Risk categories
        features['risk_category'] = pd.cut(
            features['risk_score'],
            bins=[0, 0.3, 0.6, 0.8, 1.0],
            labels=['low', 'medium', 'high', 'very_high']
        )
        
        # Account age categories
        features['account_age_group'] = pd.cut(
            features['account_age_days'],
            bins=[0, 90, 365, 1095, 3650],
            labels=['new', 'recent', 'established', 'long_term']
        )
        
        return features
    
    def create_transaction_features(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Create transaction-level features.
        
        Args:
            transactions: Transaction data
            
        Returns:
            DataFrame with transaction features
        """
        features = transactions.copy()
        
        # Time-based features
        features['timestamp'] = pd.to_datetime(features['timestamp'])
        features['date'] = features['timestamp'].dt.date
        features['month'] = features['timestamp'].dt.month
        features['quarter'] = features['timestamp'].dt.quarter
        
        # Amount-based features
        features['amount_log'] = np.log1p(features['amount'])
        features['amount_zscore'] = self._calculate_zscore(features['amount'])
        
        # Transaction size categories
        features['amount_category'] = pd.cut(
            features['amount'],
            bins=[0, 1000, 5000, 10000, 50000, float('inf')],
            labels=['small', 'medium', 'large', 'very_large', 'huge']
        )
        
        # Time-based patterns
        features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)
        features['is_business_hours'] = features['hour'].between(9, 17).astype(int)
        features['is_night'] = features['hour'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
        
        # Round amount detection (structuring)
        features['is_round_amount'] = (
            (features['amount'] % 1000 == 0) & 
            (features['amount'] > 1000)
        ).astype(int)
        
        # Just under threshold detection
        features['just_under_threshold'] = (
            (features['amount'] >= 9000) & 
            (features['amount'] < 10000)
        ).astype(int)
        
        return features
    
    def create_customer_transaction_features(
        self, 
        transactions: pd.DataFrame, 
        customers: pd.DataFrame
    ) -> pd.DataFrame:
        """Create customer-transaction aggregated features.
        
        Args:
            transactions: Transaction data
            customers: Customer profile data
            
        Returns:
            DataFrame with customer-transaction features
        """
        # Merge with customer data
        merged = transactions.merge(customers, on='customer_id', how='left')
        
        # Customer-level aggregations
        customer_stats = transactions.groupby('customer_id').agg({
            'amount': ['sum', 'mean', 'std', 'min', 'max', 'count'],
            'is_suspicious': ['sum', 'mean'],
            'location_type': lambda x: (x == 'international').sum(),
            'transaction_type': 'nunique'
        }).round(2)
        
        # Flatten column names
        customer_stats.columns = [
            'total_amount', 'avg_amount', 'amount_std', 'min_amount', 'max_amount', 'transaction_count',
            'suspicious_count', 'suspicious_rate', 'international_count', 'transaction_type_diversity'
        ]
        
        # Reset index
        customer_stats = customer_stats.reset_index()
        
        # Add derived features
        customer_stats['amount_volatility'] = customer_stats['amount_std'] / customer_stats['avg_amount']
        customer_stats['transaction_frequency'] = customer_stats['transaction_count'] / 180  # per day
        customer_stats['international_ratio'] = customer_stats['international_count'] / customer_stats['transaction_count']
        
        # Merge back with transactions
        features = transactions.merge(customer_stats, on='customer_id', how='left')
        
        return features
    
    def create_temporal_features(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Create temporal pattern features.
        
        Args:
            transactions: Transaction data
            
        Returns:
            DataFrame with temporal features
        """
        features = transactions.copy()
        features['timestamp'] = pd.to_datetime(features['timestamp'])
        
        # Sort by customer and timestamp
        features = features.sort_values(['customer_id', 'timestamp'])
        
        # Time between transactions
        features['time_since_last'] = features.groupby('customer_id')['timestamp'].diff().dt.total_seconds() / 3600  # hours
        
        # Transaction frequency (rolling window)
        features['transactions_last_7d'] = features.groupby('customer_id').rolling(
            '7D', on='timestamp'
        )['transaction_id'].count().reset_index(0, drop=True)
        
        features['transactions_last_30d'] = features.groupby('customer_id').rolling(
            '30D', on='timestamp'
        )['transaction_id'].count().reset_index(0, drop=True)
        
        # Amount patterns
        features['amount_last_7d'] = features.groupby('customer_id').rolling(
            '7D', on='timestamp'
        )['amount'].sum().reset_index(0, drop=True)
        
        features['amount_last_30d'] = features.groupby('customer_id').rolling(
            '30D', on='timestamp'
        )['amount'].sum().reset_index(0, drop=True)
        
        # Fill NaN values
        features['time_since_last'] = features['time_since_last'].fillna(24)  # Default 24 hours
        features['transactions_last_7d'] = features['transactions_last_7d'].fillna(1)
        features['transactions_last_30d'] = features['transactions_last_30d'].fillna(1)
        features['amount_last_7d'] = features['amount_last_7d'].fillna(features['amount'])
        features['amount_last_30d'] = features['amount_last_30d'].fillna(features['amount'])
        
        return features
    
    def create_network_features(
        self, 
        transactions: pd.DataFrame, 
        relationships: pd.DataFrame
    ) -> pd.DataFrame:
        """Create network-based features.
        
        Args:
            transactions: Transaction data
            relationships: Customer relationship data
            
        Returns:
            DataFrame with network features
        """
        features = transactions.copy()
        
        # Customer network statistics
        customer_connections = relationships.groupby('customer_1').size().reset_index(name='connection_count')
        customer_connections.columns = ['customer_id', 'connection_count']
        
        # Average connection strength
        customer_strength = relationships.groupby('customer_1')['strength'].mean().reset_index()
        customer_strength.columns = ['customer_id', 'avg_connection_strength']
        
        # Merge network features
        features = features.merge(customer_connections, on='customer_id', how='left')
        features = features.merge(customer_strength, on='customer_id', how='left')
        
        # Fill missing values
        features['connection_count'] = features['connection_count'].fillna(0)
        features['avg_connection_strength'] = features['avg_connection_strength'].fillna(0)
        
        # Network risk score
        features['network_risk_score'] = (
            features['connection_count'] * 0.3 + 
            features['avg_connection_strength'] * 0.7
        )
        
        return features
    
    def create_behavioral_features(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Create behavioral pattern features.
        
        Args:
            transactions: Transaction data
            
        Returns:
            DataFrame with behavioral features
        """
        features = transactions.copy()
        
        # Transaction pattern analysis
        customer_patterns = transactions.groupby('customer_id').agg({
            'amount': ['mean', 'std', 'skew'],
            'hour': ['mean', 'std'],
            'day_of_week': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0,
            'transaction_type': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'transfer',
            'location_type': lambda x: (x == 'international').mean()
        })
        
        # Flatten column names
        customer_patterns.columns = [
            'avg_amount', 'amount_std', 'amount_skew',
            'avg_hour', 'hour_std', 'preferred_day', 'preferred_type', 'international_ratio'
        ]
        
        customer_patterns = customer_patterns.reset_index()
        
        # Merge with transactions
        features = features.merge(customer_patterns, on='customer_id', how='left')
        
        # Deviation from normal patterns
        features['amount_deviation'] = abs(features['amount'] - features['avg_amount']) / features['avg_amount']
        features['hour_deviation'] = abs(features['hour'] - features['avg_hour'])
        features['type_deviation'] = (features['transaction_type'] != features['preferred_type']).astype(int)
        
        return features
    
    def _calculate_zscore(self, series: pd.Series) -> pd.Series:
        """Calculate z-score for a series.
        
        Args:
            series: Input series
            
        Returns:
            Series with z-scores
        """
        return (series - series.mean()) / series.std()
    
    def create_all_features(
        self, 
        customers: pd.DataFrame, 
        transactions: pd.DataFrame, 
        relationships: pd.DataFrame
    ) -> pd.DataFrame:
        """Create all features for the AML model.
        
        Args:
            customers: Customer profile data
            transactions: Transaction data
            relationships: Customer relationship data
            
        Returns:
            DataFrame with all engineered features
        """
        print("Creating customer features...")
        customer_features = self.create_customer_features(customers)
        
        print("Creating transaction features...")
        transaction_features = self.create_transaction_features(transactions)
        
        print("Creating customer-transaction features...")
        customer_transaction_features = self.create_customer_transaction_features(
            transactions, customers
        )
        
        print("Creating temporal features...")
        temporal_features = self.create_temporal_features(transactions)
        
        print("Creating network features...")
        network_features = self.create_network_features(transactions, relationships)
        
        print("Creating behavioral features...")
        behavioral_features = self.create_behavioral_features(transactions)
        
        # Merge all features
        print("Merging all features...")
        features = transaction_features.merge(
            customer_features, on='customer_id', how='left'
        )
        
        features = features.merge(
            customer_transaction_features[['transaction_id'] + 
            [col for col in customer_transaction_features.columns if col not in features.columns]], 
            on='transaction_id', how='left'
        )
        
        features = features.merge(
            temporal_features[['transaction_id'] + 
            [col for col in temporal_features.columns if col not in features.columns]], 
            on='transaction_id', how='left'
        )
        
        features = features.merge(
            network_features[['transaction_id'] + 
            [col for col in network_features.columns if col not in features.columns]], 
            on='transaction_id', how='left'
        )
        
        features = features.merge(
            behavioral_features[['transaction_id'] + 
            [col for col in behavioral_features.columns if col not in features.columns]], 
            on='transaction_id', how='left'
        )
        
        # Fill missing values
        features = features.fillna(0)
        
        print(f"Created {len(features.columns)} features")
        return features


def main() -> None:
    """Test feature engineering."""
    # Load data
    customers = pd.read_csv('data/customers.csv')
    transactions = pd.read_csv('data/transactions.csv')
    relationships = pd.read_csv('data/relationships.csv')
    
    # Create features
    engineer = AMLFeatureEngineer()
    features = engineer.create_all_features(customers, transactions, relationships)
    
    # Save features
    features.to_csv('data/features.csv', index=False)
    print(f"Saved features to data/features.csv")
    print(f"Feature shape: {features.shape}")


if __name__ == "__main__":
    main()
