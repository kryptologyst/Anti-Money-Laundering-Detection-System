"""Explainability and interpretability for AML detection models."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import warnings
warnings.filterwarnings('ignore')


class AMLExplainer:
    """Explainability framework for AML detection models."""
    
    def __init__(self, model: Any, X_train: pd.DataFrame) -> None:
        """Initialize the explainer.
        
        Args:
            model: Trained AML model
            X_train: Training data for background
        """
        self.model = model
        self.X_train = X_train
        self.explainer = None
        self.shap_values = None
        
    def create_shap_explainer(self) -> None:
        """Create SHAP explainer for the model."""
        try:
            # Try different explainers based on model type
            if hasattr(self.model, 'predict_proba'):
                # For tree-based models
                if hasattr(self.model.model, 'feature_importances_'):
                    self.explainer = shap.TreeExplainer(self.model.model)
                else:
                    # For other models, use KernelExplainer
                    self.explainer = shap.KernelExplainer(
                        self.model.predict_proba, 
                        self.X_train.sample(min(100, len(self.X_train)))
                    )
            else:
                # Fallback to KernelExplainer
                self.explainer = shap.KernelExplainer(
                    self.model.predict, 
                    self.X_train.sample(min(100, len(self.X_train)))
                )
        except Exception as e:
            print(f"Error creating SHAP explainer: {e}")
            # Fallback to KernelExplainer
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba, 
                self.X_train.sample(min(100, len(self.X_train)))
            )
    
    def calculate_shap_values(self, X_test: pd.DataFrame) -> np.ndarray:
        """Calculate SHAP values for test data.
        
        Args:
            X_test: Test data
            
        Returns:
            SHAP values
        """
        if self.explainer is None:
            self.create_shap_explainer()
        
        try:
            self.shap_values = self.explainer.shap_values(X_test)
            return self.shap_values
        except Exception as e:
            print(f"Error calculating SHAP values: {e}")
            return None
    
    def plot_feature_importance(
        self, 
        X_test: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> None:
        """Plot feature importance using SHAP.
        
        Args:
            X_test: Test data
            save_path: Optional path to save plot
        """
        if self.shap_values is None:
            self.calculate_shap_values(X_test)
        
        if self.shap_values is not None:
            # Handle binary classification case
            if isinstance(self.shap_values, list) and len(self.shap_values) == 2:
                shap_values_to_plot = self.shap_values[1]  # Use positive class
            else:
                shap_values_to_plot = self.shap_values
            
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values_to_plot, 
                X_test, 
                feature_names=X_test.columns,
                show=False
            )
            plt.title('SHAP Feature Importance for AML Detection')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
    
    def plot_waterfall_plot(
        self, 
        X_test: pd.DataFrame, 
        instance_idx: int = 0,
        save_path: Optional[str] = None
    ) -> None:
        """Plot waterfall plot for a specific instance.
        
        Args:
            X_test: Test data
            instance_idx: Index of instance to explain
            save_path: Optional path to save plot
        """
        if self.shap_values is None:
            self.calculate_shap_values(X_test)
        
        if self.shap_values is not None:
            # Handle binary classification case
            if isinstance(self.shap_values, list) and len(self.shap_values) == 2:
                shap_values_to_plot = self.shap_values[1]  # Use positive class
            else:
                shap_values_to_plot = self.shap_values
            
            plt.figure(figsize=(12, 8))
            shap.waterfall_plot(
                self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list) else self.explainer.expected_value,
                shap_values_to_plot[instance_idx],
                X_test.iloc[instance_idx],
                feature_names=X_test.columns,
                show=False
            )
            plt.title(f'SHAP Waterfall Plot - Instance {instance_idx}')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
    
    def plot_force_plot(
        self, 
        X_test: pd.DataFrame, 
        instance_idx: int = 0
    ) -> None:
        """Plot force plot for a specific instance.
        
        Args:
            X_test: Test data
            instance_idx: Index of instance to explain
        """
        if self.shap_values is None:
            self.calculate_shap_values(X_test)
        
        if self.shap_values is not None:
            # Handle binary classification case
            if isinstance(self.shap_values, list) and len(self.shap_values) == 2:
                shap_values_to_plot = self.shap_values[1]  # Use positive class
                expected_value = self.explainer.expected_value[1]
            else:
                shap_values_to_plot = self.shap_values
                expected_value = self.explainer.expected_value
            
            shap.force_plot(
                expected_value,
                shap_values_to_plot[instance_idx],
                X_test.iloc[instance_idx],
                feature_names=X_test.columns
            )
    
    def create_feature_attribution_summary(
        self, 
        X_test: pd.DataFrame,
        y_test: pd.Series,
        top_n: int = 20
    ) -> pd.DataFrame:
        """Create summary of feature attributions.
        
        Args:
            X_test: Test data
            y_test: Test labels
            top_n: Number of top features to include
            
        Returns:
            DataFrame with feature attribution summary
        """
        if self.shap_values is None:
            self.calculate_shap_values(X_test)
        
        if self.shap_values is None:
            return pd.DataFrame()
        
        # Handle binary classification case
        if isinstance(self.shap_values, list) and len(self.shap_values) == 2:
            shap_values_to_plot = self.shap_values[1]  # Use positive class
        else:
            shap_values_to_plot = self.shap_values
        
        # Calculate mean absolute SHAP values
        mean_shap_values = np.abs(shap_values_to_plot).mean(axis=0)
        
        # Create summary DataFrame
        summary_df = pd.DataFrame({
            'feature': X_test.columns,
            'mean_abs_shap': mean_shap_values,
            'mean_shap': shap_values_to_plot.mean(axis=0),
            'std_shap': shap_values_to_plot.std(axis=0)
        }).sort_values('mean_abs_shap', ascending=False)
        
        # Add feature impact categories
        summary_df['impact_category'] = pd.cut(
            summary_df['mean_abs_shap'],
            bins=[0, 0.01, 0.05, 0.1, float('inf')],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        return summary_df.head(top_n)
    
    def analyze_suspicious_cases(
        self, 
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_pred: np.ndarray,
        top_n: int = 10
    ) -> Dict[str, Any]:
        """Analyze feature attributions for suspicious cases.
        
        Args:
            X_test: Test data
            y_test: Test labels
            y_pred: Predictions
            top_n: Number of top cases to analyze
            
        Returns:
            Dictionary with analysis results
        """
        if self.shap_values is None:
            self.calculate_shap_values(X_test)
        
        if self.shap_values is None:
            return {}
        
        # Handle binary classification case
        if isinstance(self.shap_values, list) and len(self.shap_values) == 2:
            shap_values_to_plot = self.shap_values[1]  # Use positive class
        else:
            shap_values_to_plot = self.shap_values
        
        # Get suspicious cases
        suspicious_mask = y_test == 1
        suspicious_indices = X_test[suspicious_mask].index
        
        if len(suspicious_indices) == 0:
            return {}
        
        # Analyze top suspicious cases
        suspicious_shap = shap_values_to_plot[suspicious_mask]
        suspicious_X = X_test[suspicious_mask]
        
        # Get top contributing features for each suspicious case
        case_analyses = []
        
        for i, idx in enumerate(suspicious_indices[:top_n]):
            case_shap = suspicious_shap[i]
            case_features = suspicious_X.iloc[i]
            
            # Get top contributing features
            feature_contributions = pd.DataFrame({
                'feature': X_test.columns,
                'value': case_features.values,
                'shap_value': case_shap,
                'abs_shap_value': np.abs(case_shap)
            }).sort_values('abs_shap_value', ascending=False)
            
            case_analyses.append({
                'case_id': idx,
                'top_features': feature_contributions.head(10),
                'total_contribution': case_shap.sum(),
                'max_contribution': case_shap.max(),
                'min_contribution': case_shap.min()
            })
        
        # Overall analysis
        overall_analysis = {
            'total_suspicious_cases': len(suspicious_indices),
            'avg_contribution': suspicious_shap.mean(axis=0),
            'std_contribution': suspicious_shap.std(axis=0),
            'case_analyses': case_analyses
        }
        
        return overall_analysis
    
    def create_interactive_explanation(
        self, 
        X_test: pd.DataFrame,
        instance_idx: int = 0
    ) -> go.Figure:
        """Create interactive explanation plot.
        
        Args:
            X_test: Test data
            instance_idx: Index of instance to explain
            
        Returns:
            Plotly figure
        """
        if self.shap_values is None:
            self.calculate_shap_values(X_test)
        
        if self.shap_values is None:
            return go.Figure()
        
        # Handle binary classification case
        if isinstance(self.shap_values, list) and len(self.shap_values) == 2:
            shap_values_to_plot = self.shap_values[1]  # Use positive class
            expected_value = self.explainer.expected_value[1]
        else:
            shap_values_to_plot = self.shap_values
            expected_value = self.explainer.expected_value
        
        # Get instance data
        instance_shap = shap_values_to_plot[instance_idx]
        instance_values = X_test.iloc[instance_idx]
        
        # Sort features by absolute SHAP value
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'shap_value': instance_shap,
            'abs_shap_value': np.abs(instance_shap),
            'feature_value': instance_values.values
        }).sort_values('abs_shap_value', ascending=True)
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        # Add bars for positive contributions
        positive_mask = feature_importance['shap_value'] > 0
        if positive_mask.any():
            fig.add_trace(go.Bar(
                y=feature_importance[positive_mask]['feature'],
                x=feature_importance[positive_mask]['shap_value'],
                orientation='h',
                name='Positive Contribution',
                marker_color='red',
                text=[f"{val:.3f}" for val in feature_importance[positive_mask]['shap_value']],
                textposition='auto'
            ))
        
        # Add bars for negative contributions
        negative_mask = feature_importance['shap_value'] < 0
        if negative_mask.any():
            fig.add_trace(go.Bar(
                y=feature_importance[negative_mask]['feature'],
                x=feature_importance[negative_mask]['shap_value'],
                orientation='h',
                name='Negative Contribution',
                marker_color='blue',
                text=[f"{val:.3f}" for val in feature_importance[negative_mask]['shap_value']],
                textposition='auto'
            ))
        
        # Add expected value line
        fig.add_vline(x=expected_value, line_dash="dash", line_color="green", 
                     annotation_text=f"Expected Value: {expected_value:.3f}")
        
        # Update layout
        fig.update_layout(
            title=f'SHAP Feature Contributions - Instance {instance_idx}',
            xaxis_title='SHAP Value',
            yaxis_title='Features',
            height=600,
            showlegend=True
        )
        
        return fig
    
    def generate_explanation_report(
        self, 
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_pred: np.ndarray,
        save_path: Optional[str] = None
    ) -> str:
        """Generate comprehensive explanation report.
        
        Args:
            X_test: Test data
            y_test: Test labels
            y_pred: Predictions
            save_path: Optional path to save report
            
        Returns:
            Formatted explanation report
        """
        if self.shap_values is None:
            self.calculate_shap_values(X_test)
        
        if self.shap_values is None:
            return "Error: Could not calculate SHAP values"
        
        # Get feature attribution summary
        feature_summary = self.create_feature_attribution_summary(X_test, y_test)
        
        # Analyze suspicious cases
        suspicious_analysis = self.analyze_suspicious_cases(X_test, y_test, y_pred)
        
        # Generate report
        report = f"""
AML Model Explanation Report
===========================
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

TOP FEATURE CONTRIBUTIONS
-------------------------
"""
        
        for i, row in feature_summary.head(10).iterrows():
            report += f"{row['feature']:<30} {row['mean_abs_shap']:.4f} ({row['impact_category']})\n"
        
        if suspicious_analysis:
            report += f"""
SUSPICIOUS CASE ANALYSIS
------------------------
Total Suspicious Cases: {suspicious_analysis['total_suspicious_cases']}

Top Contributing Features for Suspicious Cases:
"""
            top_features = pd.DataFrame({
                'feature': X_test.columns,
                'avg_contribution': suspicious_analysis['avg_contribution'],
                'std_contribution': suspicious_analysis['std_contribution']
            }).sort_values('avg_contribution', key=abs, ascending=False)
            
            for i, row in top_features.head(10).iterrows():
                report += f"{row['feature']:<30} {row['avg_contribution']:.4f} (±{row['std_contribution']:.4f})\n"
        
        report += """
DISCLAIMER
----------
This explanation is for research and educational purposes only.
Feature attributions should not be used for real-world AML compliance decisions.
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report


def main() -> None:
    """Test explainability framework."""
    # Load data
    features = pd.read_csv('data/features.csv')
    
    # Prepare data
    target_cols = ['is_suspicious', 'transaction_id', 'customer_id']
    X = features.drop(columns=target_cols)
    y = features['is_suspicious']
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train a model
    from src.models.aml_models import RandomForestAML
    model = RandomForestAML()
    model.fit(X_train, y_train)
    
    # Create explainer
    explainer = AMLExplainer(model, X_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Generate explanations
    print("Creating feature importance plot...")
    explainer.plot_feature_importance(X_test, 'assets/feature_importance.png')
    
    print("Creating waterfall plot...")
    explainer.plot_waterfall_plot(X_test, 0, 'assets/waterfall_plot.png')
    
    print("Generating explanation report...")
    report = explainer.generate_explanation_report(X_test, y_test, y_pred, 'assets/explanation_report.txt')
    print(report)


if __name__ == "__main__":
    main()
