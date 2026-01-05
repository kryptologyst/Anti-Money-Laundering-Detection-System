"""Evaluation and backtesting framework for AML detection."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, 
    precision_recall_curve, roc_auc_score, average_precision_score
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


class AMLEvaluator:
    """Comprehensive evaluator for AML detection models."""
    
    def __init__(self) -> None:
        """Initialize the evaluator."""
        pass
    
    def evaluate_model_performance(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray, 
        y_proba: np.ndarray,
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """Evaluate model performance comprehensively.
        
        Args:
            y_true: True labels
            y_pred: Predictions
            y_proba: Prediction probabilities
            model_name: Name of the model
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Basic classification metrics
        metrics = self._calculate_basic_metrics(y_true, y_pred, y_proba)
        
        # AML-specific metrics
        aml_metrics = self._calculate_aml_metrics(y_true, y_pred, y_proba)
        
        # Business impact metrics
        business_metrics = self._calculate_business_metrics(y_true, y_pred, y_proba)
        
        # Combine all metrics
        all_metrics = {
            **metrics,
            **aml_metrics,
            **business_metrics
        }
        
        return all_metrics
    
    def _calculate_basic_metrics(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray, 
        y_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate basic classification metrics."""
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Basic metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # AUC metrics
        auc_roc = roc_auc_score(y_true, y_proba)
        auc_pr = average_precision_score(y_true, y_proba)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }
    
    def _calculate_aml_metrics(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray, 
        y_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate AML-specific metrics."""
        # Precision@K metrics
        precision_at_k = {}
        k_values = [10, 25, 50, 100, 200, 500]
        
        for k in k_values:
            if len(y_proba) >= k:
                top_k_indices = np.argsort(y_proba)[-k:]
                precision_at_k[f'precision_at_{k}'] = y_true.iloc[top_k_indices].mean()
            else:
                precision_at_k[f'precision_at_{k}'] = 0.0
        
        # Case-level analysis
        suspicious_cases = y_true.sum()
        detected_cases = ((y_true == 1) & (y_pred == 1)).sum()
        case_detection_rate = detected_cases / suspicious_cases if suspicious_cases > 0 else 0.0
        
        # Alert quality metrics
        total_alerts = y_pred.sum()
        true_alerts = ((y_true == 1) & (y_pred == 1)).sum()
        alert_quality = true_alerts / total_alerts if total_alerts > 0 else 0.0
        
        # Investigator workload reduction
        baseline_alerts = len(y_true) * 0.1  # Assume 10% manual review rate
        workload_reduction = (baseline_alerts - total_alerts) / baseline_alerts if baseline_alerts > 0 else 0.0
        
        return {
            **precision_at_k,
            'case_detection_rate': case_detection_rate,
            'alert_quality': alert_quality,
            'workload_reduction': workload_reduction,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0.0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0.0
        }
    
    def _calculate_business_metrics(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray, 
        y_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate business impact metrics."""
        # Cost-benefit analysis (simplified)
        cost_per_false_positive = 100  # Cost of investigating false positive
        cost_per_false_negative = 10000  # Cost of missing true positive
        
        fp_cost = ((y_true == 0) & (y_pred == 1)).sum() * cost_per_false_positive
        fn_cost = ((y_true == 1) & (y_pred == 0)).sum() * cost_per_false_negative
        total_cost = fp_cost + fn_cost
        
        # Efficiency metrics
        investigation_efficiency = ((y_true == 1) & (y_pred == 1)).sum() / y_pred.sum() if y_pred.sum() > 0 else 0.0
        
        return {
            'total_cost': total_cost,
            'false_positive_cost': fp_cost,
            'false_negative_cost': fn_cost,
            'investigation_efficiency': investigation_efficiency
        }
    
    def create_evaluation_report(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray, 
        y_proba: np.ndarray,
        model_name: str = "Model"
    ) -> str:
        """Create a comprehensive evaluation report.
        
        Args:
            y_true: True labels
            y_pred: Predictions
            y_proba: Prediction probabilities
            model_name: Name of the model
            
        Returns:
            Formatted evaluation report
        """
        metrics = self.evaluate_model_performance(y_true, y_pred, y_proba, model_name)
        
        report = f"""
AML Detection Model Evaluation Report
====================================
Model: {model_name}
Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

BASIC CLASSIFICATION METRICS
----------------------------
Accuracy:           {metrics['accuracy']:.4f}
Precision:          {metrics['precision']:.4f}
Recall:             {metrics['recall']:.4f}
F1-Score:           {metrics['f1_score']:.4f}
AUC-ROC:            {metrics['auc_roc']:.4f}
AUC-PR:             {metrics['auc_pr']:.4f}

CONFUSION MATRIX
----------------
True Positives:     {metrics['true_positives']}
False Positives:    {metrics['false_positives']}
True Negatives:     {metrics['true_negatives']}
False Negatives:    {metrics['false_negatives']}

AML-SPECIFIC METRICS
--------------------
Case Detection Rate: {metrics['case_detection_rate']:.4f}
Alert Quality:      {metrics['alert_quality']:.4f}
Workload Reduction: {metrics['workload_reduction']:.4f}
False Positive Rate: {metrics['false_positive_rate']:.4f}
False Negative Rate: {metrics['false_negative_rate']:.4f}

PRECISION@K METRICS
-------------------
"""
        
        for k in [10, 25, 50, 100, 200, 500]:
            if f'precision_at_{k}' in metrics:
                report += f"Precision@{k}:         {metrics[f'precision_at_{k}']:.4f}\n"
        
        report += f"""
BUSINESS IMPACT METRICS
-----------------------
Total Cost:         ${metrics['total_cost']:,.2f}
False Positive Cost: ${metrics['false_positive_cost']:,.2f}
False Negative Cost: ${metrics['false_negative_cost']:,.2f}
Investigation Efficiency: {metrics['investigation_efficiency']:.4f}

DISCLAIMER
----------
This evaluation is for research and educational purposes only.
Results should not be used for real-world AML compliance decisions.
"""
        
        return report
    
    def plot_evaluation_curves(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray, 
        y_proba: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None
    ) -> None:
        """Plot evaluation curves.
        
        Args:
            y_true: True labels
            y_pred: Predictions
            y_proba: Prediction probabilities
            model_name: Name of the model
            save_path: Optional path to save plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'AML Detection Model Evaluation: {model_name}', fontsize=16)
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_roc = roc_auc_score(y_true, y_proba)
        
        axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_roc:.2f})')
        axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 0].set_xlim([0.0, 1.0])
        axes[0, 0].set_ylim([0.0, 1.05])
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curve')
        axes[0, 0].legend(loc="lower right")
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        auc_pr = average_precision_score(y_true, y_proba)
        
        axes[0, 1].plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {auc_pr:.2f})')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall Curve')
        axes[0, 1].legend(loc="lower left")
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        axes[1, 0].set_title('Confusion Matrix')
        
        # Precision@K Curve
        k_values = range(10, min(501, len(y_proba)), 10)
        precision_at_k_values = []
        
        for k in k_values:
            top_k_indices = np.argsort(y_proba)[-k:]
            precision_at_k = y_true.iloc[top_k_indices].mean()
            precision_at_k_values.append(precision_at_k)
        
        axes[1, 1].plot(k_values, precision_at_k_values, color='darkorange', lw=2)
        axes[1, 1].set_xlabel('K (Number of Top Predictions)')
        axes[1, 1].set_ylabel('Precision@K')
        axes[1, 1].set_title('Precision@K Curve')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_interactive_evaluation(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray, 
        y_proba: np.ndarray,
        model_name: str = "Model"
    ) -> go.Figure:
        """Create interactive evaluation plots.
        
        Args:
            y_true: True labels
            y_pred: Predictions
            y_proba: Prediction probabilities
            model_name: Name of the model
            
        Returns:
            Plotly figure
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('ROC Curve', 'Precision-Recall Curve', 
                          'Confusion Matrix', 'Precision@K Curve'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_roc = roc_auc_score(y_true, y_proba)
        
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={auc_roc:.2f})',
                      line=dict(color='darkorange', width=2)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random',
                      line=dict(color='navy', width=2, dash='dash')),
            row=1, col=1
        )
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        auc_pr = average_precision_score(y_true, y_proba)
        
        fig.add_trace(
            go.Scatter(x=recall, y=precision, mode='lines', name=f'PR (AUC={auc_pr:.2f})',
                      line=dict(color='darkorange', width=2)),
            row=1, col=2
        )
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        fig.add_trace(
            go.Heatmap(z=cm, text=cm, texttemplate="%{text}", textfont={"size": 16},
                      colorscale='Blues', showscale=False),
            row=2, col=1
        )
        
        # Precision@K Curve
        k_values = list(range(10, min(501, len(y_proba)), 10))
        precision_at_k_values = []
        
        for k in k_values:
            top_k_indices = np.argsort(y_proba)[-k:]
            precision_at_k = y_true.iloc[top_k_indices].mean()
            precision_at_k_values.append(precision_at_k)
        
        fig.add_trace(
            go.Scatter(x=k_values, y=precision_at_k_values, mode='lines',
                      name='Precision@K', line=dict(color='darkorange', width=2)),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f'AML Detection Model Evaluation: {model_name}',
            height=800,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
        fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
        fig.update_xaxes(title_text="Recall", row=1, col=2)
        fig.update_yaxes(title_text="Precision", row=1, col=2)
        fig.update_xaxes(title_text="Predicted", row=2, col=1)
        fig.update_yaxes(title_text="Actual", row=2, col=1)
        fig.update_xaxes(title_text="K", row=2, col=2)
        fig.update_yaxes(title_text="Precision@K", row=2, col=2)
        
        return fig


class AMLBacktester:
    """Backtesting framework for AML detection models."""
    
    def __init__(self) -> None:
        """Initialize the backtester."""
        pass
    
    def time_series_split(
        self, 
        data: pd.DataFrame, 
        n_splits: int = 5,
        test_size: float = 0.2
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Create time-based splits for backtesting.
        
        Args:
            data: Input data with timestamp
            n_splits: Number of splits
            test_size: Size of test set
            
        Returns:
            List of (train, test) splits
        """
        data_sorted = data.sort_values('timestamp')
        splits = []
        
        for i in range(n_splits):
            # Calculate split points
            total_size = len(data_sorted)
            train_size = int(total_size * (1 - test_size))
            
            # Create train and test sets
            train_data = data_sorted.iloc[:train_size]
            test_data = data_sorted.iloc[train_size:]
            
            splits.append((train_data, test_data))
            
            # Move window forward
            window_size = int(total_size * 0.1)  # 10% window movement
            data_sorted = data_sorted.iloc[window_size:]
        
        return splits
    
    def walk_forward_validation(
        self, 
        model: Any, 
        data: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        n_splits: int = 5
    ) -> Dict[str, List[float]]:
        """Perform walk-forward validation.
        
        Args:
            model: AML model to test
            data: Input data
            feature_cols: Feature column names
            target_col: Target column name
            n_splits: Number of splits
            
        Returns:
            Dictionary with validation results
        """
        splits = self.time_series_split(data, n_splits)
        results = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'auc_roc': [],
            'auc_pr': []
        }
        
        evaluator = AMLEvaluator()
        
        for i, (train_data, test_data) in enumerate(splits):
            print(f"Training on split {i+1}/{n_splits}...")
            
            # Prepare data
            X_train = train_data[feature_cols]
            y_train = train_data[target_col]
            X_test = test_data[feature_cols]
            y_test = test_data[target_col]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            
            # Evaluate
            metrics = evaluator.evaluate_model_performance(y_test, y_pred, y_proba[:, 1])
            
            # Store results
            for metric in results.keys():
                results[metric].append(metrics[metric])
        
        return results
    
    def plot_backtest_results(
        self, 
        results: Dict[str, List[float]],
        save_path: Optional[str] = None
    ) -> None:
        """Plot backtest results.
        
        Args:
            results: Backtest results
            save_path: Optional path to save plots
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('AML Model Backtest Results', fontsize=16)
        
        metrics = list(results.keys())
        
        for i, metric in enumerate(metrics):
            row = i // 3
            col = i % 3
            
            axes[row, col].plot(results[metric], marker='o', linewidth=2, markersize=6)
            axes[row, col].set_title(f'{metric.replace("_", " ").title()}')
            axes[row, col].set_xlabel('Split')
            axes[row, col].set_ylabel(metric.replace("_", " ").title())
            axes[row, col].grid(True, alpha=0.3)
            
            # Add mean line
            mean_value = np.mean(results[metric])
            axes[row, col].axhline(y=mean_value, color='red', linestyle='--', alpha=0.7)
            axes[row, col].text(0.02, 0.98, f'Mean: {mean_value:.3f}', 
                              transform=axes[row, col].transAxes, 
                              verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def main() -> None:
    """Test evaluation framework."""
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
    
    # Train a simple model
    from src.models.aml_models import RandomForestAML
    model = RandomForestAML()
    model.fit(X_train, y_train)
    
    # Evaluate
    evaluator = AMLEvaluator()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Create report
    report = evaluator.create_evaluation_report(y_test, y_pred, y_proba[:, 1])
    print(report)
    
    # Plot results
    evaluator.plot_evaluation_curves(y_test, y_pred, y_proba[:, 1])
    
    # Save report
    with open('assets/evaluation_report.txt', 'w') as f:
        f.write(report)


if __name__ == "__main__":
    main()
