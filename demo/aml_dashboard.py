"""Streamlit demo application for AML detection system."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.aml_models import (
    DecisionTreeAML, RandomForestAML, XGBoostAML, 
    LightGBMAML, IsolationForestAML, AutoEncoderAML, AMLEnsemble
)
from src.utils.explainability import AMLExplainer
from src.backtest.evaluation import AMLEvaluator


# Page configuration
st.set_page_config(
    page_title="AML Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🔍 Anti-Money Laundering Detection System</h1>', unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
    <h4>⚠️ IMPORTANT DISCLAIMER</h4>
    <p><strong>This is a research and educational demonstration only.</strong></p>
    <ul>
        <li>NOT FOR REAL-WORLD AML COMPLIANCE OR REGULATORY REPORTING</li>
        <li>Results should NOT be used for making financial decisions</li>
        <li>This is NOT investment advice</li>
        <li>Backtests are hypothetical and do not guarantee future performance</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Select Page",
    ["Overview", "Data Explorer", "Model Training", "Model Evaluation", "Case Investigation", "Network Analysis"]
)

# Load data function
@st.cache_data
def load_data():
    """Load AML data."""
    try:
        customers = pd.read_csv('data/customers.csv')
        transactions = pd.read_csv('data/transactions.csv')
        relationships = pd.read_csv('data/relationships.csv')
        features = pd.read_csv('data/features.csv')
        return customers, transactions, relationships, features
    except FileNotFoundError:
        st.error("Data files not found. Please run the data generation script first.")
        return None, None, None, None

# Load data
customers, transactions, relationships, features = load_data()

if customers is not None:
    
    if page == "Overview":
        st.header("System Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", f"{len(customers):,}")
        
        with col2:
            st.metric("Total Transactions", f"{len(transactions):,}")
        
        with col3:
            suspicious_rate = transactions['is_suspicious'].mean()
            st.metric("Suspicious Rate", f"{suspicious_rate:.2%}")
        
        with col4:
            st.metric("Network Relationships", f"{len(relationships):,}")
        
        # Key statistics
        st.subheader("Key Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Transaction Amount Distribution")
            fig = px.histogram(
                transactions, 
                x='amount', 
                nbins=50,
                title="Transaction Amount Distribution",
                labels={'amount': 'Amount (USD)', 'count': 'Count'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Suspicious Transactions by Type")
            suspicious_by_type = transactions.groupby('transaction_type')['is_suspicious'].mean().reset_index()
            fig = px.bar(
                suspicious_by_type,
                x='transaction_type',
                y='is_suspicious',
                title="Suspicious Rate by Transaction Type",
                labels={'is_suspicious': 'Suspicious Rate', 'transaction_type': 'Transaction Type'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Model performance summary
        st.subheader("Model Performance Summary")
        
        # Create sample model results
        models = ['Decision Tree', 'Random Forest', 'XGBoost', 'LightGBM', 'Isolation Forest', 'AutoEncoder']
        performance_data = {
            'Model': models,
            'Accuracy': [0.85, 0.92, 0.94, 0.93, 0.88, 0.89],
            'Precision': [0.78, 0.89, 0.91, 0.90, 0.82, 0.84],
            'Recall': [0.82, 0.88, 0.90, 0.89, 0.85, 0.87],
            'F1-Score': [0.80, 0.88, 0.90, 0.89, 0.83, 0.85],
            'AUC-ROC': [0.87, 0.94, 0.96, 0.95, 0.90, 0.91]
        }
        
        performance_df = pd.DataFrame(performance_data)
        
        fig = go.Figure()
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']:
            fig.add_trace(go.Scatter(
                x=performance_df['Model'],
                y=performance_df[metric],
                mode='lines+markers',
                name=metric,
                line=dict(width=3)
            ))
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Model",
            yaxis_title="Score",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Data Explorer":
        st.header("Data Explorer")
        
        # Data overview
        st.subheader("Data Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Customer Demographics")
            age_dist = customers['age'].value_counts().sort_index()
            fig = px.bar(
                x=age_dist.index,
                y=age_dist.values,
                title="Age Distribution",
                labels={'x': 'Age', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Risk Score Distribution")
            fig = px.histogram(
                customers,
                x='risk_score',
                nbins=30,
                title="Risk Score Distribution",
                labels={'risk_score': 'Risk Score', 'count': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Transaction patterns
        st.subheader("Transaction Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Transactions Over Time")
            transactions['timestamp'] = pd.to_datetime(transactions['timestamp'])
            daily_transactions = transactions.groupby(transactions['timestamp'].dt.date).size().reset_index()
            daily_transactions.columns = ['date', 'count']
            
            fig = px.line(
                daily_transactions,
                x='date',
                y='count',
                title="Daily Transaction Volume",
                labels={'date': 'Date', 'count': 'Transaction Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Suspicious Transactions Over Time")
            daily_suspicious = transactions.groupby(transactions['timestamp'].dt.date)['is_suspicious'].sum().reset_index()
            daily_suspicious.columns = ['date', 'suspicious_count']
            
            fig = px.line(
                daily_suspicious,
                x='date',
                y='suspicious_count',
                title="Daily Suspicious Transactions",
                labels={'date': 'Date', 'suspicious_count': 'Suspicious Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Interactive filters
        st.subheader("Interactive Data Filters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_amount = st.number_input("Minimum Amount", min_value=0, value=0)
            max_amount = st.number_input("Maximum Amount", min_value=min_amount, value=100000)
        
        with col2:
            transaction_types = st.multiselect(
                "Transaction Types",
                options=transactions['transaction_type'].unique(),
                default=transactions['transaction_type'].unique()
            )
        
        with col3:
            countries = st.multiselect(
                "Countries",
                options=customers['country'].unique(),
                default=customers['country'].unique()
            )
        
        # Apply filters
        filtered_transactions = transactions[
            (transactions['amount'] >= min_amount) &
            (transactions['amount'] <= max_amount) &
            (transactions['transaction_type'].isin(transaction_types))
        ]
        
        filtered_customers = customers[customers['country'].isin(countries)]
        filtered_transactions = filtered_transactions[
            filtered_transactions['customer_id'].isin(filtered_customers['customer_id'])
        ]
        
        st.subheader("Filtered Data Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Filtered Transactions", f"{len(filtered_transactions):,}")
        
        with col2:
            st.metric("Filtered Customers", f"{len(filtered_customers):,}")
        
        with col3:
            if len(filtered_transactions) > 0:
                suspicious_rate = filtered_transactions['is_suspicious'].mean()
                st.metric("Suspicious Rate", f"{suspicious_rate:.2%}")
            else:
                st.metric("Suspicious Rate", "N/A")
        
        with col4:
            if len(filtered_transactions) > 0:
                avg_amount = filtered_transactions['amount'].mean()
                st.metric("Average Amount", f"${avg_amount:,.2f}")
            else:
                st.metric("Average Amount", "N/A")
    
    elif page == "Model Training":
        st.header("Model Training")
        
        st.subheader("Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Select Model Type",
                ["Decision Tree", "Random Forest", "XGBoost", "LightGBM", "Isolation Forest", "AutoEncoder"]
            )
        
        with col2:
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
        
        # Model parameters
        st.subheader("Model Parameters")
        
        if model_type == "Random Forest":
            n_estimators = st.slider("Number of Estimators", 10, 200, 100)
            max_depth = st.slider("Max Depth", 3, 20, 10)
            min_samples_split = st.slider("Min Samples Split", 2, 20, 5)
        
        elif model_type == "XGBoost":
            n_estimators = st.slider("Number of Estimators", 10, 200, 100)
            max_depth = st.slider("Max Depth", 3, 10, 6)
            learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01)
        
        elif model_type == "LightGBM":
            n_estimators = st.slider("Number of Estimators", 10, 200, 100)
            max_depth = st.slider("Max Depth", 3, 10, 6)
            learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01)
        
        # Train model
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                # Prepare data
                target_cols = ['is_suspicious', 'transaction_id', 'customer_id']
                X = features.drop(columns=target_cols)
                y = features['is_suspicious']
                
                # Split data
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
                
                # Initialize model
                if model_type == "Decision Tree":
                    model = DecisionTreeAML()
                elif model_type == "Random Forest":
                    model = RandomForestAML(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
                elif model_type == "XGBoost":
                    model = XGBoostAML(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
                elif model_type == "LightGBM":
                    model = LightGBMAML(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
                elif model_type == "Isolation Forest":
                    model = IsolationForestAML()
                elif model_type == "AutoEncoder":
                    model = AutoEncoderAML()
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)
                
                # Evaluate
                evaluator = AMLEvaluator()
                metrics = evaluator.evaluate_model_performance(y_test, y_pred, y_proba[:, 1])
                
                # Display results
                st.success("Model training completed!")
                
                # Performance metrics
                st.subheader("Performance Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                
                with col2:
                    st.metric("Precision", f"{metrics['precision']:.3f}")
                
                with col3:
                    st.metric("Recall", f"{metrics['recall']:.3f}")
                
                with col4:
                    st.metric("F1-Score", f"{metrics['f1_score']:.3f}")
                
                # Additional metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("AUC-ROC", f"{metrics['auc_roc']:.3f}")
                
                with col2:
                    st.metric("AUC-PR", f"{metrics['auc_pr']:.3f}")
                
                with col3:
                    st.metric("Case Detection Rate", f"{metrics['case_detection_rate']:.3f}")
                
                with col4:
                    st.metric("Alert Quality", f"{metrics['alert_quality']:.3f}")
                
                # Store model in session state
                st.session_state['trained_model'] = model
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                st.session_state['y_pred'] = y_pred
                st.session_state['y_proba'] = y_proba
                st.session_state['metrics'] = metrics
    
    elif page == "Model Evaluation":
        st.header("Model Evaluation")
        
        if 'trained_model' not in st.session_state:
            st.warning("Please train a model first in the 'Model Training' page.")
        else:
            model = st.session_state['trained_model']
            X_test = st.session_state['X_test']
            y_test = st.session_state['y_test']
            y_pred = st.session_state['y_pred']
            y_proba = st.session_state['y_proba']
            metrics = st.session_state['metrics']
            
            # Evaluation metrics
            st.subheader("Evaluation Metrics")
            
            # Create metrics comparison
            metrics_data = {
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'AUC-PR'],
                'Value': [
                    metrics['accuracy'],
                    metrics['precision'],
                    metrics['recall'],
                    metrics['f1_score'],
                    metrics['auc_roc'],
                    metrics['auc_pr']
                ]
            }
            
            metrics_df = pd.DataFrame(metrics_data)
            
            fig = px.bar(
                metrics_df,
                x='Metric',
                y='Value',
                title="Model Performance Metrics",
                color='Value',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # AML-specific metrics
            st.subheader("AML-Specific Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Precision@K
                k_values = [10, 25, 50, 100, 200, 500]
                precision_at_k = []
                
                for k in k_values:
                    if f'precision_at_{k}' in metrics:
                        precision_at_k.append(metrics[f'precision_at_{k}'])
                    else:
                        precision_at_k.append(0.0)
                
                fig = px.line(
                    x=k_values,
                    y=precision_at_k,
                    title="Precision@K",
                    labels={'x': 'K', 'y': 'Precision@K'},
                    markers=True
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Business metrics
                business_metrics = {
                    'Metric': ['Case Detection Rate', 'Alert Quality', 'Workload Reduction'],
                    'Value': [
                        metrics['case_detection_rate'],
                        metrics['alert_quality'],
                        metrics['workload_reduction']
                    ]
                }
                
                business_df = pd.DataFrame(business_metrics)
                
                fig = px.bar(
                    business_df,
                    x='Metric',
                    y='Value',
                    title="Business Impact Metrics",
                    color='Value',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Confusion matrix
            st.subheader("Confusion Matrix")
            
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            
            fig = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                title="Confusion Matrix",
                labels=dict(x="Predicted", y="Actual", color="Count")
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # ROC and PR curves
            st.subheader("Performance Curves")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # ROC Curve
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode='lines',
                    name=f'ROC Curve (AUC = {metrics["auc_roc"]:.3f})',
                    line=dict(color='darkorange', width=3)
                ))
                fig.add_trace(go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode='lines',
                    name='Random',
                    line=dict(color='navy', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title="ROC Curve",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Precision-Recall Curve
                from sklearn.metrics import precision_recall_curve
                precision, recall, _ = precision_recall_curve(y_test, y_proba[:, 1])
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=recall,
                    y=precision,
                    mode='lines',
                    name=f'PR Curve (AUC = {metrics["auc_pr"]:.3f})',
                    line=dict(color='darkorange', width=3)
                ))
                
                fig.update_layout(
                    title="Precision-Recall Curve",
                    xaxis_title="Recall",
                    yaxis_title="Precision",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Case Investigation":
        st.header("Case Investigation")
        
        if 'trained_model' not in st.session_state:
            st.warning("Please train a model first in the 'Model Training' page.")
        else:
            model = st.session_state['trained_model']
            X_test = st.session_state['X_test']
            y_test = st.session_state['y_test']
            y_pred = st.session_state['y_pred']
            y_proba = st.session_state['y_proba']
            
            # Case selection
            st.subheader("Select Case for Investigation")
            
            # Filter suspicious cases
            suspicious_mask = y_test == 1
            suspicious_indices = X_test[suspicious_mask].index
            
            if len(suspicious_indices) > 0:
                case_idx = st.selectbox(
                    "Select Case Index",
                    options=suspicious_indices[:20],  # Show first 20
                    format_func=lambda x: f"Case {x} (Suspicious: {y_test.iloc[x]}, Predicted: {y_pred[x]}, Probability: {y_proba[x, 1]:.3f})"
                )
                
                # Case details
                st.subheader("Case Details")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("True Label", "Suspicious" if y_test.iloc[case_idx] == 1 else "Normal")
                
                with col2:
                    st.metric("Predicted Label", "Suspicious" if y_pred[case_idx] == 1 else "Normal")
                
                with col3:
                    st.metric("Suspicious Probability", f"{y_proba[case_idx, 1]:.3f}")
                
                # Feature values
                st.subheader("Feature Values")
                
                case_features = X_test.iloc[case_idx]
                
                # Create feature importance plot
                if hasattr(model, 'get_feature_importance'):
                    feature_importance = model.get_feature_importance()
                    
                    fig = px.bar(
                        feature_importance.head(15),
                        x='importance',
                        y='feature',
                        orientation='h',
                        title="Feature Importance (Top 15)",
                        labels={'importance': 'Importance', 'feature': 'Feature'}
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Feature values table
                st.subheader("Feature Values for Selected Case")
                
                feature_df = pd.DataFrame({
                    'Feature': case_features.index,
                    'Value': case_features.values
                }).sort_values('Value', key=abs, ascending=False)
                
                st.dataframe(feature_df, use_container_width=True)
                
                # SHAP explanation
                st.subheader("SHAP Explanation")
                
                if st.button("Generate SHAP Explanation"):
                    with st.spinner("Generating SHAP explanation..."):
                        try:
                            # Create explainer
                            explainer = AMLExplainer(model, X_test)
                            
                            # Create interactive explanation
                            fig = explainer.create_interactive_explanation(X_test, case_idx)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Feature attribution summary
                            feature_summary = explainer.create_feature_attribution_summary(X_test, y_test)
                            
                            st.subheader("Feature Attribution Summary")
                            st.dataframe(feature_summary, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Error generating SHAP explanation: {e}")
            
            else:
                st.warning("No suspicious cases found in the test set.")
    
    elif page == "Network Analysis":
        st.header("Network Analysis")
        
        # Network overview
        st.subheader("Network Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Relationships", f"{len(relationships):,}")
        
        with col2:
            avg_connections = relationships.groupby('customer_1').size().mean()
            st.metric("Average Connections", f"{avg_connections:.1f}")
        
        with col3:
            max_connections = relationships.groupby('customer_1').size().max()
            st.metric("Max Connections", f"{max_connections}")
        
        # Network visualization
        st.subheader("Network Visualization")
        
        # Sample network for visualization
        sample_size = st.slider("Sample Size for Visualization", 10, 100, 50)
        
        # Get top connected customers
        top_customers = relationships.groupby('customer_1').size().nlargest(sample_size).index
        
        # Filter relationships for top customers
        sample_relationships = relationships[
            relationships['customer_1'].isin(top_customers) |
            relationships['customer_2'].isin(top_customers)
        ]
        
        if len(sample_relationships) > 0:
            # Create network graph
            import networkx as nx
            
            G = nx.Graph()
            
            # Add nodes
            for customer in top_customers:
                G.add_node(customer)
            
            # Add edges
            for _, row in sample_relationships.iterrows():
                G.add_edge(row['customer_1'], row['customer_2'], weight=row['strength'])
            
            # Create plotly network visualization
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Extract edge information
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            # Extract node information
            node_x = []
            node_y = []
            node_text = []
            node_size = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
                
                # Size based on degree
                degree = G.degree(node)
                node_size.append(degree * 10 + 10)
            
            # Create edge trace
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
            
            # Create node trace
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                text=node_text,
                marker=dict(
                    showscale=True,
                    colorscale='YlOrRd',
                    reversescale=True,
                    color=node_size,
                    size=node_size,
                    colorbar=dict(
                        thickness=15,
                        title="Connections",
                        xanchor="left",
                        titleside="right"
                    ),
                    line=dict(width=2)
                )
            )
            
            # Create figure
            fig = go.Figure(data=[edge_trace, node_trace],
                           layout=go.Layout(
                               title='Customer Network (Sample)',
                               titlefont_size=16,
                               showlegend=False,
                               hovermode='closest',
                               margin=dict(b=20,l=5,r=5,t=40),
                               annotations=[ dict(
                                   text="Network visualization of customer relationships",
                                   showarrow=False,
                                   xref="paper", yref="paper",
                                   x=0.005, y=-0.002,
                                   xanchor="left", yanchor="bottom",
                                   font=dict(color="black", size=12)
                               )],
                               xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                           )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Network statistics
        st.subheader("Network Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Degree distribution
            degree_dist = relationships.groupby('customer_1').size().value_counts().sort_index()
            
            fig = px.bar(
                x=degree_dist.index,
                y=degree_dist.values,
                title="Degree Distribution",
                labels={'x': 'Number of Connections', 'y': 'Number of Customers'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Connection strength distribution
            fig = px.histogram(
                relationships,
                x='strength',
                nbins=30,
                title="Connection Strength Distribution",
                labels={'strength': 'Connection Strength', 'count': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Suspicious network analysis
        st.subheader("Suspicious Network Analysis")
        
        # Merge with transaction data to identify suspicious customers
        suspicious_customers = transactions[transactions['is_suspicious'] == 1]['customer_id'].unique()
        
        # Analyze suspicious customer networks
        suspicious_relationships = relationships[
            relationships['customer_1'].isin(suspicious_customers) |
            relationships['customer_2'].isin(suspicious_customers)
        ]
        
        if len(suspicious_relationships) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Suspicious Customers", f"{len(suspicious_customers):,}")
            
            with col2:
                st.metric("Suspicious Relationships", f"{len(suspicious_relationships):,}")
            
            # Suspicious network visualization
            st.subheader("Suspicious Customer Network")
            
            # Create network for suspicious customers
            G_suspicious = nx.Graph()
            
            # Add suspicious customers as nodes
            for customer in suspicious_customers:
                G_suspicious.add_node(customer, suspicious=True)
            
            # Add their connections
            for _, row in suspicious_relationships.iterrows():
                G_suspicious.add_edge(row['customer_1'], row['customer_2'], weight=row['strength'])
            
            if len(G_suspicious.nodes()) > 0:
                # Create visualization
                pos = nx.spring_layout(G_suspicious, k=1, iterations=50)
                
                # Extract edge information
                edge_x = []
                edge_y = []
                for edge in G_suspicious.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                
                # Extract node information
                node_x = []
                node_y = []
                node_text = []
                node_size = []
                
                for node in G_suspicious.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(f"{node} (Suspicious)")
                    
                    # Size based on degree
                    degree = G_suspicious.degree(node)
                    node_size.append(degree * 15 + 15)
                
                # Create edge trace
                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=1, color='red'),
                    hoverinfo='none',
                    mode='lines'
                )
                
                # Create node trace
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers',
                    hoverinfo='text',
                    text=node_text,
                    marker=dict(
                        color='red',
                        size=node_size,
                        line=dict(width=2, color='darkred')
                    )
                )
                
                # Create figure
                fig = go.Figure(data=[edge_trace, node_trace],
                               layout=go.Layout(
                                   title='Suspicious Customer Network',
                                   titlefont_size=16,
                                   showlegend=False,
                                   hovermode='closest',
                                   margin=dict(b=20,l=5,r=5,t=40),
                                   xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                   yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                               )
                
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("No suspicious customer relationships found in the network.")

else:
    st.error("Data not available. Please run the data generation script first.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    <p>AML Detection System - Research and Educational Demonstration Only</p>
    <p>⚠️ NOT FOR REAL-WORLD AML COMPLIANCE OR INVESTMENT ADVICE</p>
</div>
""", unsafe_allow_html=True)
