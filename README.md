# Anti-Money Laundering Detection System

## ⚠️ IMPORTANT DISCLAIMER

**THIS IS A RESEARCH AND EDUCATIONAL DEMONSTRATION ONLY**

- This system is designed for academic research and educational purposes
- **NOT FOR INVESTMENT ADVICE OR REAL-WORLD AML COMPLIANCE**
- Results may be inaccurate and should not be used for actual financial decisions
- Backtests are hypothetical and do not guarantee future performance
- Always consult qualified financial and compliance professionals for real-world applications

## Overview

This project implements a comprehensive Anti-Money Laundering (AML) detection system using modern machine learning techniques. The system identifies suspicious financial transactions and patterns that may indicate money laundering, terrorist financing, or other illicit activities.

## Features

- **Multiple Detection Models**: Isolation Forest, Autoencoders, Graph-based methods, and traditional ML models
- **Advanced Feature Engineering**: Transaction patterns, network analysis, temporal features
- **Comprehensive Evaluation**: AML-specific metrics including precision@K, case-level analysis
- **Explainable AI**: SHAP explanations for model decisions
- **Interactive Demo**: Streamlit interface for case investigation
- **Network Analysis**: Graph-based suspicious pattern detection

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Synthetic Data**:
   ```bash
   python scripts/generate_data.py
   ```

3. **Train Models**:
   ```bash
   python scripts/train_models.py
   ```

4. **Run Interactive Demo**:
   ```bash
   streamlit run demo/aml_dashboard.py
   ```

## Project Structure

```
├── src/                    # Source code
│   ├── data/              # Data processing and generation
│   ├── features/          # Feature engineering
│   ├── models/            # ML models and training
│   ├── backtest/          # Backtesting framework
│   ├── risk/              # Risk assessment tools
│   └── utils/             # Utility functions
├── configs/               # Configuration files
├── scripts/               # Training and evaluation scripts
├── notebooks/             # Jupyter notebooks for analysis
├── tests/                 # Unit tests
├── assets/                # Generated plots and results
├── demo/                  # Interactive Streamlit demo
└── data/                  # Data storage
```

## Models Implemented

1. **Baseline Models**:
   - Decision Tree
   - Random Forest
   - XGBoost
   - LightGBM

2. **Advanced Models**:
   - Isolation Forest (anomaly detection)
   - Autoencoder (unsupervised)
   - Graph Neural Networks (network analysis)
   - Ensemble methods

## Evaluation Metrics

- **Classification Metrics**: AUC, Precision@K, Recall@K, F1-Score
- **AML-Specific Metrics**: Case-level precision, Investigator workload reduction
- **Business Metrics**: False positive rate, Alert quality score

## Configuration

The system uses Hydra for configuration management. Key configs are in `configs/`:
- `config.yaml`: Main configuration
- `model/`: Model-specific configurations
- `data/`: Data processing configurations

## Data Schema

The system expects transaction data with the following structure:
- `transaction_id`: Unique identifier
- `customer_id`: Customer identifier
- `amount`: Transaction amount
- `timestamp`: Transaction time
- `location`: Transaction location
- `transaction_type`: Type of transaction
- `is_suspicious`: Ground truth label (for supervised learning)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Citation

If you use this code in your research, please cite:

```bibtex
@software{aml_detection_system,
  title={Anti-Money Laundering Detection System},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Anti-Money-Laundering-Detection-System}
}
```
# Anti-Money-Laundering-Detection-System
