# DataSpark — ML/AI Pipelines for Data Science R&D

Production-ready, modular Python toolkit for end-to-end data science workflows: data cleansing, exploratory analysis, statistical testing, ML pipelines, time-series forecasting, deep learning, and sampling theory.

## Architecture

```
dataspark/
├── cleansing/          # Missing values, outliers, type inference, deduplication
├── eda/                # Descriptive stats, correlations, distributions, visualizations
├── statistical/        # Parametric & non-parametric hypothesis testing, effect sizes
├── ml_pipelines/       # Scikit-learn pipelines, model selection, feature engineering
├── timeseries/         # Decomposition, forecasting (ARIMA, ETS), feature extraction
├── deep_learning/      # PyTorch MLP & LSTM with training loop + early stopping
├── sampling/           # Stratified, cluster, systematic, reservoir, bootstrap
├── connectors/         # SQL (SQLAlchemy) & PySpark data I/O
└── utils/              # Validation helpers
```

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run tests
pytest -v --cov=dataspark

# Lint
ruff check dataspark/ tests/
```

## Key Technologies

| Area | Tools |
|------|-------|
| Data manipulation | Pandas, NumPy |
| ML / Pipelines | Scikit-learn (Pipeline, ColumnTransformer, GridSearch) |
| Statistics | SciPy, Statsmodels |
| Deep Learning | PyTorch (MLP, LSTM) |
| Big Data | PySpark, SQLAlchemy |
| Visualization | Matplotlib, Seaborn |
| Testing | Pytest, pytest-cov |
| CI/CD | GitHub Actions, Docker |

## Module Highlights

### Data Cleansing
- **5 imputation strategies**: mean, median, mode, forward-fill, KNN
- **4 outlier methods**: IQR, Z-score, Modified Z-score (MAD), Isolation Forest
- **Automatic type inference**: numeric, datetime, boolean, categorical with memory optimization
- **Deduplication**: exact and fuzzy (bigram Jaccard similarity)

### Exploratory Data Analysis
- Extended summary statistics (skewness, kurtosis, coefficient of variation)
- Pearson, Spearman, Kendall correlations with p-values
- Distribution fitting (7 candidates) ranked by BIC
- Normality tests (Shapiro-Wilk, D'Agostino-Pearson)
- Reusable Matplotlib/Seaborn plot factory

### Statistical Testing
- **Parametric**: t-tests (independent, paired, Welch's), ANOVA, chi-squared, proportion z-test
- **Non-parametric**: Mann-Whitney U, Wilcoxon, Kruskal-Wallis, KS test, Friedman, runs test
- **Effect sizes**: Cohen's d, Cramér's V, eta-squared, power analysis

### ML Pipelines
- Auto-preprocessing (impute + scale + encode) via `ColumnTransformer`
- Model catalog: Logistic Regression, RF, GBM, SVM, Ridge, Lasso, ElasticNet, SVR
- Cross-validation with multiple scoring metrics
- Automated model comparison & hyperparameter search (Grid/Random)
- Feature engineering: interactions, polynomials, log transforms, PCA, SelectKBest

### Time Series
- STL and classical decomposition
- Mann-Kendall trend test & ADF stationarity test
- ARIMA and Exponential Smoothing forecasters
- Rolling, lag, and calendar feature extraction

### Deep Learning
- Tabular MLP with BatchNorm + Dropout
- LSTM forecaster for sequential data
- Training loop: early stopping, LR scheduling, gradient clipping

### Sampling Theory
- Stratified (proportional allocation), cluster, systematic, reservoir sampling
- Bootstrap resampling with 95% CI
- Cochran's sample size calculator with finite population correction

## Docker

```bash
docker build -t dataspark .
docker run dataspark
```

## License

MIT
