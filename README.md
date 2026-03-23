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
├── visualization/      # Static charts, interactive sliders, multi-panel dashboards
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
| Visualization | Matplotlib, Seaborn, ipywidgets (interactive sliders) |
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

### Visualization
- **ChartBuilder**: 25+ static chart types covering all data domains
  - Data profiling: missing matrix, missing bars, outlier scatter, before/after cleaning
  - EDA: distributions with fitted overlays, correlation heatmaps, Q-Q plots, categorical bars
  - Statistics: p-value forest plots, effect-size bars, group comparisons (box/violin/strip)
  - ML: model comparison bars, feature importance, confusion matrix, PCA scree, residual analysis
  - Time series: line plots with rolling mean, decomposition panels, forecast with CI bands, ACF/PACF
  - Sampling: bootstrap CI distributions, sample-size curves, strata proportion comparison
- **Dashboard**: Multi-panel composite dashboards (data quality, EDA overview, model report, time-series report)
- **InteractiveExplorer** (Jupyter): ipywidgets-powered parameter exploration with live sliders
  - Distribution explorer: column selector, bin slider, KDE toggle, distribution fit overlay
  - Correlation explorer: method dropdown, minimum |r| threshold slider, annotation toggle
  - Outlier explorer: method selector, threshold slider with live detection preview
  - Missing-value explorer: imputation strategy selector with before/after distribution preview
  - Scatter explorer: axis selectors, hue dropdown, alpha/size sliders, regression line toggle
  - Time-series explorer: rolling window slider, period slider, decomposition toggle
  - Sampling explorer: stratum selector, method dropdown, fraction slider with proportion comparison
  - Hypothesis test explorer: column/test selectors, sample-size slider, effect-size gauge
  - `launch()`: Tabbed interface combining all explorers

```python
# Static charts
from dataspark.visualization import ChartBuilder
cb = ChartBuilder()
fig = cb.distribution(df["revenue"], bins=40, fit_dist="norm")
fig = cb.correlation_heatmap(df, method="spearman")
fig = cb.model_comparison(results_df)
cb.save(fig, "output.png")

# Multi-panel dashboards
from dataspark.visualization import Dashboard
dash = Dashboard()
fig = dash.data_quality(df)
fig = dash.eda_overview(df)

# Interactive exploration in Jupyter (pip install dataspark[notebooks])
from dataspark.visualization.interactive import InteractiveExplorer
explorer = InteractiveExplorer(df)
explorer.launch()          # tabbed interface with all explorers
explorer.explore_outliers() # individual explorer with sliders
```

## Docker

```bash
docker build -t dataspark .
docker run dataspark
```

## License

MIT
