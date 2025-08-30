# Tweet Sentiment Stock Returns Dashboard

## Overview
This repository hosts a Streamlit dashboard that visualizes the analysis of tweet sentiment's impact on stock returns. Built from a Kaggle dataset of approximately 1.4 million tweets, the dashboard offers interactive charts, metrics, and insights into sentiment-return correlations, model performance, and feature importance using regression and classification models (e.g., Random Forest).

## Project Details
- **Dataset**: [Tweet Sentiment's Impact on Stock Returns](https://www.kaggle.com/datasets/thedevastator/tweet-sentiment-s-impact-on-stock-returns)
  - Key Metrics: Returns (1/2/3/7-day), Volume, Volatility, LSTM/TextBlob Sentiments
- **Tools**: Python, Streamlit, pandas, numpy, matplotlib, seaborn, plotly, scikit-learn, statsmodels
- **Features**: Interactive scatter plots, correlation heatmaps, model performance visuals, and an insights page

## Files
- `app.py`: Main Streamlit application code.
- `requirements.txt`: List of Python dependencies for running the app.
- `processed_dataset_for_modeling.csv`, `regression_model_results.csv`, `classification_model_results.csv`, `feature_importance_analysis.csv`: Processed data and model outputs.

## Hosting
- **Google Colab**: Interactive version of the notebook is available [here](https://colab.research.google.com/drive/1YsgBPirhSuGO4mXYXK6x-2MtqZW7_T1Y#scrollTo=cbf17cb6-5f25-4bb7-bce4-866d28f4c662). Run cells to explore the analysis live.
- **Streamlit Dashboard**: Hosted version will be available [here](https://tweet-sentiment-impact-on-stock-returns-dashboard-7mbtw9qmw8xh.streamlit.app/).
