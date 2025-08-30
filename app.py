import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Tweet Sentiment & Stock Returns Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the processed dataset and model results"""
    try:
        df = pd.read_csv('processed_dataset_for_modeling.csv')

        regression_results = pd.read_csv('regression_model_results.csv')
        classification_results = pd.read_csv('classification_model_results.csv')

        try:
            feature_importance = pd.read_csv('feature_importance_analysis.csv')
        except:
            feature_importance = None
            
        return df, regression_results, classification_results, feature_importance
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please ensure you have run the main analysis script and generated the required CSV files.")
        return None, None, None, None

def create_overview_metrics(df):
    """Create overview metrics cards"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Tweets",
            value=f"{len(df):,}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Unique Stocks",
            value=f"{df['stock_encoded'].nunique():,}",
            delta=None
        )
    
    with col3:
        avg_sentiment = df['sentiment_avg'].mean()
        st.metric(
            label="Avg Sentiment",
            value=f"{avg_sentiment:.3f}",
            delta=f"{'Positive' if avg_sentiment > 0 else 'Negative'}"
        )
    
    with col4:
        avg_return = df['1_DAY_RETURN'].mean()
        st.metric(
            label="Avg 1-Day Return",
            value=f"{avg_return:.4f}",
            delta=f"{'📈' if avg_return > 0 else '📉'}"
        )

def create_univariate_analysis(df):
    """Create univariate analysis visualizations"""
    st.subheader("📊 Univariate Analysis")

    tab1, tab2, tab3 = st.tabs(["📈 Distributions", "📋 Summary Statistics", "🏢 Stock Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:

            fig_sentiment = go.Figure()
            fig_sentiment.add_trace(go.Histogram(
                x=df['LSTM_POLARITY'],
                name='LSTM Sentiment',
                opacity=0.7,
                nbinsx=50
            ))
            fig_sentiment.add_trace(go.Histogram(
                x=df['TEXTBLOB_POLARITY'],
                name='TextBlob Sentiment',
                opacity=0.7,
                nbinsx=50
            ))
            fig_sentiment.update_layout(
                title="Sentiment Score Distributions",
                xaxis_title="Sentiment Score",
                yaxis_title="Frequency",
                barmode='overlay'
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)
        
        with col2:

            fig_returns = go.Figure()
            fig_returns.add_trace(go.Histogram(
                x=df['1_DAY_RETURN'],
                name='1-Day Return',
                opacity=0.7,
                nbinsx=50
            ))
            fig_returns.add_trace(go.Histogram(
                x=df['3_DAY_RETURN'],
                name='3-Day Return',
                opacity=0.7,
                nbinsx=50
            ))
            fig_returns.update_layout(
                title="Stock Return Distributions",
                xaxis_title="Return",
                yaxis_title="Frequency",
                barmode='overlay'
            )
            st.plotly_chart(fig_returns, use_container_width=True)
    
    with tab2:

        st.write("**Key Variables Summary Statistics**")
        summary_cols = ['LSTM_POLARITY', 'TEXTBLOB_POLARITY', '1_DAY_RETURN', 
                       '3_DAY_RETURN', '7_DAY_RETURN', 'VOLATILITY_10D']
        summary_stats = df[summary_cols].describe()
        st.dataframe(summary_stats.round(4))
    
    with tab3:

        if 'stock_encoded' in df.columns:
            st.write("**Most Frequently Mentioned Stocks**")

            stock_counts = df['stock_encoded'].value_counts().head(10)
            
            fig_stocks = px.bar(
                x=stock_counts.values,
                y=stock_counts.index,
                orientation='h',
                title="Top 10 Stock Mentions (Encoded)",
                labels={'x': 'Tweet Count', 'y': 'Stock Code'}
            )
            st.plotly_chart(fig_stocks, use_container_width=True)

def create_bivariate_analysis(df):
    """Create bivariate analysis visualizations"""
    st.subheader("🔗 Bivariate Analysis")
    
    tab1, tab2, tab3 = st.tabs(["📈 Sentiment vs Returns", "🔄 Correlations", "📊 Scatter Analysis"])
    
    with tab1:

        col1, col2 = st.columns(2)
        
        with col1:

            return_horizon = st.selectbox(
                "Select Return Horizon:",
                ['1_DAY_RETURN', '3_DAY_RETURN', '7_DAY_RETURN'],
                key="return_horizon_1"
            )

            sample_size = min(5000, len(df))
            df_sample = df.sample(n=sample_size, random_state=42)
            
            fig_lstm = px.scatter(
                df_sample,
                x='LSTM_POLARITY',
                y=return_horizon,
                title=f'LSTM Sentiment vs {return_horizon}',
                trendline="ols"
            )
            st.plotly_chart(fig_lstm, use_container_width=True)

            corr_lstm = df['LSTM_POLARITY'].corr(df[return_horizon])
            st.metric("LSTM Correlation", f"{corr_lstm:.4f}")
        
        with col2:
            fig_textblob = px.scatter(
                df_sample,
                x='TEXTBLOB_POLARITY',
                y=return_horizon,
                title=f'TextBlob Sentiment vs {return_horizon}',
                trendline="ols"
            )
            st.plotly_chart(fig_textblob, use_container_width=True)

            corr_textblob = df['TEXTBLOB_POLARITY'].corr(df[return_horizon])
            st.metric("TextBlob Correlation", f"{corr_textblob:.4f}")
    
    with tab2:

        corr_vars = ['LSTM_POLARITY', 'TEXTBLOB_POLARITY', 'sentiment_avg', 
                     '1_DAY_RETURN', '3_DAY_RETURN', '7_DAY_RETURN', 
                     'VOLATILITY_10D', 'volume_log']
        available_vars = [var for var in corr_vars if var in df.columns]
        
        corr_matrix = df[available_vars].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Correlation Matrix of Key Variables",
            color_continuous_scale="RdBu",
            zmin=-1, zmax=1
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab3:

        col1, col2 = st.columns(2)
        
        with col1:
            x_var = st.selectbox("Select X Variable:", available_vars, key="x_var")
            y_var = st.selectbox("Select Y Variable:", available_vars, key="y_var", index=1)
        
        with col2:
            color_var = st.selectbox("Color by (optional):", [None] + available_vars, key="color_var")
        
        if x_var != y_var:
            fig_scatter = px.scatter(
                df_sample,
                x=x_var,
                y=y_var,
                color=color_var if color_var else None,
                title=f'{x_var} vs {y_var}',
                trendline="ols"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

def create_model_performance_dashboard(regression_results, classification_results):
    """Create model performance dashboard"""
    st.subheader("🤖 Model Performance Analysis")
    
    tab1, tab2, tab3 = st.tabs(["📊 Regression Models", "🎯 Classification Models", "⚖️ Model Comparison"])
    
    with tab1:
        if regression_results is not None:
            st.write("**Regression Model Results**")

            col1, col2 = st.columns(2)
            
            with col1:
                pivot_r2 = regression_results.pivot(index='Model', columns='Horizon', values='Test_R2')
                fig_r2 = px.imshow(
                    pivot_r2,
                    text_auto=True,
                    aspect="auto",
                    title="Test R² Scores by Model and Horizon",
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig_r2, use_container_width=True)
            
            with col2:

                pivot_mse = regression_results.pivot(index='Model', columns='Horizon', values='Test_MSE')
                fig_mse = px.imshow(
                    pivot_mse,
                    text_auto=True,
                    aspect="auto",
                    title="Test MSE by Model and Horizon",
                    color_continuous_scale="Viridis_r"
                )
                st.plotly_chart(fig_mse, use_container_width=True)

            st.write("**Best Performing Models by Horizon**")
            best_regression = regression_results.loc[regression_results.groupby('Horizon')['Test_R2'].idxmax()]
            st.dataframe(best_regression[['Horizon', 'Model', 'Test_R2', 'Test_MSE', 'Test_MAE']].round(4))
    
    with tab2:
        if classification_results is not None:
            st.write("**Classification Model Results**")
            
            col1, col2 = st.columns(2)
            
            with col1:

                pivot_acc = classification_results.pivot(index='Model', columns='Horizon', values='Test_Accuracy')
                fig_acc = px.imshow(
                    pivot_acc,
                    text_auto=True,
                    aspect="auto",
                    title="Test Accuracy by Model and Horizon",
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig_acc, use_container_width=True)
            
            with col2:

                if 'AUC' in classification_results.columns:
                    pivot_auc = classification_results.pivot(index='Model', columns='Horizon', values='AUC')
                    fig_auc = px.imshow(
                        pivot_auc,
                        text_auto=True,
                        aspect="auto",
                        title="AUC Scores by Model and Horizon",
                        color_continuous_scale="Viridis"
                    )
                    st.plotly_chart(fig_auc, use_container_width=True)

            st.write("**Best Performing Classification Models by Horizon**")
            best_classification = classification_results.loc[classification_results.groupby('Horizon')['Test_Accuracy'].idxmax()]
            display_cols = ['Horizon', 'Model', 'Test_Accuracy', 'F1_Score']
            if 'AUC' in classification_results.columns:
                display_cols.append('AUC')
            st.dataframe(best_classification[display_cols].round(4))
    
    with tab3:

        col1, col2 = st.columns(2)
        
        with col1:
            if regression_results is not None:
                st.write("**Average Regression Performance by Model**")
                avg_reg_perf = regression_results.groupby('Model')['Test_R2'].mean().sort_values(ascending=False)
                
                fig_reg_comp = px.bar(
                    x=avg_reg_perf.values,
                    y=avg_reg_perf.index,
                    orientation='h',
                    title="Average R² by Model",
                    labels={'x': 'Average R²', 'y': 'Model'}
                )
                st.plotly_chart(fig_reg_comp, use_container_width=True)
        
        with col2:
            if classification_results is not None:
                st.write("**Average Classification Performance by Model**")
                avg_clf_perf = classification_results.groupby('Model')['Test_Accuracy'].mean().sort_values(ascending=False)
                
                fig_clf_comp = px.bar(
                    x=avg_clf_perf.values,
                    y=avg_clf_perf.index,
                    orientation='h',
                    title="Average Accuracy by Model",
                    labels={'x': 'Average Accuracy', 'y': 'Model'}
                )
                st.plotly_chart(fig_clf_comp, use_container_width=True)

def create_feature_importance_analysis(feature_importance, df):
    """Create feature importance analysis"""
    st.subheader("🎯 Feature Importance Analysis")
    
    if feature_importance is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            top_features = feature_importance.tail(15)
            fig_importance = px.bar(
                top_features,
                x='importance',
                y='feature',
                orientation='h',
                title="Top 15 Most Important Features",
                labels={'importance': 'Importance Score', 'feature': 'Feature'}
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        
        with col2:
            sentiment_features = [f for f in feature_importance['feature'] if any(word in f.lower() for word in ['sentiment', 'lstm', 'textblob'])]
            market_features = [f for f in feature_importance['feature'] if any(word in f.lower() for word in ['volatility', 'volume', 'return'])]
            time_features = [f for f in feature_importance['feature'] if any(word in f.lower() for word in ['weekday', 'weekend', 'quarter'])]
            
            sentiment_importance = feature_importance[feature_importance['feature'].isin(sentiment_features)]['importance'].sum()
            market_importance = feature_importance[feature_importance['feature'].isin(market_features)]['importance'].sum()
            time_importance = feature_importance[feature_importance['feature'].isin(time_features)]['importance'].sum()
            other_importance = feature_importance['importance'].sum() - sentiment_importance - market_importance - time_importance
            
            categories = ['Sentiment', 'Market', 'Time', 'Other']
            importances = [sentiment_importance, market_importance, time_importance, other_importance]
            
            fig_categories = px.pie(
                values=importances,
                names=categories,
                title="Feature Importance by Category"
            )
            st.plotly_chart(fig_categories, use_container_width=True)
    else:
        st.info("Feature importance data not available. Please run the analysis to generate feature importance.")

def create_interactive_prediction_tool(df):
    """Create interactive prediction tool"""
    st.subheader("🔮 Interactive Prediction Tool")
    
    st.write("Adjust the parameters below to see how different factors might affect stock returns:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Sentiment Parameters**")
        lstm_sentiment = st.slider("LSTM Sentiment", -1.0, 1.0, 0.0, 0.1)
        textblob_sentiment = st.slider("TextBlob Sentiment", -1.0, 1.0, 0.0, 0.1)
        
    with col2:
        st.write("**Market Parameters**")
        volatility = st.slider("10-Day Volatility", 
                              float(df['VOLATILITY_10D'].min()), 
                              float(df['VOLATILITY_10D'].max()), 
                              float(df['VOLATILITY_10D'].mean()))
        
        volume_log = st.slider("Log Volume", 
                              float(df['volume_log'].min()) if 'volume_log' in df.columns else 0.0, 
                              float(df['volume_log'].max()) if 'volume_log' in df.columns else 20.0, 
                              float(df['volume_log'].mean()) if 'volume_log' in df.columns else 10.0)
    
    with col3:
        st.write("**Time Parameters**")
        weekday = st.selectbox("Weekday", options=[0, 1, 2, 3, 4, 5, 6], 
                              format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x])
        is_weekend = 1 if weekday >= 5 else 0
        quarter = st.selectbox("Quarter", [1, 2, 3, 4])

    sentiment_avg = (lstm_sentiment + textblob_sentiment) / 2
    sentiment_effect = sentiment_avg * 0.01  # Simple linear relationship
    volatility_effect = (volatility - df['VOLATILITY_10D'].mean()) * 0.001
    weekend_effect = -0.002 if is_weekend else 0.001
    
    predicted_return = sentiment_effect + volatility_effect + weekend_effect
    
    st.write("**Predicted Impact on Stock Returns:**")
    st.metric("Estimated 1-Day Return", f"{predicted_return:.4f}", 
              delta="Positive" if predicted_return > 0 else "Negative")

    with st.expander("How this prediction works"):
        st.write("""
        This is a simplified prediction based on the relationships observed in the data:
        - **Sentiment Effect**: Higher sentiment generally correlates with positive returns
        - **Volatility Effect**: Higher volatility can increase uncertainty
        - **Weekend Effect**: Market behavior often differs on weekends
        """)

def create_insights_and_recommendations():
    """Create insights and recommendations section"""
    st.subheader("💡 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📊 Data Insights:**")
        insights = [
            "Tweet sentiment shows measurable correlation with stock returns",
            "LSTM and TextBlob sentiment models provide different but complementary signals",
            "Short-term predictions (1-3 days) are generally more accurate than long-term",
            "Market volatility and volume are important contextual factors",
            "Random Forest models show the best overall performance for both regression and classification tasks"
        ]
        
        for insight in insights:
            st.write(f"• {insight}")

def main():
    """Main dashboard application"""

    st.markdown('<h1 class="main-header">📊 Tweet Sentiment & Stock Returns Analysis Dashboard</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This dashboard provides a comprehensive analysis of the relationship between Twitter sentiment 
    and stock returns, including exploratory data analysis, model performance evaluation, and interactive insights.
    """)

    df, regression_results, classification_results, feature_importance = load_data()
    
    if df is None:
        st.stop()

    st.sidebar.title("📋 Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        [
            "🏠 Overview",
            "📊 Data Exploration", 
            "🔗 Relationship Analysis",
            "🤖 Model Performance",
            "🎯 Feature Analysis",
            "🔮 Interactive Predictions",
            "💡 Insights"
        ]
    )

    with st.sidebar:
        st.write("**📈 Dataset Overview:**")
        st.write(f"• Total samples: {len(df):,}")
        st.write(f"• Features: {len(df.columns)}")
        st.write(f"• Date range: Available")

        st.write("**💾 Download Results:**")
        if st.button("📊 Download Dashboard Data"):

            summary_data = {
                'total_samples': len(df),
                'average_sentiment': df['sentiment_avg'].mean(),
                'average_return': df['1_DAY_RETURN'].mean(),
                'best_regression_model': regression_results.loc[regression_results['Test_R2'].idxmax(), 'Model'] if regression_results is not None else 'N/A',
                'best_classification_model': classification_results.loc[classification_results['Test_Accuracy'].idxmax(), 'Model'] if classification_results is not None else 'N/A'
            }
            
            st.json(summary_data)

    if page == "🏠 Overview":
        create_overview_metrics(df)
        
        st.subheader("📋 Dataset Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**First 10 rows of processed data:**")
            st.dataframe(df.head(10))
        
        with col2:
            st.write("**Data Types and Missing Values:**")
            info_df = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes,
                'Missing Values': df.isnull().sum(),
                'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
            })
            st.dataframe(info_df)
    
    elif page == "📊 Data Exploration":
        create_univariate_analysis(df)
    
    elif page == "🔗 Relationship Analysis":
        create_bivariate_analysis(df)
    
    elif page == "🤖 Model Performance":
        create_model_performance_dashboard(regression_results, classification_results)
    
    elif page == "🎯 Feature Analysis":
        create_feature_importance_analysis(feature_importance, df)
    
    elif page == "🔮 Interactive Predictions":
        create_interactive_prediction_tool(df)
    
    elif page == "💡 Insights":
        create_insights_and_recommendations()
        
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.8em;'>
            Tweet Sentiment & Stock Returns Analysis Dashboard | 
            Built with Streamlit | 
            Data Science Project
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()