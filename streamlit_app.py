"""
IoT Critical Device Prediction - Streamlit App
Multi-page application for CatBoost model deployment
"""
import streamlit as st

# Page config
st.set_page_config(
    page_title="IoT Critical Device Prediction",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define pages
home_page = st.Page("pages/1_Home.py", title="Home", icon="🏠", default=True)
batch_page = st.Page("pages/2_Batch_Upload.py", title="Batch Upload", icon="📤")
single_page = st.Page("pages/3_Single_Predict.py", title="Single Prediction", icon="🔍")
insights_page = st.Page("pages/4_Insights.py", title="Model Insights", icon="📊")

# Navigation
pg = st.navigation([home_page, batch_page, single_page, insights_page])

# Sidebar info
st.sidebar.title("🔧 IoT Device Prediction")
st.sidebar.markdown("---")
st.sidebar.markdown("**Model Version:** v1.0")
st.sidebar.markdown("**Training Date:** 2025-11-07")
st.sidebar.markdown("**Algorithm:** CatBoost + SMOTE")
st.sidebar.markdown("---")

# Key metrics in sidebar
st.sidebar.metric("Recall", "78.6%", "+28.6% vs baseline")
st.sidebar.metric("Precision", "84.6%", "+4.6% vs target")
st.sidebar.metric("F1-Score", "81.5%")

# Run selected page
pg.run()
