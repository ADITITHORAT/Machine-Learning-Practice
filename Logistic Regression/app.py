#--------------------------------------------------------------------------------
# Titanic Survival Predictor - Streamlit Web Application
#--------------------------------------------------------------------------------
# This app predicts whether a passenger would survive the Titanic
# disaster based on personal and travel details.
# Model used: Logistic Regression 
#--------------------------------------------------------------------------------


import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model
pipeline = joblib.load("logistic_regression_model.pkl")

# Custom Page Config
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="wide")

# Custom Styles for Better Visibility
st.markdown(
    """
    <style>
    /* Fix the selection box background & text visibility */
    div[data-baseweb="select"] {
        background-color: #ffffff !important;  /* White background */
        color: #000000 !important;  /* Black text */
        border-radius: 8px;
        padding: 5px;
    }
    
    /* Improve visibility of dropdown text */
    div[data-baseweb="select"] div {
        color: #000000 !important;  /* Black text */
        font-weight: bold;
    }

    /* Adjust sliders for better visibility */
    .stSlider {
        background-color: #f8f9fa !important;
        border-radius: 10px;
    }
    
    /* Improve button styling */
    .stButton>button {
        background-color: #0056b3;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App Title
st.title("🚢 Titanic Survival Prediction")
st.markdown('<p class="big-font">Would you have survived the Titanic disaster? Enter your details below to find out!</p>', unsafe_allow_html=True)

# Create Columns for Better Layout
col1, col2 = st.columns([1, 2])

# Sidebar for User Inputs
with col1:
    st.markdown("### Passenger Details 📝")
    
    # Define input fields with better styling
    pclass = st.radio("Passenger Class", [1, 2, 3], format_func=lambda x: f"Class {x}")
    sex = st.radio("Sex", ["Male", "Female"])
    age = st.slider("Age", min_value=0, max_value=100, value=30)
    sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
    fare = st.slider("Fare Paid", min_value=0.0, max_value=500.0, value=30.0, step=5.0)
    embarked = st.radio("Port of Embarkation", ["C", "Q", "S"], format_func=lambda x: {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}[x])

# Convert inputs into a DataFrame for model prediction
input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex.lower()],  # Convert to lowercase for consistency
    'Age': [age],
    'SibSp': [sibsp],
    'Fare': [fare],
    'Embarked': [embarked]
})

# Prediction Display Area
with col2:
    st.markdown("### Prediction Outcome 🎯")
    
    if st.button("🔍 Predict"):
        prediction = pipeline.predict(input_data)
        prob = pipeline.predict_proba(input_data)[:, 1]  # Probability of survival

        # Show prediction results with enhanced visibility
        if prediction[0] == 1:
            st.markdown('<p class="success">✅ The passenger **would survive**.</p>', unsafe_allow_html=True)
            st.progress(prob[0])
        else:
            st.markdown('<p class="failure">❌ The passenger **would not survive**.</p>', unsafe_allow_html=True)
            st.progress(1 - prob[0])

        # Show probability
        st.markdown(f'<p class="big-font">Confidence Level: <span style="color: #0056b3;">{prob[0]*100:.2f}%</span></p>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown('<p class="big-font">🚀 Built with <b>Streamlit</b> | Model: <b>Logistic Regression</b></p>', unsafe_allow_html=True)
