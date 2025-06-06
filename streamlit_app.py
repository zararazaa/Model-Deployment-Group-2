import streamlit as st
import pandas as pd
import pickle
import numpy as np # Although numpy isn't directly used in the final version, it's a common dependency for data science projects.

# Function to load the trained model
@st.cache_resource e
def load_model(filename):
    """
    Loads a pickled machine learning pipeline from the specified file.

    Args:
        filename (str): The path to the pickle file.

    Returns:
        object: The loaded scikit-learn Pipeline object.
    """
    try:
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        return model



def predict_with_model(model, user_input_df):
    """
    Makes a prediction using the loaded pipeline and raw user input.
    The pipeline will internally handle all necessary preprocessing (scaling, one-hot encoding).

    Args:
        model (object): The trained scikit-learn Pipeline object.
        user_input_df (pd.DataFrame): The raw user input DataFrame from Streamlit widgets.

    Returns:
        float: The predicted actual productivity.
    """
    prediction = model.predict(user_input_df)
    return prediction[0]

st.set_page_config(layout="centered")

st.write("Group 2")

def main():

    st.title("Garment Worker Productivity Prediction")
    st.info("Predicting the Actual Productivity of Garment Workers")

    st.subheader("Please Input the Data:")

    # Input widgets for features expected by your model's pipeline
    # Features from your notebook's X:
    # 'department', 'team', 'targeted_productivity', 'smv', 'wip',
    # 'over_time', 'incentive', 'idle_time', 'idle_men',
    # 'no_of_style_change', 'no_of_workers', 'month'

    department = st.selectbox("Department", ["sewing", "finishing"], help="Department of the team")
    team = st.slider("Team Number", 1, 12, value=1, help="Unique ID of the team")
    targeted_productivity = st.slider("Targeted Productivity", 0.0, 1.0, value=0.6, step=0.01, help="Targeted productivity set by the management for each team for the day.")
    smv = st.slider("Standard Minute Value (SMV)", 0.0, 100.0, value=20.0, step=0.1, help="Standard Minute Value: the allocated time for a task.")
    wip = st.slider("Work in Progress (WIP)", 0.0, 25000.0, value=0.0, step=100.0, help="Number of unfinished items in production. Fill 0 if not applicable.")
    over_time = st.slider("Over Time (minutes)", 0, 30000, value=3000, step=60, help="Amount of overtime by each team in minutes.")
    incentive = st.slider("Incentive (BDT)", 0, 4000, value=30, step=10, help="Financial incentive provided to the team for their productivity.")
    idle_time = st.slider("Idle Time (minutes)", 0.0, 500.0, value=0.0, step=0.1, help="Amount of idle time in minutes.")
    idle_men = st.slider("Idle Men", 0, 50, value=0, step=1, help="Number of idle human resources.")
    no_of_style_change = st.slider("Number of Style Changes", 0, 3, value=0, step=1, help="Number of times the style of the product has changed.")
    no_of_workers = st.slider("Number of Workers", 1.0, 100.0, value=25.0, step=0.5, help="Number of workers in the team.")
    month = st.slider("Month", 1, 3, value=1, step=1, help="Month of the year (1 for Jan, 2 for Feb, 3 for Mar) - based on data available in notebook.")

    # Create a DataFrame from user inputs
    # Ensure column order matches the training data features expected by the pipeline
    # The order of columns in X from your notebook was:
    # 'department', 'day', 'team', 'targeted_productivity', 'smv', 'wip',
    # 'over_time', 'incentive', 'idle_time', 'idle_men', 'no_of_style_change',
    # 'no_of_workers', 'month'
    # Note: 'day' was dropped before modeling in the notebook, so it's not included here.
    user_input_df = pd.DataFrame([{
        "department": department,
        "team": team,
        "targeted_productivity": targeted_productivity,
        "smv": smv,
        "wip": wip,
        "over_time": over_time,
        "incentive": incentive,
        "idle_time": idle_time,
        "idle_men": idle_men,
        "no_of_style_change": no_of_style_change,
        "no_of_workers": no_of_workers,
        "month": month,
    }])

    model_pipeline = load_model("trained_model.pkl")

    if model_pipeline is not None:
        if st.button("Predict Productivity"):
            prediction = predict_with_model(model_pipeline, user_input_df)

            st.success(f"Predicted Actual Productivity: **{prediction:.4f}**")
            st.write("*(A higher value indicates higher productivity.)*")

if __name__ == "__main__":
    main()
