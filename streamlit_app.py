import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Function to load the trained model
def load_model(filename):
    """
    Loads a pickled machine learning model from the specified file.

    Args:
        filename (str): The path to the pickle file.

    Returns:
        object: The loaded machine learning model.
    """
    try:
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"Error: The file '{filename}' was not found. Please ensure the model pickle file is in the same directory.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

def predict_with_model(model, user_input_processed):
    """
    Makes a prediction using the loaded model and preprocessed user input.

    Args:
        model (object): The trained machine learning model.
        user_input_processed (pd.DataFrame): The preprocessed user input DataFrame.

    Returns:
        float: The predicted actual productivity.
    """
    prediction = model.predict(user_input_processed)
    return prediction[0]

def preprocess_input(user_input_df):
    """
    Preprocesses the user input DataFrame to match the format expected by the model.
    This includes handling categorical features with OneHotEncoder and numerical
    features with MinMaxScaler, consistent with the original notebook's ColumnTransformer.

    Args:
        user_input_df (pd.DataFrame): The raw user input as a DataFrame.

    Returns:
        pd.DataFrame: The preprocessed DataFrame ready for prediction.
    """
    # Define the columns as used in the original notebook's preprocessor
    num_columns = ['team', 'targeted_productivity', 'smv', 'wip', 'over_time',
                   'incentive', 'idle_time', 'idle_men', 'no_of_style_change',
                   'no_of_workers', 'month']
    cat_columns = ['department']

    # Create the preprocessor pipeline exactly as in the notebook
    numeric_transformer = Pipeline(steps=[
        ('scaler', MinMaxScaler())
    ])

    category_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_columns),
            ('cat', category_transformer, cat_columns)
        ],
        remainder='passthrough' # Keep other columns if any, though none are expected here
    )

    # Fit the preprocessor on a dummy DataFrame with all possible categories
    # This is crucial for 'handle_unknown='ignore'' to work correctly
    # For 'department', we know the unique values are 'sewing' and 'finishing'
    dummy_data = pd.DataFrame({
        'team': [1], 'targeted_productivity': [0.5], 'smv': [10.0], 'wip': [100.0],
        'over_time': [1000], 'incentive': [50], 'idle_time': [0.0], 'idle_men': [0],
        'no_of_style_change': [0], 'no_of_workers': [30.0], 'month': [1],
        'department': ['sewing'] # Include all possible categories for the encoder
    })
    dummy_data = pd.concat([dummy_data, pd.DataFrame([{
        'team': [1], 'targeted_productivity': [0.5], 'smv': [10.0], 'wip': [100.0],
        'over_time': [1000], 'incentive': [50], 'idle_time': [0.0], 'idle_men': [0],
        'no_of_style_change': [0], 'no_of_workers': [30.0], 'month': [1],
        'department': ['finishing']
    }])], ignore_index=True)


    preprocessor.fit(dummy_data)


    # Transform the actual user input
    user_input_processed_array = preprocessor.transform(user_input_df)

    # Get feature names after one-hot encoding
    cat_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(cat_columns)
    all_feature_names = num_columns + list(cat_feature_names)

    user_input_processed_df = pd.DataFrame(user_input_processed_array, columns=all_feature_names)

    return user_input_processed_df

st.set_page_config(layout="centered")

st.write("Zara Abigail Budiman - 2702353221")

def main():
    """
    Main function to run the Streamlit application.
    """
    st.title("Garment Worker Productivity Prediction")
    st.info("Predicting the Actual Productivity of Garment Workers")

    st.subheader("Please Input the Data:")

    # Input widgets based on the notebook's features
    # 'department', 'day', 'team', 'targeted_productivity', 'smv', 'wip',
    # 'over_time', 'incentive', 'idle_time', 'idle_men',
    # 'no_of_style_change', 'no_of_workers', 'month'

    # Note: 'day' is dropped implicitly by not including it in num_columns or cat_columns
    # 'day' was derived from 'date' and then dropped in the notebook before modeling.
    # 'year' and 'new_quarter' were also dropped in the notebook.

    department = st.selectbox("Department", ["sewing", "finishing"])
    team = st.slider("Team", 1, 12, step=1)
    targeted_productivity = st.slider("Targeted Productivity", 0.0, 1.0, value=0.5, step=0.01)
    smv = st.slider("Standard Minute Value (SMV)", 0.0, 100.0, value=10.0, step=0.1)
    wip = st.slider("Work in Progress (WIP)", 0.0, 25000.0, value=0.0, step=100.0)
    over_time = st.slider("Over Time (minutes)", 0, 30000, step=60)
    incentive = st.slider("Incentive (BDT)", 0, 4000, step=10)
    idle_time = st.slider("Idle Time (minutes)", 0.0, 500.0, value=0.0, step=0.1)
    idle_men = st.slider("Idle Men", 0, 50, step=1)
    no_of_style_change = st.slider("Number of Style Changes", 0, 3, step=1)
    no_of_workers = st.slider("Number of Workers", 1.0, 100.0, value=30.0, step=0.5)
    month = st.slider("Month", 1, 3, step=1) # Based on the notebook, only months 1, 2, 3 were present

    user_input = pd.DataFrame([{
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

    # Load the trained model
    model = load_model("trained_model.pkl")

    if model is not None:
        if st.button("Predict Productivity"):
            # Preprocess user input
            processed_input = preprocess_input(user_input)

            # Ensure the order of columns matches the training data of the model
            # This is critical when the model is a pipeline and expects specific feature order
            # The 'get_feature_names_out' from OneHotEncoder and the order of num_columns
            # in ColumnTransformer will define this.
            # We can inspect the model's expected features via model.named_steps['preprocessor'].get_feature_names_out()
            # Or, if the final model itself stores feature_names_in_ or similar after fitting.
            # For simplicity and robustness with Pipelines, we rely on the preprocessor
            # reconstruction within preprocess_input.

            # Make prediction
            prediction = predict_with_model(model, processed_input)

            st.success(f"Predicted Actual Productivity: **{prediction:.4f}**")
            st.write("*(A higher value indicates higher productivity.)*")

if __name__ == "__main__":
    main()
