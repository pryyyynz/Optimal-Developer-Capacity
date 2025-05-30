import gradio as gr
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Load the dataset for preprocessing
ai_dev_productivity_df = pd.read_csv("./ai_developer_productivity_data/ai_dev_productivity.csv")

# Preprocess the data
X = ai_dev_productivity_df.drop(columns=['optimal_hours_tomorrow'])
y = ai_dev_productivity_df['optimal_hours_tomorrow']

# One-hot encoding for categorical variables if any
X = pd.get_dummies(X, drop_first=True)

# Standardize the features
scaler = StandardScaler()
scaler.fit(X)

# Load the trained XGBoost model
model = joblib.load("xgboost_model.pkl")

# Define the prediction function
def predict_optimal_hours(hours_coding, coffee_intake_mg, distractions, sleep_hours, commits, bugs_reported, ai_usage_hours, cognitive_load, task_success):
    # Create a feature array
    features = pd.DataFrame({
        "hours_coding": [hours_coding],
        "coffee_intake_mg": [coffee_intake_mg],
        "distractions": [distractions],
        "sleep_hours": [sleep_hours],
        "commits": [commits],
        "bugs_reported": [bugs_reported],
        "ai_usage_hours": [ai_usage_hours],
        "cognitive_load": [cognitive_load],
        "task_success": [task_success]
    })

    # One-hot encoding for categorical variables if any
    features = pd.get_dummies(features, drop_first=True)

    # Align columns with training data
    features = features.reindex(columns=X.columns, fill_value=0)

    # Standardize the features using the scaler
    features_scaled = scaler.transform(features)

    # Predict using the model
    prediction = model.predict(features_scaled)
    return prediction[0]

# Create the Gradio interface
inputs = [
    gr.Number(label="Hours Coding"),
    gr.Number(label="Coffee Intake (mg)"),
    gr.Number(label="Distractions"),
    gr.Number(label="Sleep Hours"),
    gr.Number(label="Commits"),
    gr.Number(label="Bugs Reported"),
    gr.Number(label="AI Usage Hours"),
    gr.Number(label="Cognitive Load"),
    gr.Number(label="Task Success (0 or 1)")
]

outputs = gr.Textbox(label="Optimal Hours Tomorrow")

gr.Interface(
    fn=predict_optimal_hours,
    inputs=inputs,
    outputs=outputs,
    title="Optimal Coding Hours Predictor",
    description="Predict the optimal coding hours for tomorrow based on today's metrics."
).launch()