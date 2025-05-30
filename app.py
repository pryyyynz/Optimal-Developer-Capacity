import gradio as gr
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Load the dataset for preprocessing
ai_dev_productivity_df = pd.read_csv("ai_dev_productivity_updated.csv")

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
    return f"{prediction[0]:.2f}"

# Create the Gradio interface
inputs = [
    gr.Number(label="Coding Hours Today", info="How many hours did you code today?"),
    gr.Number(label="Coffee Intake (mg)", info="Total caffeine consumed today (mg)"),
    gr.Number(label="Number of Distractions", info="How many times were you distracted?"),
    gr.Number(label="Sleep Hours", info="How many hours did you sleep last night?"),
    gr.Number(label="Number of Commits", info="Total code commits today"),
    gr.Number(label="Bugs Encountered", info="How many bugs did you encounter?"),
    gr.Number(label="AI Usage Hours (Claude, Copilot, etc.)", info="Hours spent using AI tools"),
    gr.Slider(minimum=0, maximum=10, step=1, label="Cognitive Load", info="0 = relaxed, 10 = extremely stressful"),
    gr.Radio(choices=[0, 1], label="Task Success", info="1 = Success, 0 = Not Successful")
]

outputs = gr.Textbox(label="Optimal Working Hours Tomorrow", lines=1, interactive=False)

gr.Interface(
    fn=predict_optimal_hours,
    inputs=inputs,
    outputs=outputs,
    title="Optimal Working Hours For Developers",
    description=(
        "Predict your optimal coding hours for tomorrow based on today's metrics.<br>"
        "<ul>"
        "<li>Fill in your daily stats below.</li>"
        "<li>Get a personalized recommendation for tomorrow's coding hours!</li>"
        "</ul>"
        "<b>Tip:</b> Adjust your habits and see how it affects your optimal hours."
    ),
    theme="soft",
    allow_flagging="never"
).launch(share=True)