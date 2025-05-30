---
title: Optimal-Developer-Capacity
app_file: app.py
sdk: gradio
sdk_version: 5.32.0
---

# Optimal Developer Capacity

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Optimal--Developer--Capacity-blue?logo=Hugging%20Face)](https://huggingface.co/spaces/pryyyynz/Optimal-Developer-Capacity)

**Live App:** [https://huggingface.co/spaces/pryyyynz/Optimal-Developer-Capacity](https://huggingface.co/spaces/pryyyynz/Optimal-Developer-Capacity)

## Overview
Optimal Developer Capacity is a Gradio-powered web app that predicts your optimal coding hours for tomorrow based on today's productivity metrics. It uses a machine learning model (XGBoost) trained on developer productivity data.

## Features
- Predicts optimal coding hours for the next day
- Considers factors like coding hours, caffeine intake, distractions, sleep, commits, bugs, AI usage, cognitive load, and task success
- Clean, interactive web interface
- Powered by Gradio and Hugging Face Spaces

## How to Use
1. Visit the [Live App](https://huggingface.co/spaces/pryyyynz/Optimal-Developer-Capacity)
2. Fill in your daily stats in the input fields
3. Click to get a personalized recommendation for tomorrow's coding hours
4. Adjust your habits and see how it affects your optimal hours

## Running Locally
1. Clone this repository:
   ```bash
   git clone https://huggingface.co/spaces/pryyyynz/Optimal-Developer-Capacity
   cd Optimal-Developer-Capacity
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   python app.py
   ```

## Files
- `app.py`: Main Gradio app
- `ai_dev_productivity_updated.csv`: Dataset used for preprocessing
- `xgboost_model.pkl`: Trained XGBoost regression model
- `requirements.txt`: Python dependencies
- `optimal_capacity.ipynb`: (Optional) Jupyter notebook for model development

## Requirements
- Python 3.8+
- See `requirements.txt` for all dependencies

## License
This project is for educational and demonstration purposes.

---

*Made with ❤️ using Gradio and Hugging Face Spaces.*