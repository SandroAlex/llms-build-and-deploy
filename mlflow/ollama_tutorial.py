"""
Script to demonstrate how to use MLflow to track interactions with a local Ollama model using the OpenAI API. 
This script sets up MLflow to use a SQLite database for tracking, enables auto-tracing for OpenAI interactions, 
and makes a chat completion request to the local Ollama model. The interactions will be automatically logged in 
MLflow, allowing for easy observability and analysis of the model's performance.

Source: https://medium.com/@hitorunajp/bringing-observability-to-local-llms-first-experiments-with-mlflow-tracing-and-ollama-8f2f18cf9968
"""

# Initial imports
from openai import OpenAI

import mlflow

# Define the tracking URI for MLflow to use SQLite as the backend store
TRACKING_URI: str = "sqlite:////llm_app/mlflow/database.db"

# Experiment name for MLflow
EXPERIMENT_NAME: str = "Tracking Ollama with MLflow"

# The local Ollama REST endpoint
BASE_URL: str = "http://localhost:11434/v1"


# Mlflow setup
mlflow.set_tracking_uri(TRACKING_URI)
experiment = mlflow.set_experiment(EXPERIMENT_NAME)


# Enable auto-tracing for OpenAI
mlflow.openai.autolog()

# Initialize the OpenAI client to connect to the local Ollama REST endpoint
client = OpenAI(
    base_url=BASE_URL,
    api_key="dummy",  # Required to instantiate OpenAI client, it can be a random string
)

# Make a chat completion request to the local Ollama model, which will be automatically traced by MLflow
response = client.chat.completions.create(
    model="llama3.2",
    messages=[
        {
            "role": "system",
            "content": "You are a specialist in personal investing and the stock market.",
        },
        {
            "role": "user",
            "content": "Please briefly explain to a layman what was subprime mortgage crisis in 2008. What were the main causes and consequences of this crisis? What changes were implemented in the financial industry after the crisis to prevent a similar event from happening again?",
        },
    ],
)
