"""
Script to demonstrate how to use MLflow to track interactions with OpenAI models using an API key.
Token usage (prompt tokens, completion tokens, total tokens) is captured automatically by MLflow's
OpenAI auto-tracing and also printed directly from the response object.

Usage:
    Set OPENAI_API_KEY and optionally MLFLOW_TRACKING_URI in your environment before running.
"""

# Initial imports
import os

import mlflow
from openai import OpenAI

# Define the tracking URI for MLflow to use SQLite as the backend store
TRACKING_URI: str = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:////llm_app/mlflow/database.db")

# Experiment name for MLflow
EXPERIMENT_NAME: str = "Tracking OpenAI with MLflow"

# The model to use
MODEL: str = "gpt-4o-mini"

# Set up MLflow
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# Enables automatic tracing of all OpenAI calls, including token counts, latency, and payloads
mlflow.openai.autolog()

# Reads OPENAI_API_KEY from the environment automatically
client = OpenAI()  

# Make a chat completion request to the OpenAI model, which will be automatically traced by MLflow
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content": "You are a specialist in personal investing and the stock market.",
        },
        {
            "role": "user",
            "content": (
                "Please briefly explain to a layman what was the subprime mortgage crisis in 2008. "
                "What were the main causes and consequences of this crisis? "
                "What changes were implemented in the financial industry after the crisis to prevent "
                "a similar event from happening again?"
            ),
        },
    ],
)

print(response.choices[0].message.content)

usage = response.usage
print("\n--- Token Usage ---")
print(f"Prompt tokens:     {usage.prompt_tokens}")
print(f"Completion tokens: {usage.completion_tokens}")
print(f"Total tokens:      {usage.total_tokens}")
