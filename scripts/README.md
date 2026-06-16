# Useful Scripts

## `cpu_gpu_speed_test.py`
Benchmarks CPU vs GPU performance using PyTorch by running 1000 iterations of a matrix addition on a 4000×4000 tensor and printing the elapsed time for each.

## `ollama_tutorial.py`
Demonstrates MLflow tracing for a local Ollama model. Uses the OpenAI client pointed at a local Ollama endpoint (`localhost:11434`) and enables MLflow auto-tracing to log the interaction to a SQLite backend.

## `openai_mlflow_tracking.py`
Demonstrates MLflow tracing for the OpenAI API (`gpt-4o-mini`). Enables auto-tracing to capture token usage, latency, and payloads, then prints the response and token counts directly.