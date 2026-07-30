# Agents

This folder contains hands-on examples for learning how to develop LLM agents — programs that combine a language model with tools to reason about tasks and take actions in a loop.

## Contents

### `math.py`

A minimal math agent that demonstrates the core building blocks of agent development:

- **LangChain tools** — a `calculate_power` tool that parses two numbers from natural-language input and returns `base ** exponent`.
- **Agent setup** — uses `create_agent` with an OpenAI chat model, a system prompt, and the tool list.
- **MLflow tracing** — logs the full agent run (LLM calls, tool invocations, and graph steps) via `mlflow.langchain.autolog()` and `mlflow.openai.autolog()`, with run parameters recorded under the `"Math Agent Test"` experiment.

Run it with `OPENAI_API_KEY` set and the MLflow tracking server available. Traces can be viewed in the MLflow UI at [http://localhost:5000](http://localhost:5000).
