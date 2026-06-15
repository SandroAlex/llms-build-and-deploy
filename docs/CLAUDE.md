# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a personal learning playground for Large Language Models and Transformer architectures. It is not a production project — the goal is hands-on understanding of how LLMs work at every level:

- **Building from scratch** — implementing Transformer models and LLM architectures from the ground up to understand the internals (attention mechanisms, tokenization, training loops, fine-tuning, etc.)
- **Using LLMs via API** — learning how to interact with hosted models (primarily OpenAI) through API keys, with a focus on understanding token usage, tracking inputs/outputs, and observing model behavior
- **Building simple AI agents** — experimenting with OpenAI models in simple agentic applications to understand how agents work in practice
- **Deploying and running models locally** — using **Ollama** to run open-weight models locally, experimenting with local inference and understanding how model serving works

The primary medium for learning is Jupyter notebooks. The core stack is **Docker / Docker Compose, JupyterLab, Python, PyTorch / PyTorch Lightning, MLflow** for experiment tracking, and **Ollama** for local model serving.
