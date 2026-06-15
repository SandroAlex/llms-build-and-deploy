# Learning Large Language Models

A personal hands-on playground for understanding LLMs and Transformer architectures from the ground up. Not a production project — the goal is deep, practical understanding at every level.

## What this covers

- **Building from scratch** — Transformer models, attention mechanisms, tokenization, training loops, fine-tuning
- **Using LLMs via API** — interacting with OpenAI models, tracking token usage, observing model behaviour
- **Building simple AI agents** — agentic experiments using OpenAI models
- **Running models locally** — serving open-weight models with Ollama for local inference

All experiments are done in **Jupyter notebooks**. The stack is Python 3.11, PyTorch, JupyterLab, MLflow (experiment tracking), and Ollama (local model serving), all running inside Docker.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) ≥ 24
- [Docker Compose](https://docs.docker.com/compose/) plugin (V2, the `docker compose` subcommand)
- GNU Make
- An OpenAI API key (for the API and agent notebooks)

## Getting started

### 1. Set up environment variables

Copy the example env file and fill in your OpenAI key:

```shell
cp .env.dev.cpu.example .env
# then edit .env and set OPENAI_API_KEY=<your-key>
```

### 2. Build the Docker image

```shell
make build
```

This builds a single `llmsplay-dev-cpu` image (Python 3.11 + PyTorch CPU + JupyterLab + MLflow). Docker layer caching makes subsequent builds fast — only changed layers are rebuilt.

For a full rebuild from scratch (e.g. to pick up a new base image):

```shell
make build-fresh
```

### 3. Start all services

```shell
make up
```

This starts three containers in the background:

| Service | What it does | Default port |
|---|---|---|
| `worker-service` | Development container; runs notebooks and experiments | — |
| `mlflow-service` | MLflow tracking UI, backed by SQLite | `5000` |
| `ollama-service` | Ollama local model server | `11434` |

### 4. Open JupyterLab

```shell
make jupyter
```

JupyterLab will be available at [http://localhost:8888](http://localhost:8888).

### 5. Pull a local model (optional)

To run experiments against a locally served open-weight model:

```shell
make ollama-pull                    # pulls llama3.2 (default)
make ollama-pull MODEL=mistral      # pull a specific model
make ollama-pull MODEL=qwen2.5:7b   # pull a specific tag
```

Model weights are stored in a named Docker volume (`ollama_models`) and survive container restarts — you only download once.

### 6. Stop services

```shell
make down
```

This stops all containers without removing volumes, so Ollama model weights and MLflow data are preserved.

## All available commands

```shell
make help
```

| Command | Description |
|---|---|
| `make build` | Build image using Docker layer cache (fast on repeat builds) |
| `make build-fresh` | Force a full rebuild from scratch (no cache) |
| `make up` | Start all services in the background |
| `make down` | Stop all services without removing volumes |
| `make lock` | Regenerate `uv.lock` for reproducible builds |
| `make jupyter` | Start JupyterLab inside the worker container |
| `make logs` | Follow logs for all services |
| `make debug-worker` | Open a bash shell inside the worker container |
| `make ollama-pull` | Pull an Ollama model (default: llama3.2) |
| `make ollama-shell` | Open a shell inside the Ollama container |
| `make compose` | Show which compose file is being used |
| `make config` | Show the resolved docker-compose configuration |

## MLflow

The MLflow tracking UI runs at [http://localhost:5000](http://localhost:5000). Experiment data is persisted to `mlflow/database.db` (SQLite) inside the project directory and survives container restarts.

## Project layout

```
.
├── docker-compose.dev.cpu.yaml   # CPU compose configuration
├── Dockerfile.dev.cpu            # Worker + MLflow image definition
├── pyproject.toml                # Python dependencies (managed by uv)
├── uv.lock                       # Locked dependency versions
├── .env.dev.cpu.example          # Template for environment variables
├── Makefile                      # All developer commands
├── mlflow/                       # MLflow SQLite database (git-ignored)
└── llmsplay/                     # Notebooks and experiment code
```

## Dependency management

Dependencies are managed with [uv](https://github.com/astral-sh/uv). To regenerate the lock file after editing `pyproject.toml`:

```shell
make lock
```

Commit `uv.lock` so that every build uses the exact same package versions.

---

> **GPU support** — A GPU-accelerated variant (`Dockerfile.dev.gpu`, `docker-compose.dev.gpu.yaml`) is planned but not yet implemented. Once ready it will allow running larger models locally using CUDA, with no changes to notebooks or experiment code.
