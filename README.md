# Learning Large Language Models

A personal hands-on playground for understanding LLMs and Transformer architectures from the ground up. Not a production project — the goal is deep, practical understanding at every level.

## What this covers

- **Building from scratch** — Transformer models, attention mechanisms, tokenization, training loops, fine-tuning
- **Using LLMs via API** — interacting with OpenAI models, tracking token usage, observing model behaviour
- **Building simple AI agents** — agentic experiments using OpenAI models
- **Running models locally** — serving open-weight models with Ollama for local inference

All experiments are done in **Jupyter notebooks**. The stack is Python 3.11, PyTorch (CPU or CUDA), JupyterLab, MLflow (experiment tracking), and Ollama (local model serving), all running inside Docker.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) ≥ 24
- [Docker Compose](https://docs.docker.com/compose/) plugin (V2, the `docker compose` subcommand)
- GNU Make
- An OpenAI API key (for the API and agent notebooks)

**GPU (optional):** NVIDIA GPU, driver, and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on the host so Docker can pass GPUs into the worker container.

## Getting started

### 1. Set up environment variables

Copy the example env file for **CPU** or **GPU** and fill in your OpenAI key:

```shell
cp .env.cpu.example .env          # CPU (default): USE_GPU=false
# or
cp .env.gpu.example .env          # GPU: USE_GPU=true
# then edit .env and set OPENAI_API_KEY=<your-key>
```

The Makefile picks `docker-compose.cpu.yaml` or `docker-compose.gpu.yaml` from `USE_GPU` in `.env`. You can also override for a single command, e.g. `make build USE_GPU=true`.

### 2. Build the Docker image

```shell
make build
```

This builds a single development image (`llmsplay-dev-cpu` or `llmsplay-dev-gpu`, depending on `USE_GPU`) with Python 3.11, PyTorch, JupyterLab, and MLflow. Docker layer caching makes subsequent builds fast — only changed layers are rebuilt.

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

## GPU support

A GPU stack is available for CUDA-accelerated PyTorch inside the same notebooks and workflow as the CPU setup.

1. Install the NVIDIA Container Toolkit and confirm the host sees your GPU: `nvidia-smi`
2. Use `.env` from `.env.gpu.example` (`USE_GPU=true`) or set `USE_GPU=true` when running Make targets
3. `make build` and `make up` as usual — `worker-service` is started with **all** host NVIDIA GPUs (`gpus: all`)

The GPU worker image is built from `Dockerfile.gpu` (CUDA 13 + cuDNN). MLflow and Ollama still run as in the CPU stack; only the worker container uses the GPU for training and inference in PyTorch.

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
├── docker-compose.cpu.yaml   # CPU compose configuration
├── docker-compose.gpu.yaml   # GPU compose configuration (all NVIDIA GPUs on worker)
├── Dockerfile.cpu            # CPU worker + MLflow image
├── Dockerfile.gpu            # CUDA worker + MLflow image
├── pyproject.toml            # Python dependencies (managed by uv)
├── uv.lock                   # Locked dependency versions
├── .env.cpu.example          # Env template (USE_GPU=false)
├── .env.gpu.example          # Env template (USE_GPU=true)
├── Makefile                  # All developer commands
├── mlflow/                   # MLflow SQLite database (git-ignored)
└── llmsplay/                 # Notebooks and experiment code
```

## Dependency management

Dependencies are managed with [uv](https://github.com/astral-sh/uv). To regenerate the lock file after editing `pyproject.toml`:

```shell
make lock
```

Commit `uv.lock` so that every build uses the exact same package versions.

To export pinned requirements for the active CPU/GPU variant:

```shell
make requirements-dev
```

This writes `requirements.cpu.txt` or `requirements.gpu.txt` according to `USE_GPU`.
