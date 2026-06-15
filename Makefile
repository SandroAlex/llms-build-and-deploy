# Makefile for managing the llmsplay Docker development environment
.PHONY: help
.PHONY: build build-fresh up down lock
.PHONY: compose config debug-worker jupyter logs
.PHONY: ollama-pull ollama-shell

# Load environment variables from .env (copy from .env.dev.cpu.example first)
-include .env
export

# Suppress make's default "Entering directory" output
MAKEFLAGS += --no-print-directory

# Terminal colours
GREEN=\033[0;32m
BLUE=\033[0;34m
RED=\033[0;31m
YELLOW=\033[0;33m
NC=\033[0;0m

# Default model for `make ollama-pull` — override with: make ollama-pull MODEL=mistral
MODEL ?= llama3.2

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------
help:
	@printf "\n"
	@printf "${BLUE}Available commands:${NC}\n"
	@printf "${YELLOW}make build${NC}        ${RED}- Build image using Docker layer cache (fast on repeat builds)${NC}\n"
	@printf "${YELLOW}make build-fresh${NC}  ${RED}- Force a full rebuild from scratch (no cache)${NC}\n"
	@printf "${YELLOW}make up${NC}           ${RED}- Start all services in the background${NC}\n"
	@printf "${YELLOW}make down${NC}         ${RED}- Stop all services without removing volumes${NC}\n"
	@printf "${YELLOW}make lock${NC}         ${RED}- Generate uv.lock for reproducible builds${NC}\n"
	@printf "${YELLOW}make ollama-pull${NC}  ${RED}- Pull an Ollama model (default: llama3.2 — override with MODEL=<name>)${NC}\n"
	@printf "${YELLOW}make ollama-shell${NC} ${RED}- Open an interactive shell inside the Ollama container${NC}\n"
	@printf "${YELLOW}make compose${NC}      ${RED}- Show which compose file is being used${NC}\n"
	@printf "${YELLOW}make config${NC}       ${RED}- Show the resolved docker-compose configuration${NC}\n"
	@printf "${YELLOW}make debug-worker${NC} ${RED}- Open a bash shell inside the worker container${NC}\n"
	@printf "${YELLOW}make jupyter${NC}      ${RED}- Start JupyterLab inside the worker container${NC}\n"
	@printf "${YELLOW}make logs${NC}         ${RED}- Follow logs for all services${NC}\n"
	@printf "\n"

# -----------------------------------------------------------------------------
# Build
# -----------------------------------------------------------------------------

# Standard build — uses Docker's layer cache.
# Layers that have not changed (apt, UV install, Python deps) are reused,
# making repeat builds significantly faster.  Use this for day-to-day work.
build:
	@printf "\n"
	@printf "${GREEN}Building image (with cache) ...${NC}\n"
	@docker compose --file ${COMPOSE_FILE} --project-name ${PROJECT_NAME} build
	@printf "${BLUE}Image built successfully!${NC}\n"
	@printf "\n"

# Fresh build — discards ALL cached layers and rebuilds from scratch.
# Use this only when you need to pick up updated base images or when the
# cache has become stale in an unexpected way.
build-fresh:
	@printf "\n"
	@printf "${GREEN}Building image from scratch (no cache) ...${NC}\n"
	@docker compose --file ${COMPOSE_FILE} --project-name ${PROJECT_NAME} build --no-cache
	@printf "${BLUE}Image built successfully!${NC}\n"
	@printf "\n"

# -----------------------------------------------------------------------------
# Services
# -----------------------------------------------------------------------------

# Start all services in the background
up:
	@printf "\n"
	@printf "${GREEN}Starting all services ...${NC}\n"
	@docker compose --file ${COMPOSE_FILE} --project-name ${PROJECT_NAME} up --detach
	@printf "${BLUE}All services are up and running!${NC}\n"
	@printf "\n"

# Stop all services without removing volumes
down:
	@printf "\n"
	@printf "${GREEN}Stopping all services ...${NC}\n"
	@docker compose --file ${COMPOSE_FILE} --project-name ${PROJECT_NAME} down
	@printf "${BLUE}All services have been stopped!${NC}\n"
	@printf "\n"

# -----------------------------------------------------------------------------
# Lock file
# -----------------------------------------------------------------------------

# Generate (or refresh) uv.lock from pyproject.toml inside the worker
# container.  The lock file is written back to the host via the volume mount.
# Commit uv.lock after running this — subsequent `make build` runs will use
# `--frozen` automatically once the Dockerfile is updated to COPY the lock file.
lock:
	@printf "\n"
	@printf "${GREEN}Generating uv.lock inside worker container ...${NC}\n"
	@docker compose --file ${COMPOSE_FILE} --project-name ${PROJECT_NAME} \
	    run --rm --no-deps ${WORKER_SERVICE} uv lock
	@printf "${BLUE}uv.lock generated — commit it for reproducible builds!${NC}\n"
	@printf "\n"

# -----------------------------------------------------------------------------
# Ollama
# -----------------------------------------------------------------------------

# Pull a model into the ollama-service container.
# Models are stored in the `ollama_models` named volume and persist across
# container restarts.  Usage examples:
#   make ollama-pull                   → pulls llama3.2 (default)
#   make ollama-pull MODEL=mistral     → pulls mistral
#   make ollama-pull MODEL=qwen2.5:7b  → pulls a specific tag
ollama-pull:
	@printf "\n"
	@printf "${GREEN}Pulling Ollama model: ${YELLOW}${MODEL}${NC}\n"
	@docker compose --file ${COMPOSE_FILE} --project-name ${PROJECT_NAME} \
	    exec ollama-service ollama pull ${MODEL}
	@printf "${BLUE}Model '${MODEL}' is ready!${NC}\n"
	@printf "\n"

# Open an interactive shell inside the Ollama container (useful for running
# `ollama list`, `ollama run <model>` interactively, or troubleshooting).
ollama-shell:
	@printf "\n"
	@printf "${GREEN}Opening shell inside Ollama container ...${NC}\n"
	@docker compose --file ${COMPOSE_FILE} --project-name ${PROJECT_NAME} \
	    exec ollama-service /bin/bash
	@printf "${BLUE}Exited Ollama shell.${NC}\n"
	@printf "\n"

# -----------------------------------------------------------------------------
# Debug and development
# -----------------------------------------------------------------------------

# Show the compose file being used
compose:
	@printf "\n"
	@printf "${GREEN}Using compose file: ${YELLOW}${COMPOSE_FILE}${NC}\n"
	@printf "\n"

# Show the resolved docker-compose configuration
config:
	@printf "\n"
	@$(MAKE) compose
	@printf "${GREEN}Resolved docker-compose configuration:${NC}\n"
	@docker compose --file ${COMPOSE_FILE} --project-name ${PROJECT_NAME} config
	@printf "${BLUE}Compose file is valid!${NC}\n"
	@printf "\n"

debug-worker:
	@printf "\n"
	@printf "${GREEN}Opening bash shell inside worker container ...${NC}\n"
	@docker compose --file ${COMPOSE_FILE} --project-name ${PROJECT_NAME} exec ${WORKER_SERVICE} /bin/bash
	@printf "${BLUE}Exited bash shell.${NC}\n"
	@printf "\n"

jupyter:
	@printf "\n"
	@printf "${GREEN}Starting JupyterLab inside worker container ...${NC}\n"
	@docker compose --file ${COMPOSE_FILE} --project-name ${PROJECT_NAME} exec ${WORKER_SERVICE} \
	    jupyter lab --ip 0.0.0.0 --no-browser || true
	@printf "${BLUE}JupyterLab session ended.${NC}\n"
	@printf "\n"

logs:
	@printf "\n"
	@printf "${GREEN}Following logs for all services ...${NC}\n"
	@docker compose --file ${COMPOSE_FILE} --project-name ${PROJECT_NAME} logs --follow --timestamps
	@printf "${BLUE}Exited log view.${NC}\n"
	@printf "\n"
