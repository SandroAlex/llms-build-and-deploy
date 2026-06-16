# Makefile for managing the llmsplay Docker development environment
.PHONY: help
.PHONY: build build-fresh up down lock requirements-dev
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

# Derive compose file and uv extras from USE_GPU (set in .env or passed on the command line)
USE_GPU ?= false
ifeq ($(USE_GPU),true)
  COMPOSE_FILE = docker-compose.dev.gpu.yaml
  UV_EXTRA     = gpu
  REQ_FILE     = requirements.dev.gpu.txt
else
  COMPOSE_FILE = docker-compose.dev.cpu.yaml
  UV_EXTRA     = cpu
  REQ_FILE     = requirements.dev.cpu.txt
endif

# Shorthand — avoids repeating --file / --project-name on every command
DC = docker compose --file $(COMPOSE_FILE) --project-name $(PROJECT_NAME)

# Default model for `make ollama-pull` — override with: make ollama-pull MODEL=mistral
MODEL ?= llama3.2

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------
help:
	@printf "\n"
	@printf "${BLUE}Available commands:${NC}\n"
	@printf "${YELLOW}make build${NC}            ${RED}- Build image using Docker layer cache (fast on repeat builds)${NC}\n"
	@printf "${YELLOW}make build-fresh${NC}      ${RED}- Force a full rebuild from scratch (no cache)${NC}\n"
	@printf "${YELLOW}make up${NC}               ${RED}- Start all services in the background${NC}\n"
	@printf "${YELLOW}make down${NC}             ${RED}- Stop all services without removing volumes${NC}\n"
	@printf "${YELLOW}make lock${NC}             ${RED}- Generate uv.lock for reproducible builds${NC}\n"
	@printf "${YELLOW}make requirements-dev${NC} ${RED}- Export pinned requirements (cpu or gpu, driven by USE_GPU)${NC}\n"
	@printf "${YELLOW}make ollama-pull${NC}      ${RED}- Pull an Ollama model (default: llama3.2 — override with MODEL=<name>)${NC}\n"
	@printf "${YELLOW}make ollama-shell${NC}     ${RED}- Open an interactive shell inside the Ollama container${NC}\n"
	@printf "${YELLOW}make compose${NC}          ${RED}- Show which compose file is being used${NC}\n"
	@printf "${YELLOW}make config${NC}           ${RED}- Show the resolved docker-compose configuration${NC}\n"
	@printf "${YELLOW}make debug-worker${NC}     ${RED}- Open a bash shell inside the worker container${NC}\n"
	@printf "${YELLOW}make jupyter${NC}          ${RED}- Start JupyterLab inside the worker container${NC}\n"
	@printf "${YELLOW}make logs${NC}             ${RED}- Follow logs for all services${NC}\n"
	@printf "\n"

# -----------------------------------------------------------------------------
# Build
# -----------------------------------------------------------------------------

build:
	@printf "\n"
	@printf "${GREEN}Building image (with cache) ...${NC}\n"
	@$(DC) build
	@printf "${BLUE}Image built successfully!${NC}\n"
	@printf "\n"

build-fresh:
	@printf "\n"
	@printf "${GREEN}Building image from scratch (no cache) ...${NC}\n"
	@$(DC) build --no-cache
	@printf "${BLUE}Image built successfully!${NC}\n"
	@printf "\n"

# -----------------------------------------------------------------------------
# Services
# -----------------------------------------------------------------------------

up:
	@printf "\n"
	@printf "${GREEN}Starting all services ...${NC}\n"
	@$(DC) up --detach
	@printf "${BLUE}All services are up and running!${NC}\n"
	@printf "\n"

down:
	@printf "\n"
	@printf "${GREEN}Stopping all services ...${NC}\n"
	@$(DC) down
	@printf "${BLUE}All services have been stopped!${NC}\n"
	@printf "\n"

# -----------------------------------------------------------------------------
# Lock file
# -----------------------------------------------------------------------------

lock:
	@printf "\n"
	@printf "${GREEN}Generating uv.lock inside worker container ...${NC}\n"
	@$(DC) run --rm --no-deps $(WORKER_SERVICE) uv lock
	@printf "${BLUE}uv.lock generated — commit it for reproducible builds!${NC}\n"
	@printf "\n"

# -----------------------------------------------------------------------------
# Requirements export
# -----------------------------------------------------------------------------

requirements-dev:
	@printf "\n"
	@printf "${GREEN}Exporting $(UV_EXTRA) requirements ...${NC}\n"
	@$(DC) run --rm --no-deps $(WORKER_SERVICE) uv export --no-hashes --extra $(UV_EXTRA) --output-file $(REQ_FILE) 
	@printf "${BLUE}$(REQ_FILE) written — commit it for reproducible installs!${NC}\n"
	@printf "\n"

# -----------------------------------------------------------------------------
# Ollama
# -----------------------------------------------------------------------------

ollama-pull:
	@printf "\n"
	@printf "${GREEN}Pulling Ollama model: ${YELLOW}$(MODEL)${NC}\n"
	@$(DC) exec ollama-service ollama pull $(MODEL)
	@printf "${BLUE}Model '$(MODEL)' is ready!${NC}\n"
	@printf "\n"

ollama-shell:
	@printf "\n"
	@printf "${GREEN}Opening shell inside Ollama container ...${NC}\n"
	@$(DC) exec ollama-service /bin/bash
	@printf "${BLUE}Exited Ollama shell.${NC}\n"
	@printf "\n"

# -----------------------------------------------------------------------------
# Debug and development
# -----------------------------------------------------------------------------

compose:
	@printf "\n"
	@printf "${GREEN}Using compose file: ${YELLOW}$(COMPOSE_FILE)${NC}\n"
	@printf "\n"

config:
	@printf "\n"
	@$(MAKE) compose
	@printf "${GREEN}Resolved docker-compose configuration:${NC}\n"
	@$(DC) config
	@printf "${BLUE}Compose file is valid!${NC}\n"
	@printf "\n"

debug-worker:
	@printf "\n"
	@printf "${GREEN}Opening bash shell inside worker container ...${NC}\n"
	@$(DC) exec $(WORKER_SERVICE) /bin/bash
	@printf "${BLUE}Exited bash shell.${NC}\n"
	@printf "\n"

jupyter:
	@printf "\n"
	@printf "${GREEN}Starting JupyterLab inside worker container ...${NC}\n"
	@$(DC) exec $(WORKER_SERVICE) jupyter lab --ip 0.0.0.0 --no-browser || true
	@printf "${BLUE}JupyterLab session ended.${NC}\n"
	@printf "\n"

logs:
	@printf "\n"
	@printf "${GREEN}Following logs for all services ...${NC}\n"
	@$(DC) logs --follow --timestamps
	@printf "${BLUE}Exited log view.${NC}\n"
	@printf "\n"
