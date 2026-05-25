# Makefile for managing Docker-based micro services for LLMs project
.PHONY: help # Help command
.PHONY: build up down # Boot and halt commands
.PHONY: compose config debug-worker jupyter logs # Debug and development commands

# Include environment variables from .env file if it exists
-include .env
export

# Suppress make's default output
MAKEFLAGS += --no-print-directory

# Colorful output in terminal
GREEN=\033[0;32m
BLUE=\033[0;34m
RED=\033[0;31m
YELLOW=\033[0;33m
NC=\033[0;0m # No Color

# Help command 
help: # Show this help message
	@printf "\n" ;
	@printf "${BLUE}Available commands:${NC}\n" ;
	@printf "${YELLOW}make build${NC}        ${RED}- Build local micro services${NC}\n" ;
	@printf "${YELLOW}make up${NC}           ${RED}- Turn on all local micro services${NC}\n" ;
	@printf "${YELLOW}make down${NC}         ${RED}- Stop all local micro services${NC}\n" ;
	@printf "${YELLOW}make compose${NC}      ${RED}- Show which compose file is being used${NC}\n" ;
	@printf "${YELLOW}make config${NC}       ${RED}- Show a panoramic view of containers${NC}\n" ;
	@printf "${YELLOW}make debug-worker${NC} ${RED}- Debug running worker container${NC}\n" ;
	@printf "${YELLOW}make jupyter${NC}      ${RED}- Run jupyterlab on worker service${NC}\n" ;
	@printf "${YELLOW}make logs${NC}         ${RED}- Show logs for all micro services${NC}\n" ;
	@printf "\n" ;

# Boot and halt commands 
build: # Build local micro services
	@printf "\n" ;
	@printf "${GREEN}Building all services ...${NC}\n" ;
	@docker-compose --file ${COMPOSE_FILE} --project-name ${PROJECT_NAME} build --no-cache --build-arg NO_PROXY=${NO_PROXY} --build-arg HTTP_PROXY=${HTTP_PROXY} --build-arg HTTPS_PROXY=${HTTPS_PROXY} worker-service ;
	@printf "${BLUE}Docker images built successfully!${NC}\n" ;
	@printf "\n" ; 

up: # Turn on all local micro services
	@printf "\n" ;
	@printf "${GREEN}Starting all services ...${NC}\n" ;
	@docker-compose --file ${COMPOSE_FILE} --project-name ${PROJECT_NAME} up --detach ;
	@printf "${BLUE}All services are up and running!${NC}\n" ;
	@printf "\n" ;

down: # Stop all local micro services
	@printf "\n" ;
	@printf "${GREEN}Stopping all services ...${NC}\n" ;
	@docker-compose --file ${COMPOSE_FILE} --project-name ${PROJECT_NAME} down --volumes ;
	@printf "${BLUE}All services have been stopped!${NC}\n" ;
	@printf "\n" ;	

# Debug and development commands 
compose: # Show which compose file is being used
	@printf "\n" ;
	@printf "${GREEN}Using compose file: ${YELLOW}${COMPOSE_FILE}${NC}\n" ;
	@printf "\n" ;

config: # Show a panoramic view of containers
	@printf "\n" ;
	@$(MAKE) compose ;
	@printf "${GREEN}Docker compose configuration:${NC}\n" ;
	@docker-compose --file ${COMPOSE_FILE} --project-name ${PROJECT_NAME} config ;
	@printf "${BLUE}Compose file is OK!${NC}\n" ;
	@printf "\n" ;

debug-worker: # Debug running worker container
	@printf "\n" ;
	@printf "${GREEN}Debugging worker service ...${NC}\n" ;
	@docker-compose --file ${COMPOSE_FILE} --project-name ${PROJECT_NAME} exec ${WORKER_SERVICE} sh -c "/bin/bash" ;
	@printf "${BLUE}Exited bash terminal inside worker container!${NC}\n" ;
	@printf "\n" ;

jupyter: # Run jupyterlab on worker service
	@printf "\n" ;
	@printf "${GREEN}Starting JupyterLab on worker service ...${NC}\n" ;
	@docker-compose --file ${COMPOSE_FILE} --project-name ${PROJECT_NAME} exec ${WORKER_SERVICE} sh -c "jupyter-lab --allow-root --ip 0.0.0.0" || true ;
	@printf "${BLUE}Exited JupyterLab on worker container!${NC}\n" ;
	@printf "\n" ;

logs: # Show logs for all micro services
	@printf "\n" ;
	@printf "${GREEN}Showing logs for all services ...${NC}\n" ;
	@docker-compose --file ${COMPOSE_FILE} --project-name ${PROJECT_NAME} logs --follow --timestamps ;
	@printf "${BLUE}Exited logs view!${NC}\n" ;
	@printf "\n" ;
