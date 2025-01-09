# Project name.
PROJECT_NAME="llms-project"

# Environment used in building API: development (true) or production (false).
DEVELOPMENT=true

# Environment used in building API: with gpu support (true) or without gpu support (false).
USE_GPUS=false

# Select development or production docker compose file.
ifeq ($(DEVELOPMENT),true)
	ifeq ($(USE_GPUS),true)
		# Development environment with gpu support.
		COMPOSE_FILE="docker-compose-gpu.dev.yaml"
		WORKER_SERVICE="worker-service-gpu"
	else
		# Development environment without gpu support.
		COMPOSE_FILE="docker-compose.dev.yaml"
		WORKER_SERVICE="worker-service"
	endif
else
	# Not implemented yet.
	COMPOSE_FILE="docker-compose.yaml"
endif

# BUILDING.
#####################################################################
# Build local micro services.
docker-build-all:
	docker-compose --file ${COMPOSE_FILE} --project-name ${PROJECT_NAME} up --build --detach ;

# Turn on all local micro services.
docker-up-all:
	docker-compose --file ${COMPOSE_FILE} --project-name ${PROJECT_NAME} up --detach ;

# Stop all local micro services.
docker-down-all:
	docker-compose --file ${COMPOSE_FILE} --project-name ${PROJECT_NAME} down --volumes ;
#####################################################################

# DEBUG AND TESTS.
#####################################################################
# Show which compose file is being used.
which-compose:
	@echo ${COMPOSE_FILE} ;

# Show a panoramic view of containers.
docker-config:
	docker-compose --file ${COMPOSE_FILE} --project-name ${PROJECT_NAME} config ;

# Show logs for all micro services.
docker-logs:
	docker-compose --file ${COMPOSE_FILE} --project-name ${PROJECT_NAME} logs --follow --timestamps ;

# Run jupyterlab on worker service.
docker-exec-jupyterlab:
	docker-compose --file ${COMPOSE_FILE} --project-name ${PROJECT_NAME} exec ${WORKER_SERVICE} sh -c "jupyter-lab --allow-root --ip 0.0.0.0" ;

# Debug running worker container.
docker-debug-worker-service:
	docker-compose --file ${COMPOSE_FILE} --project-name ${PROJECT_NAME} exec ${WORKER_SERVICE} sh -c "/usr/bin/zsh" ;

# Debug running chroma database container.
docker-debug-chroma-service:
	docker-compose --file ${COMPOSE_FILE} --project-name ${PROJECT_NAME} exec chroma-service sh -c "/usr/bin/zsh" ;
#####################################################################