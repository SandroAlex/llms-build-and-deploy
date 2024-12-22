# Project name.
PROJECT_NAME="llms-project"

# Environment used in building API: development (true) or production (false).
DEVELOPMENT=true

# Select development or production docker compose file.
ifeq ($(DEVELOPMENT),true)
	# Development environment.
	COMPOSE_FILE="docker-compose.dev.yaml"
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
# Show a panoramic view of containers.
docker-config:
	docker-compose --file ${COMPOSE_FILE} --project-name ${PROJECT_NAME} config ;

# Show logs for all micro services.
docker-logs:
	docker-compose --file ${COMPOSE_FILE} --project-name ${PROJECT_NAME} logs --follow --timestamps ;

# Run jupyterlab on worker service.
docker-exec-jupyterlab:
	docker-compose --file ${COMPOSE_FILE} --project-name ${PROJECT_NAME} exec worker-service sh -c "jupyter-lab --allow-root --ip 0.0.0.0" ;

# Debug running worker container.
docker-debug-worker-service:
	docker-compose --file ${COMPOSE_FILE} --project-name ${PROJECT_NAME} exec worker-service sh -c "/usr/bin/zsh" ;

# Debug running chroma database container.
docker-debug-chroma-service:
	docker-compose --file ${COMPOSE_FILE} --project-name ${PROJECT_NAME} exec chroma-service sh -c "/usr/bin/zsh" ;	
#####################################################################
