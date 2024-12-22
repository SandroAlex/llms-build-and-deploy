#!/bin/bash
# Initialize chroma vector data base.

# If any of the commands in your code fails for any reason, the entire script fails.
set -o errexit

# Fail exit if one of your pipe command fails.
set -o pipefail

# Exits if any of your variables is not set.
set -o nounset

# Initialize the database API.
uvicorn chroma.server:app --host 0.0.0.0 --port ${CHROMA_PORT}