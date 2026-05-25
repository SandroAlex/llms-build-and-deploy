#! /bin/bash

# Define colors
BLUE="\033[0;34m"
GREEN="\033[0;32m"
NC="\033[0m"

# Parameters of the request
MODEL="qwen3:8b-32k-context"
MESSAGE="Can you tell me more about the books of Nassim Taleb?"

# Quick test of OpenAI compatible endpoint for Qwen3:8b model
echo ""
echo -e "${BLUE}Testing OpenAI compatible endpoint for Qwen3:8b model ...${NC}\n"
echo -e "${GREEN}Model: ${MODEL}${NC}"
echo -e "${GREEN}Message: ${MESSAGE}${NC}"
echo ""

# Do it
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"${MESSAGE}\"}]
  }" | jq
