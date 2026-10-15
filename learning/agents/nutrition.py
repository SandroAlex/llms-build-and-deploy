# Initial imports
#########################################################################################
import os
from typing import Dict, List, Any

from langchain_community.tools.tavily_search import TavilySearchResults 
#########################################################################################

# Execution parameters
#########################################################################################
# Whether to testweb search tool
TEST_TAVILY: bool = True
SAMPLE_QUERY: str = "Give me reviews of Nassim Taleb Antifragile book"
########################################################################################

# Initialize LLM
#########################################################################################
#########################################################################################

# Mlflow setup
#########################################################################################
#########################################################################################

# Building Graph
#########################################################################################
#########################################################################################

# Main code
#########################################################################################
if __name__ == "__main__":

    # Test tavily API tool
    if TEST_TAVILY:
        print(f"\nTesting Tavily search ...")
        
        tavily_tool = TavilySearchResults(max_results=5)
        sample_query:str = SAMPLE_QUERY
        search_results: List[Dict[str, Any]] = tavily_tool.invoke(sample_query)

        print(f"\n>>> Query: {SAMPLE_QUERY}")

        for i, result in enumerate(search_results):

            print(f"\n>>> Result {i}:")
            print(f"title {i}: {result['title']}")
            print(f"url {i}: {result['url']}")
            print(f"content {i}: {result['content']}")
            print(f"score {i}: {result['score']}")
#########################################################################################
