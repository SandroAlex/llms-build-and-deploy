# Initial imports
#########################################################################################
import os
import json
import mlflow
import warnings 

from typing import List, Dict, Any, Annotated, Sequence, TypedDict

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools.structured import StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableSequence
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages


warnings.filterwarnings("ignore")
#########################################################################################

# Execution parameters
#########################################################################################
# Maximum number of results to return from the Tavily search API
MAX_RESULTS: int = 2

# Mlflow parameters
TRACKING_URI: str = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-service:5000")
EXPERIMENT_NAME: str = "GISREP"

# Large language model parameters
MODEL: str = "gpt-4o-mini" 
TEMPERATURE: float = 0.0
LOGPROBS: bool = True
#########################################################################################

# Mlflow setup
#########################################################################################
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# Captures LLM calls, tool calls, chain/graph steps as nested spans
mlflow.langchain.autolog()

# Optional: also trace raw OpenAI HTTP calls (token usage, latency)
mlflow.openai.autolog()
#########################################################################################

# Tools and agents setup
#########################################################################################
# Initialize the Tavily search tool
search = TavilySearchResults(max_results=MAX_RESULTS)

@tool
def search_tool(query: str) -> str:
    """
    Use the Tavily search API to find information on the web.

    Parameters
    ----------
    query: str
        The query to search for.

    Returns
    -------
    List[Dict[str, Any]]
        The results of the search.
    """
    
    results: List[Dict[str, Any]] = search.invoke(query)
    
    return results

@tool
def recommend_clothing(weather: str) -> str:
    """
    Returns a clothing recommendation based on the provided weather description.

    This function examines the input string for specific keywords or temperature indicators 
    (e.g., "snow", "freezing", "rain", "85°F") to suggest appropriate attire. It handles 
    common weather conditions like snow, rain, heat, and cold by providing simple and practical 
    clothing advice.

    Parameters
    ----------
    weather: str
        A brief description of the weather (e.g., "Overcast, 64.9°F")

    Returns
    -------
    str
        A string with clothing recommendations suitable for the weather
    """
    
    weather = weather.lower()
    if "snow" in weather or "freezing" in weather:
        return "Wear a heavy coat, gloves, and boots."
    
    elif "rain" in weather or "wet" in weather:
        return "Bring a raincoat and waterproof shoes."
    
    elif "hot" in weather or "85" in weather:
        return "T-shirt, shorts, and sunscreen recommended."
    
    elif "cold" in weather or "50" in weather:
        return "Wear a warm jacket or sweater."
    
    else:
        return "A light jacket should be fine."    

# Initialize the language model
llm: ChatOpenAI = ChatOpenAI(model=MODEL, temperature=TEMPERATURE)

# Creating the tool registry
tools: List[StructuredTool] = [search_tool, recommend_clothing]
tools_by_name: Dict[str, StructuredTool] = {tool.name: tool for tool in tools}

# Defines the agents behavior and personality
chat_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages([
    ("system", """
You are a helpful AI assistant that thinks step-by-step and uses tools when needed.

When responding to queries:
1. First, think about what information you need
2. Use available tools if you need current data or specific capabilities  
3. Provide clear, helpful responses based on your reasoning and any tool results

Always explain your thinking process to help users understand your approach.
"""),
    MessagesPlaceholder(variable_name="scratch_pad")
])

# Bind the tools to the language model
model_react_agent: RunnableSequence = chat_prompt | llm.bind_tools(tools)

# Agent must maintain context across multiple reasoning and acting steps
class AgentState(TypedDict):
    """
    The state of the agent.
    """

    # add_messages is a reducer
    # See https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers
    messages: Annotated[Sequence[BaseMessage], add_messages]
#########################################################################################

# Main code
#########################################################################################

#########################################################################################
