"""
ReAct (Reasoning + Acting) agent built with LangGraph.

The agent alternates between reasoning with an OpenAI chat model and calling tools
(Tavily web search and a clothing recommender) until it can answer the user's query.
It answers an example question about Zurich's weather and what to wear, streams each
step to the console, saves the graph diagram to `react.png`, and traces runs in MLflow.
"""

# Initial imports
#########################################################################################
import os
import json
import mlflow
import warnings

from typing import List, Dict, Any, Annotated, Iterator, Sequence, TypedDict

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools.structured import StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableSequence
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph

warnings.filterwarnings("ignore")
#########################################################################################

# Execution parameters
#########################################################################################
# Maximum number of results to return from the Tavily search API
MAX_RESULTS: int = 2

# Mlflow parameters
TRACKING_URI: str = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-service:5000")
EXPERIMENT_NAME: str = "REACT WEATHER AGENT"

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

# Tools, models and methods
#########################################################################################
# Initialize the language model
llm: ChatOpenAI = ChatOpenAI(model=MODEL, temperature=TEMPERATURE)

# Defines the agents behavior and personality
chat_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful AI assistant that thinks step-by-step and uses tools when needed.

When responding to queries:
1. First, think about what information you need
2. Use available tools if you need current data or specific capabilities  
3. Provide clear, helpful responses based on your reasoning and any tool results

Always explain your thinking process to help users understand your approach.
""",
        ),
        MessagesPlaceholder(variable_name="scratch_pad"),
    ]
)

# Initialize the Tavily search tool
search: TavilySearchResults = TavilySearchResults(max_results=MAX_RESULTS)


@tool
def search_tool(query: str) -> List[Dict[str, Any]]:
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


# Agent must maintain context across multiple reasoning and acting steps
class AgentState(TypedDict):
    """
    The state of the agent.
    """

    # add_messages is a reducer
    # See https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers
    messages: Annotated[Sequence[BaseMessage], add_messages]


# Function Purpose:
# - Automatically execute all tool calls from the model
# - Handle multiple simultaneous tool calls
# - Return properly formatted tool messages
def tool_node(state: AgentState) -> Dict[str, Any]:
    """
    Execute all tool calls from the last message in the state.
    """

    outputs: List[ToolMessage] = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_result: Any = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=json.dumps(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )

    return {"messages": outputs}


# Model invocation node
def call_model_node(state: AgentState) -> Dict[str, List[AIMessage]]:
    """
    Invoke the model with the current conversation state.
    """

    response: AIMessage = model_react.invoke({"scratch_pad": state["messages"]})

    return {"messages": [response]}


# Decision logic node
def should_continue(state: AgentState) -> str:
    """
    Determine whether to continue with tool use or end the conversation.
    """

    messages: Sequence[BaseMessage] = state["messages"]
    last_message: AIMessage = messages[-1]

    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"

    # Otherwise if there is, we continue
    else:
        return "continue"


def print_stream(stream: Iterator[Dict[str, Any]]) -> None:
    """
    Helper function for formatting the stream nicely.
    """

    for s in stream:

        message: BaseMessage | tuple = s["messages"][-1]

        if isinstance(message, tuple):
            print(message)

        else:
            message.pretty_print()


# Creating the tool registry
tools: List[StructuredTool] = [search_tool, recommend_clothing]
tools_by_name: Dict[str, StructuredTool] = {tool.name: tool for tool in tools}

# Bind the tools to the language model
model_react: RunnableSequence = chat_prompt | llm.bind_tools(tools)
#########################################################################################

# Constructing the state graph:
# 1. Agent Node: Where reasoning happens and tool calls are generated
# 2. Tools Node: Where tool execution occurs
# 3. Conditional Edge: Determines whether to continue or finish
# 4. Entry Point: Conversation always starts with the agent reasoning
#########################################################################################
# Define a new graph
workflow: StateGraph = StateGraph(AgentState)

# Define the two nodes we will cycle between
workflow.add_node("AGENTE", call_model_node)
workflow.add_node("FERRAMENTAS", tool_node)

# Add edges between nodes
workflow.add_edge("FERRAMENTAS", "AGENTE")  # After tools, always go back to agent

# Add conditional logic
workflow.add_conditional_edges(
    "AGENTE",
    should_continue,
    {
        "continue": "FERRAMENTAS",  # If tools needed, go to tools node
        "end": END,  # If done, end the conversation
    },
)

# Set entry point
workflow.set_entry_point("AGENTE")

# Compile the graph
graph: CompiledStateGraph = workflow.compile()

# Visualize the graph
_: bytes = graph.get_graph().draw_mermaid_png(
    output_file_path="./learning/agents/react.png"
)
#########################################################################################

# Main code
#########################################################################################
inputs: Dict[str, List[HumanMessage]] = {
    "messages": [
        HumanMessage(
            content="What's the weather like in Zurich, and what should I wear based on the temperature?"
        )
    ]
}
# What You'll See:
# 1. Initial Reasoning: Agent analyzes the query
# 2. Tool Call 1: Searches for Zurich weather
# 3. Tool Result Processing: Agent examines weather data
# 4. Tool Call 2: Gets clothing recommendations
# 5. Final Synthesis: Agent combines all information into a helpful response
stream: Iterator[Dict[str, Any]] = graph.stream(inputs, stream_mode="values")
print_stream(stream)
#########################################################################################
