# Initial imports
#########################################################################################
import os
import mlflow

from typing import Any, List, Dict, Callable

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools.structured import StructuredTool  
#########################################################################################

# Execution parameters
#########################################################################################
# OpenAI model
MODEL: str = "gpt-4o-mini"
MODEL_PROVIDER: str = "openai"

# User query
QUERY: str = "Please could you tell me what the sum of 100 and 200 is?"

# Mlflow parameters
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
EXPERIMENT_NAME = "Interactive LLM Agents with Tools"
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

# Tools definition
#########################################################################################
@tool
def add(a: int, b: int) -> int:
    """
    Add a and b.
    
    Parameters
    ----------
    a : int
        first integer to be added
    b : int
        second integer to be added

    Returns
    -------
    int
        sum of a and b
    """

    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """
    Subtract b from a.
    
    Parameters
    ----------
    a : int
        first integer to be subtracted
    b : int
        second integer to be subtracted

    Returns
    -------
    int
        difference between a and b
    """

    return a - b

@tool
def multiply(a: int, b: int) -> int:
    """
    Multiply a and b.
    
    Parameters
    ----------
    a : int
        first integer to be multiplied
    b : int
        second integer to be multiplied

    Returns
    -------
    int
        product of a and b
    """

    return a * b

# Tools list
tools: List[StructuredTool] = [add, subtract, multiply]
#########################################################################################

# Main code
#########################################################################################
# Initialize the language model from OpenAI
llm = init_chat_model(model=MODEL, model_provider=MODEL_PROVIDER)

# Let's connect and bind the function to the chat model
llm_with_tools = llm.bind_tools(tools)

# Test the tools
tool_map: Dict[str, Callable] = {
    "add": add,
    "subtract": subtract,
    "multiply": multiply
}

# Default inputs
input_values: Dict[str, int] = {
    "a": 10,
    "b": 20,
}

expected_outputs: Dict[str, int] = {
    "add": 30,
    "subtract": -10,
    "multiply": 200,
}

# Test the tools    
print("\n>>> Testing tools ...")
for tool_name in tool_map.keys():

    expected_output: int = expected_outputs[tool_name]
    tool: Callable = tool_map[tool_name]
    result: int = tool.invoke(input_values)
    
    assert result == expected_output
    print(f"\t - {tool_name} tested successfully with result {result} and expected output {expected_output}")

print(">>> All tools tested successfully!")

# Contain the entire conversation between user and LLM
chat_history: List[HumanMessage | AIMessage | ToolMessage] = [HumanMessage(content=QUERY)]

# Run the model with the context (chat history) that contains the user query
response_1: AIMessage = llm_with_tools.invoke(chat_history)
print(f"\n>>> Response 1 (type is {type(response_1)}): {response_1}")

# Extract the tool calls from the response
tool_calls_1 = response_1.tool_calls
tool_1_name: str = tool_calls_1[0]["name"]
tool_1_args: Dict[str, int] = tool_calls_1[0]["args"]
tool_call_1_id: str = tool_calls_1[0]["id"]

# Given the tool call details from the LLM, invoke the correct tool with the correct arguments.
tool_response: int = tool_map[tool_1_name].invoke(tool_1_args)
tool_message: ToolMessage = ToolMessage(content=tool_response, tool_call_id=tool_call_1_id)

print(f">>> Tool response (type is {type(tool_response)}): {tool_response}")

# Append the response to the chat history
chat_history.append(response_1)
chat_history.append(tool_message)
#########################################################################################