"""
Math agent very simple test using langchain tools and mlflow. It will calculate the power 
of two numbers and record all reasoning steps in mlflow.
"""

# Initial imports
#########################################################################################
import re
import os
import mlflow

from typing import List, Dict, Union, Any

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.tools.structured import StructuredTool
from langchain.agents import create_agent
#########################################################################################

# Execution parameters
#########################################################################################
# Main parameters
MODEL: str = "gpt-5-nano"
TEMPERATURE: float = 0.0
DEBUG: bool = True
NAME: str = "math-agent-ted"
SYSTEM_PROMPT: str = "You are a helpful assistant that can do math calculations."
QUERY: str = (
    "Please calculate for me the power of 10 and 3. Give me the result in a json format."
)

# Mlflow parameters
TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", "sqlite:////llm_app/mlflow/database.db"
)
EXPERIMENT_NAME = "Math Agent Test"
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
def calculate_power(input: str) -> Dict[str, Union[float, str]]:
    """
    Calculates the power of a number (x ** y). Input should be two numbers: base and
    exponent.

    Paramters
    ----------
    input : str
        The input string containing the number and the power.

    Returns
    -------
    Dict[float]
        A dictionary with a single key "result" containing the power of the number.

    Examples
    --------
    >>> calculate_power("2^3")
    {"result": 8.0}
    >>> calculate_power("2, 3")
    {"result": 8.0}
    >>> calculate_power("Calculate the power of 2 and 3")
    {"result": 8.0}
    >>> calculate_power("2, 3, 4")
    {"result": "Invalid input. Provide just two numbers to calculate the power."}
    >>> calculate_power("Calculate the power of 2")
    {"result": "Invalid input. Provide just two numbers to calculate the power."}
    """

    try:
        # Extract all numbers from the input as a list of floats
        numbers: List[float] = [float(num) for num in re.findall(r"\d+", input)]

        # If the list of numbers has not 2 elements, return an error
        if len(numbers) != 2:
            return {
                "result": "Invalid input. Provide just two numbers to calculate the power."
            }

        # Calculate the power of the two numbers
        base: float = numbers[0]
        exponent: float = numbers[1]
        result: float = base**exponent

        return {"result": result}

    # If the input is not a valid number, return an error
    except Exception as e:
        return {"result": f"Error: {e}"}


# All available tools
tools: List[StructuredTool] = [calculate_power]

#########################################################################################

# LLM initialization
#########################################################################################
llm_model = ChatOpenAI(model=MODEL, temperature=TEMPERATURE)
#########################################################################################

# Agent initialization
#########################################################################################
agent = create_agent(
    model=llm_model, tools=tools, debug=DEBUG, name=NAME, system_prompt=SYSTEM_PROMPT
)
#########################################################################################

# Main code
#########################################################################################
with mlflow.start_run(run_name="power-calculation"):
    # Record the parameters
    mlflow.log_params(
        {
            "model": MODEL,
            "temperature": TEMPERATURE,
            "agent_name": NAME,
            "system_prompt": SYSTEM_PROMPT,
            "query": QUERY,
        }
    )

    # Invoke the agent
    result: Dict[str, Any] = agent.invoke(
        {"messages": [{"role": "user", "content": QUERY}]},
    )

    print(result)
#########################################################################################
