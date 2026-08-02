"""
AI-powered agent that can help non-technical users perform data science tasks through
natural language.
"""

# Initial imports
#########################################################################################
import glob
import os
import mlflow

import matplotlib
import seaborn
import sklearn
import langchain
import openai
import langchain_openai

import numpy as np
import pandas as pd

from pathlib import Path
from typing import List, Optional, Dict, Any, Callable, Union

from langchain_classic.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_core.tools.structured import StructuredTool
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

#########################################################################################

# Execution parameters
#########################################################################################
# Folder with .csv files
DATASET_FOLDER: str = str(Path(__file__).parent / "data")

# OpenAI model
MODEL: str = "gpt-4o-mini"

# Define agent scope and purpose
SYSTEM_PROMPT: str = (
    "You are a data science assistant. Use the available tools to analyze CSV files. "
    "Your job is to determine whether each dataset is for classification or regression, "
    "based on its structure."
)

# Mlflow parameters
TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", "sqlite:////llm_app/mlflow/database.db"
)
EXPERIMENT_NAME = "Data Wizard Agent Test"
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


# Langchain tools
#########################################################################################
@tool
def list_csv_files(directory: Optional[str] = None) -> Optional[List[str]]:
    """
    List all CSV file names in the 'directory' parameter if provided, otherwise the
    current working directory.

    Parameters
    ----------
    directory: Optional[str]
        The directory to list CSV files from. If not provided, the current working
        directory is used.

    Returns
    -------
    Optional[List[str]]
        A list containing CSV file names. If no CSV files are found, returns None.
    """

    # Grab the current working directory if no directory is provided
    if directory is None:
        directory: str = os.getcwd()

    # All files in the directory that end with .csv
    csv_files: List[str] = glob.glob(os.path.join(directory, "*.csv"))

    # If no CSV files are found, return None
    if not csv_files:
        return None

    # Return the basename of the CSV files
    return [os.path.basename(file) for file in csv_files]


@tool
def preload_datasets(paths: List[str]) -> str:
    """
    Loads CSV files into a global cache if not already loaded.

    This function helps to efficiently manage datasets by loading them once and storing
    them in memory for future use. Without caching, you would waste tokens describing
    dataset contents repeatedly in agent responses.

    Parameters
    ----------
    paths: List[str]
        A list of file paths to CSV files.

    Returns
    -------
    str
        A message summarizing which datasets were loaded or already cached.
    """

    # Initialize lists to store loaded and cached datasets
    loaded: List[str] = []
    cached: List[str] = []

    # Loop through the paths and load the CSV files into the cache
    for path in paths:
        if path not in DATAFRAME_CACHE.keys():
            DATAFRAME_CACHE[path] = pd.read_csv(path)
            loaded.append(path)
        else:
            cached.append(path)

    return f"Loaded datasets: {loaded}\nAlready cached: {cached}"


@tool
def get_dataset_summaries(dataset_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Analyze multiple CSV files and return metadata summaries for each.

    Parameters
    ----------
    dataset_paths: List[str]
        A list of file paths to CSV datasets.

    Returns
    -------
    List[Dict[str, Any]]
        A list of summaries, one per dataset, each containing:
        - "file_name": The path of the dataset file.
        - "column_names": A list of column names in the dataset.
        - "data_types": A dictionary mapping column names to their data types (as strings).
    """

    # Initialize a list to store the summaries
    summaries: List[Dict[str, Any]] = []

    # Loop through the dataset paths and get the summaries
    for path in dataset_paths:
        # Load and cache the dataset if not already cached
        if path not in DATAFRAME_CACHE.keys():
            DATAFRAME_CACHE[path] = pd.read_csv(path)

        # Into memory
        df: pd.DataFrame = DATAFRAME_CACHE[path]

        # Build summary
        summary: Dict[str, Any] = {
            "file_name": path,
            "column_names": df.columns.tolist(),
            "data_types": df.dtypes.astype(str).to_dict(),
        }

        # Add the summary to the list
        summaries.append(summary)

    return summaries


@tool
def call_dataframe_method(file_name: str, method: str) -> str:
    """
    Execute a method on a DataFrame and return the result.

    This tool lets you run simple DataFrame methods like 'head', 'tail', or 'describe'
    on a dataset that has already been loaded and cached using 'preload_datasets' tool.

    Parameters
    ----------
    file_name: str
        The path or name of the dataset in the global cache.
    method: str
        The name of the method to call on the DataFrame. Only no-argument methods are
        supported (e.g., 'head', 'describe', 'info').

    Returns
    -------
    str
        The output of the method as a formatted string, or an error message if the dataset
        is not found or the method is invalid.

    Examples
    --------
    >>> call_dataframe_method(file_name="./data.csv", method="head")
    col1 col2 col3
    1    2    3
    4    5    6
    7    8    9
    10   11   12
    13   14   15
    """

    # Try to get the DataFrame from cache, or load it if not already cached
    if file_name not in DATAFRAME_CACHE.keys():
        try:
            DATAFRAME_CACHE[file_name] = pd.read_csv(file_name)

        except FileNotFoundError:
            return f"DataFrame '{file_name}' not found in cache or on disk."

        except Exception as e:
            return f"Error loading '{file_name}': {str(e)}"

    # Into memory
    df: pd.DataFrame = DATAFRAME_CACHE[file_name]

    # Get the method from the DataFrame. Return None if the method is not found.
    func: Callable | None = getattr(df, method, None)

    # If the method is not callable, return an error message
    if not callable(func):
        return f"'{method}' is not a valid method of a pandas DataFrame."

    try:
        result: Any = func()
        return str(result)

    except Exception as e:
        return f"Error calling '{method}' on '{file_name}': {str(e)}"


@tool
def evaluate_classification_dataset(
    file_name: str, target_column: str
) -> Dict[str, Union[float, str]]:
    """
    Train and evaluate a classifier on a dataset using the specified target column.

    Parameters
    ----------
    file_name: str
        The name or path of the dataset stored in `DATAFRAME_CACHE`.
    target_column: str
        The name of the column to use as the classification target.

    Returns
    -------
    Dict[str, Union[float, str]]
        A dictionary with the model's accuracy score or an error message if the dataset
        is not found or the target column is not found.

    Examples
    --------
    >>> evaluate_classification_dataset(file_name="./data.csv", target_column="target")
    {"accuracy": 0.95}
    """

    # Try to get the DataFrame from cache, or load it if not already cached
    if file_name not in DATAFRAME_CACHE.keys():
        try:
            DATAFRAME_CACHE[file_name] = pd.read_csv(file_name)

        except FileNotFoundError:
            return {"error": f"DataFrame '{file_name}' not found in cache or on disk"}

        except Exception as e:
            return {"error": f"Error loading '{file_name}': {str(e)}"}

    # Into memory
    df: pd.DataFrame = DATAFRAME_CACHE[file_name]

    # If the target column is not found, return an error message
    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found in '{file_name}'"}

    # Extract the features and target columns
    X: pd.DataFrame = df.drop(columns=[target_column])
    y: pd.Series = df[target_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the model
    model: RandomForestClassifier = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Predict the target column
    y_pred: np.ndarray = model.predict(X_test)

    # Calculate the accuracy score
    acc: float = accuracy_score(y_test, y_pred)

    # Return the accuracy score
    return {"accuracy": acc}


@tool
def evaluate_regression_dataset(
    file_name: str, target_column: str
) -> Dict[str, Union[float, str]]:
    """
    Train and evaluate a regression model on a dataset using the specified target column.

    Parameters
    ----------
    file_name: str
        The name or path of the dataset stored in `DATAFRAME_CACHE`.
    target_column: str
        The name of the column to use as the regression target.

    Returns
    -------
    Dict[str, Union[float, str]]
        A dictionary with R² score and Mean Squared Error or an error message if the dataset
        is not found or the target column is not found.

    Examples
    --------
    >>> evaluate_regression_dataset(file_name="./data.csv", target_column="target")
    {"r2_score": 0.95, "mean_squared_error": 0.05}
    """

    # Try to get the DataFrame from cache, or load it if not already cached
    if file_name not in DATAFRAME_CACHE:
        try:
            DATAFRAME_CACHE[file_name] = pd.read_csv(file_name)

        except FileNotFoundError:
            return {"error": f"DataFrame '{file_name}' not found in cache or on disk"}

        except Exception as e:
            return {"error": f"Error loading '{file_name}': {str(e)}"}

    # Into memory
    df: pd.DataFrame = DATAFRAME_CACHE[file_name]

    # If the target column is not found, return an error message
    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found in '{file_name}'"}

    # Extract the features and target columns
    X: pd.DataFrame = df.drop(columns=[target_column])
    y: pd.Series = df[target_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the model
    model: RandomForestRegressor = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Predict the target column
    y_pred: np.ndarray = model.predict(X_test)

    # Calculate the R² score and Mean Squared Error
    r2: float = r2_score(y_test, y_pred)
    mse: float = mean_squared_error(y_test, y_pred)

    # Return the R² score and Mean Squared Error
    return {"r2_score": r2, "mean_squared_error": mse}


# All available tools
tools: List[StructuredTool] = [
    list_csv_files,
    preload_datasets,
    get_dataset_summaries,
    call_dataframe_method,
    evaluate_classification_dataset,
    evaluate_regression_dataset,
]
#########################################################################################

# Main code
#########################################################################################
# Initialize a dictionary to store the cached datasets. It must live outside any
# function. It creates a persistent storage space that all tools can access without
# explicitly passing it around.
DATAFRAME_CACHE: Dict[str, pd.DataFrame] = {}

# Structures prompt with essential components
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}"),  # Required for tool-calling agents
    ]
)

# Large Language Model
llm = init_chat_model(model=MODEL, model_provider="openai", streaming=False)

# This raw agent has significant limitations when used directly. It only performs a
# single step of reasoning and tool usage per invocation, then returns its intermediate
# thought process rather than a final answer. This behavior occurs because the agent
# doesn't automatically manage the full loop of thinking, acting, observing results, and
# continuing to reason until reaching a complete solution.
agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)

# Creates a complete, autonomous agent system by wrapping your basic agent with
# additional functionality.
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)

# Disables streaming mode for the agent
agent_executor.agent.stream_runnable = False

print("\n📊 Ask questions about your dataset (type 'exit' or 'quit' to quit):")
print(
    f" Example: Can you list for me all datasets available in {DATASET_FOLDER} folder?\n"
)

while True:
    user_input = input(" You: ")
    if user_input.strip().lower() in ["exit", "quit"]:
        print("see ya later")
        break

    result = agent_executor.invoke({"input": user_input})

    print(f" Agent: {result['output']}")
#########################################################################################
