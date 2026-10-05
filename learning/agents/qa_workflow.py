"""
LangGraph QA agent that validates a question, decides whether it relates to the guided
project, and answers it with an LLM when relevant context is available.
"""

# Initial imports
#########################################################################################
import os
import mlflow
from typing import TypedDict, Optional, Dict, Union

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph
from langgraph.graph import END

#########################################################################################

# Execution parameters
#########################################################################################
CONTEXT: str = (
    "This guided project is about using LangGraph, a Python library to design state-based "
    "workflows. LangGraph simplifies building complex applications by connecting modular "
    "nodes with conditional edges."
)

# OpenAI model
MODEL: str = "gpt-4o-mini"
MODEL_PROVIDER: str = "openai"

# Mlflow parameters
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
EXPERIMENT_NAME = "QA Workflow Graph"
#########################################################################################

# Initialize LLM
#########################################################################################
# Initialize the language model from OpenAI
llm = init_chat_model(model=MODEL, model_provider=MODEL_PROVIDER)
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


# Building state, nodes and edges
#########################################################################################
# Defining the State for the QA Workflow
class QAState(TypedDict):
    """
    This state will be used to store and manage the data as the workflow progresses,
    including the user's question, the relevant context (if any), and the generated
    answer
    """

    # 'question' stores the user's input question. It can be a string or None if not
    # provided
    question: Optional[str]

    # 'context' stores relevant context about the guided project, if the question
    # pertains to it. If the question isn't related to the project, this will be None
    context: Optional[str]

    # 'answer' stores the generated response or answer. It can be None until the answer
    # is generated
    answer: Optional[str]


def input_validation_node(state: QAState) -> Dict[str, Union[bool, str]]:
    """
    In this node, we validate the user's input (the question). The node checks whether
    the question is provided and if it's not empty. If the question is empty, it
    returns an error message indicating that the question cannot be empty. If the
    question is valid, it proceeds to the next node
    """

    # Extract the question from the state, and strip any leading or trailing spaces
    question: str = state.get("question", "").strip()

    # If the question is empty, return an error message indicating invalid input
    if not question:
        return {"valid": False, "error": "Question cannot be empty."}

    # If the question is valid, return valid status
    return {"valid": True}


def context_provider_node(state: QAState) -> Dict[str, Union[str, None]]:
    """
    This node checks if the question is related to the guided project. If it mentions
    "LangGraph" or "guided project," it provides the relevant context. Otherwise, it sets
    the context to `None':
    """

    question: str = state.get("question", "").lower()

    # Check if the question is related to the guided project
    if "langgraph" in question or "guided project" in question:
        context: str = CONTEXT
        return {"context": context}

    # If unrelated, set context to null
    return {"context": None}


def llm_qa_node(state: QAState) -> Dict[str, str]:
    """
    Utilizes an LLM to answer user questions based on the provided context. If the
    question is unrelated to the guided project, the node handles this gracefully by
    returning a predefined response
    """

    # Extract the question and context from the state
    question: str = state.get("question", "")
    context: Union[str, None] = state.get("context", None)

    # Check for missing context and return a fallback response
    if not context:
        return {"answer": "I don't have enough context to answer your question."}

    # Construct the prompt dynamically
    prompt: str = (
        f"Context: {context}\nQuestion: {question}\nAnswer the question based on the provided context."
    )

    # Use LangChain's ChatOpenAI to get the response
    try:
        response = llm.invoke(prompt)
        return {"answer": response.content.strip()}

    except Exception as e:
        return {"answer": f"An error occurred: {str(e)}"}


#########################################################################################

# Main code
#########################################################################################
if __name__ == "__main__":

    # Create an example object
    qa_state_example: QAState = QAState(
        question="What is the purpose of this guided project?",
        context="This project focuses on building a chatbot using Python.",
        answer=None,
    )

    # Initialize the state-based graph for our QA workflow
    qa_workflow = StateGraph(QAState)

    # This node is responsible for validating whether the user-provided question is
    # non-empty and properly formatted
    qa_workflow.add_node("InputNode", input_validation_node)

    # This node determines whether the question is related to the guided project and
    # provides relevant context accordingly
    qa_workflow.add_node("ContextNode", context_provider_node)

    # This node leverages an LLM (IBM's Granite 3.3) to answer questions based on the
    # provided context
    qa_workflow.add_node("QANode", llm_qa_node)

    # This is the first node that will be executed when the workflow starts, ensuring
    # that the user's input is validated before any other operations occur
    qa_workflow.set_entry_point("InputNode")

    # Build graph edges
    qa_workflow.add_edge("InputNode", "ContextNode")
    qa_workflow.add_edge("ContextNode", "QANode")
    qa_workflow.add_edge("QANode", END)

    # Converts the workflow into an executable application that processes input data and
    # follows the defined flow
    qa_app = qa_workflow.compile()

    # 1: Irrelevant question
    print(qa_app.invoke({"question": "What is the weather today?"}))

    # 2: Relevant question
    print(qa_app.invoke({"question": "What is LangGraph?"}))
