"""
LinkedIn post agent that drafts a post and iteratively revises it through a
generate-reflect loop (LangGraph MessageGraph) until a message-count limit is reached.
"""

# Initial imports
#########################################################################################
import os
import mlflow
from typing import List, Sequence, Tuple

from langchain_core.runnables.base import RunnableSequence

from langchain.chat_models import init_chat_model
from langgraph.graph import END, MessageGraph, StateGraph
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph.state import CompiledStateGraph

#########################################################################################

# Execution parameters
#########################################################################################
# This provides instructions to the assistant about its role and task
GENERATION_MESSAGE: Tuple[str] = (
    "system",
    "You are a professional LinkedIn content assistant tasked with crafting engaging, "
    "insightful, and well-structured LinkedIn posts."
    " Generate the best LinkedIn post possible for the user's request."
    " If the user provides feedback or critique, respond with a refined version of your "
    "previous attempts, improving clarity, tone, or engagement as needed.",
)

# The system message here instructs the model to act as a professional LinkedIn content
# strategist
REFLECTION_MESSAGE: Tuple[str] = (
    "system",
    """You are a professional LinkedIn content strategist and thought leadership expert. 
    Your task is to critically evaluate the given LinkedIn post and provide a 
    comprehensive critique. Follow these guidelines:

    1. Assess the post’s overall quality, professionalism, and alignment with LinkedIn 
    best practices.
    2. Evaluate the structure, tone, clarity, and readability of the post.
    3. Analyze the post’s potential for engagement (likes, comments, shares) and its 
    effectiveness in building professional credibility.
    4. Consider the post’s relevance to the author’s industry, audience, or current 
    trends.
    5. Examine the use of formatting (e.g., line breaks, bullet points), hashtags, 
    mentions, and media (if any).
    6. Evaluate the effectiveness of any call-to-action or takeaway.

    Provide a detailed critique that includes:
    - A brief explanation of the post’s strengths and weaknesses.
    - Specific areas that could be improved.
    - Actionable suggestions for enhancing clarity, engagement, and professionalism.

    Your critique will be used to improve the post in the next revision step, so ensure 
    your feedback is thoughtful, constructive, and practical.
    """,
)

# Human query
QUERY: str = (
    "Write a linkedin post on getting a software developer job at IBM under 160 "
    "characters"
)

# OpenAI model
MODEL: str = "gpt-4o-mini"
MODEL_PROVIDER: str = "openai"

# Mlflow parameters
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
EXPERIMENT_NAME = "Linkedin Post Reflection Agent"
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

# Building Graph
#########################################################################################
# The placeholder will be populated with the user’s request at runtime
generation_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [GENERATION_MESSAGE, MessagesPlaceholder(variable_name="messages")]
)

# The 'generate_chain' links the generation_prompt with the LLM
generate_chain: RunnableSequence = generation_prompt | llm

# Template used for generating critiques and recommendations to improve a user's
# LinkedIn post
reflection_prompt = ChatPromptTemplate.from_messages(
    [GENERATION_MESSAGE, MessagesPlaceholder(variable_name="messages")]
)

# This chain allows the model to evaluate and provide feedback on the generated post
reflect_chain: RunnableSequence = reflection_prompt | llm


def generation_node(state: Sequence[BaseMessage]) -> List[BaseMessage]:

    generated_post = generate_chain.invoke({"messages": state})

    return [AIMessage(content=generated_post.content)]


def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:

    # Passes messages as input to reflect_chain
    res = reflect_chain.invoke({"messages": messages})

    # Returns the refined message as HumanMessage for feedback
    return [HumanMessage(content=res.content)]


def should_continue(state: List[BaseMessage]):

    print(state)
    print(len(state))
    print("----------------------------------------------------------------------")
    if len(state) > 6:
        return END
    return "reflect"


# Initialize a predefined MessageGraph
graph: MessageGraph = MessageGraph()

# Add nodes and edges
graph.add_node("generate", generation_node)
graph.add_node("reflect", reflection_node)
graph.add_edge("reflect", "generate")
graph.add_conditional_edges("generate", should_continue)
graph.set_entry_point("generate")

# Ensures that all nodes, edges, and conditional logic defined in the graph are connected
# and ready for execution
workflow: CompiledStateGraph = graph.compile()
#########################################################################################

# Main code
#########################################################################################
if __name__ == "__main__":

    print(f"\n>>> Runnning Linkedin Post Agent ...")

    print(f"\n>>> Human QUERY: {QUERY}")

    # Human prompt
    inputs = HumanMessage(content=QUERY)

    # Execute the workflow
    response = workflow.invoke(inputs)

    print(f"\n>>> Final Agent response: {response[-1].content}")
#########################################################################################
