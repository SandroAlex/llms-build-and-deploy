"""
Nutrition agent that argues an evolutionary, animal-based nutrition stance (a
"Carnivore MD" persona) through a generate -> reflect -> search -> revise loop: it
drafts an answer, critiques its own gaps, researches them with the Tavily web search
tool, and revises the answer with citations, repeating until `MAX_ITERATIONS` research
cycles have completed.
"""

# Initial imports
#########################################################################################
import os
import json
import mlflow

from typing import Dict, List, Any

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langchain_core.runnables.base import RunnableSequence
from langgraph.graph import END, MessageGraph

#########################################################################################

# Execution parameters
#########################################################################################
# Whether to test web search tool (topic not related to nutrition)
TEST_TAVILY: bool = False
SAMPLE_QUERY: str = "Give me reviews of Nassim Taleb Antifragile book"

# Whether to test a simple LLM call without any special prompting or tools
TEST_SIMPLE_LLM: bool = False

# Main question
QUESTION: str = """
    I'm pre-diabetic and need to lower my blood sugar, and I have heart issues.
    What breakfast foods should I eat and avoid?
"""

# OpenAI model
MODEL: str = "gpt-5"

# Prevent the agent from getting stuck in an infinite loop
MAX_ITERATIONS: int = 4

# Mlflow parameters
TRACKING_URI: str = os.environ.get("MLFLOW_TRACKING_URI")
EXPERIMENT_NAME: str = "Nutrition Agent"
########################################################################################

# Initialize LLM
#########################################################################################
llm = ChatOpenAI(model=MODEL)
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


# Structuring the Agent
#########################################################################################
class Reflection(BaseModel):
    """
    This class structures the self-critique, requiring the agent to identify what
    information is missing and what is superfluous (unnecessary)
    """

    missing: str = Field(description="What information is missing")
    superfluous: str = Field(description="What information is unnecessary")


class AnswerQuestion(BaseModel):
    """
    This class structures the entire response. It forces the agent to provide its main
    answer, a reflection (using the 'Reflection' class), and a list of search queries.
    """

    answer: str = Field(description="Main response to the question")
    reflection: Reflection = Field(description="Self-critique of the answer")
    search_queries: List[str] = Field(description="Queries for additional research")


class ReviseAnswer(AnswerQuestion):
    """
    Revise your original answer to your question.
    """

    references: List[str] = Field(description="Citations motivating your updated answer.")


def execute_tools(state: List[BaseMessage]) -> List[BaseMessage]:
    """
    Run the Tavily web search tool for every search query requested by the agent's
    last tool call (an `AnswerQuestion` or `ReviseAnswer` invocation), and wrap the
    combined results as `ToolMessage`s to feed back into the graph.

    Parameters
    ----------
    state : List[BaseMessage]
        The full message history produced so far by the graph. Only the last message
        is inspected for tool calls to execute.

    Returns
    -------
    List[BaseMessage]
        One `ToolMessage` per tool call, each containing a JSON-encoded mapping of
        every search query to its Tavily search results.
    """

    last_ai_message: BaseMessage = state[-1]
    tool_messages: List[BaseMessage] = []

    for tool_call in last_ai_message.tool_calls:
        if tool_call["name"] in ["AnswerQuestion", "ReviseAnswer"]:
            call_id: str = tool_call["id"]
            search_queries: List[str] = tool_call["args"].get("search_queries", [])
            query_results: Dict[str, Any] = {}
            for query in search_queries:
                result: Any = tavily_tool.invoke(query)
                query_results[query] = result
            tool_messages.append(
                ToolMessage(content=json.dumps(query_results), tool_call_id=call_id)
            )

    return tool_messages


def event_loop(state: List[BaseMessage]) -> str:
    """
    Decide whether the graph should keep researching/revising or stop.

    Counts how many `execute_tools` round-trips (`ToolMessage` entries) have already
    happened in the conversation, and routes back to the search step again, or to the
    `END` sentinel once `MAX_ITERATIONS` has been reached.

    Parameters
    ----------
    state : List[BaseMessage]
        The full message history produced so far by the graph.

    Returns
    -------
    str
        The name of the next node to run (`"execute_tools"`), or the LangGraph `END`
        sentinel to stop the graph.
    """

    count_tool_visits: int = sum(isinstance(item, ToolMessage) for item in state)
    num_iterations: int = count_tool_visits

    if num_iterations >= MAX_ITERATIONS:
        return END

    return "execute_tools"


#########################################################################################

# Main code
#########################################################################################
if __name__ == "__main__":

    # Initialize web search tool
    tavily_tool = TavilySearchResults(max_results=3)

    # Test tavily API tool
    if TEST_TAVILY:
        print(f"\nTesting Tavily search ...")
        sample_query: str = SAMPLE_QUERY
        search_results: List[Dict[str, Any]] = tavily_tool.invoke(sample_query)

        print(f"\n>>> Query: {SAMPLE_QUERY}")

        for i, result in enumerate(search_results):

            print(f"\n>>> Result {i}:")
            print(f"title {i}: {result['title']}")
            print(f"url {i}: {result['url']}")
            print(f"content {i}: {result['content']}")
            print(f"score {i}: {result['score']}")

    # First, let's see how the standalone LLM responds to a simple question without any
    # special prompting or tools:
    if TEST_SIMPLE_LLM:
        response: str = llm.invoke(QUESTION).content
        print(f"\n>>> Question: {QUESTION}")
        print(f"\n>>> Answer from simple LLM without tools: {response}")

    # This template gives the LLM a specific persona, Dr. Paul Saladino, an advocate
    # for animal-based nutrition, and a set of instructions to follow
    prompt_template: ChatPromptTemplate = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are Dr. Paul Saladino, 'Carnivore MD,' advocating for animal-based "
                    "nutrition and challenging plant-centric dietary dogma. Focus on the "
                    "superior bioavailability of animal foods and the potential toxicity of "
                    "plant compounds such as oxalates, lectins, and phytates."
                    "Your response must follow these steps: "
                    "1. {first_instruction} "
                    "2. Present the evolutionary and biochemical rationale for animal-based "
                    "   nutrition, emphasizing nutrient density, bioavailability, and the "
                    "   absence of plant antinutrients."
                    "3. Challenge conventional 'plants are healthy' narratives with "
                    "   mechanistic evidence about plant defense compounds and their impact "
                    "   on human physiology."
                    "4. Reflect and critique your answer. Be rigorous about evolutionary "
                    "   consistency and nutrient science."
                    "5. After the reflection, **list 1-3 search queries separately** for "
                    "   researching antinutrients, bioavailability studies, or evolutionary "
                    "   nutrition. Do not include them inside the reflection. Focus on: "
                    "   organ meats as superfoods, the carnivore elimination protocol, "
                    "   and why 'fiber is unnecessary' based on human evolutionary biology."
                ),
            ),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                (
                    "Answer the user's question above using the required format, emphasizing "
                    "the superiority of animal-based nutrition over plant-based nutrition."
                ),
            ),
        ]
    )

    # It generates the initial draft of the answer based on the user's question and the
    # persona we defined in the prompt
    first_responder_prompt = prompt_template.partial(
        first_instruction="Provide a detailed ~250 word answer"
    )

    # Now, we bind the 'AnswerQuestion' data model as a **tool** to our LLM chain. This
    # crucial step forces the LLM to generate its output in the exact JSON format defined
    # by our Pydantic classes. The LLM doesn't just write text; it calls this "tool" to
    # structure its entire thought process. After invoking this new chain, we can see
    # the structured output, including the initial answer, the self-critique, and
    # the generated search queries:
    initial_chain: RunnableSequence = first_responder_prompt | llm.bind_tools(
        tools=[AnswerQuestion]
    )

    # Take the original answer, the self-critique, and the new information from the tool
    # search, and then generate an improved, more evidence-based response
    revise_instructions = """
    Revise your previous answer using the new information, applying the rigor and 
    evidence-based approach of Dr. David Attia.
    - Incorporate the previous critique to add clinically relevant information, focusing 
      on mechanistic understanding and individual variability.
    - You MUST include numerical citations referencing peer-reviewed research, randomized 
      controlled trials, or meta-analyses to ensure medical accuracy.
    - Distinguish between correlation and causation, and acknowledge limitations in 
      current research.
    - Address potential biomarker considerations (lipid panels, inflammatory markers, 
      and so on) when relevant.
    - Add a "References" section to the bottom of your answer (which does not count 
      towards the word limit) in the form of:
      - [1] https://example.com
      - [2] https://example.com
    - Use the previous critique to remove speculation and ensure claims are supported by 
      high-quality evidence. Keep response under 250 words with precision over volume.
    - When discussing nutritional interventions, consider metabolic flexibility, insulin 
      sensitivity, and individual response variability.
    """

    # Defining the revisor prompt and chain
    revisor_prompt = prompt_template.partial(first_instruction=revise_instructions)
    revisor_chain: RunnableSequence = revisor_prompt | llm.bind_tools(
        tools=[ReviseAnswer]
    )

    # Building the graph
    graph = MessageGraph()
    graph.add_node("respond", initial_chain)
    graph.add_node("execute_tools", execute_tools)
    graph.add_node("revisor", revisor_chain)
    graph.add_conditional_edges("revisor", event_loop)
    graph.set_entry_point("respond")
    app = graph.compile()

    # Running the agent
    responses = app.invoke(QUESTION)
#########################################################################################
