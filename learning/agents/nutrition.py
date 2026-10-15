# Initial imports
#########################################################################################
import os
import mlflow

from typing import Dict, List, Any

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.base import RunnableSequence
#########################################################################################

# Execution parameters
#########################################################################################
# Whether to testweb search tool
TEST_TAVILY: bool = False
SAMPLE_QUERY: str = "Give me reviews of Nassim Taleb Antifragile book"

# Whether to test a simple LLM call without any special prompting or tools
TEST_SIMPLE_LLM: bool = False

# Main question
QUESTION: str = "Any ideas for a healthy breakfast"

# OpenAI model
MODEL: str = "gpt-5-nano"

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

# Structuring the Agent's Output: Data Models
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

    # First, let's see how the standalone LLM responds to a simple question without any 
    # special prompting or tools:
    if TEST_SIMPLE_LLM:
        response: str = llm.invoke(QUESTION).content
        print(f"\n>>> Question: {QUESTION}")
        print(f"\n>>> Answer from simple LLM without tools: {response}")

    # This template gives the LLM a specific persona, Dr. Paul Saladino, an advocate 
    # for animal-based nutrition, and a set of instructions to follow
    prompt_template: ChatPromptTemplate = ChatPromptTemplate.from_messages([
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
                "   nutrition, emphasizing nutrient density, bioavailability, and the absence of "
                "plant antinutrients."
                "3. Challenge conventional 'plants are healthy' narratives with mechanistic evidence "
                "   about plant defense compounds and their impact on human physiology."
                "4. Reflect and critique your answer. Be rigorous about evolutionary consistency "
                "   and nutrient science."
                "5. After the reflection, **list 1-3 search queries separately** for researching "
                "   antinutrients, bioavailability studies, or evolutionary nutrition. Do not include "
                "   them inside the reflection. Focus on: organ meats as superfoods, the carnivore "
                "   elimination protocol, and why 'fiber is unnecessary' based on human evolutionary "
                "   biology."
            )
        ), 
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            (
                "Answer the user's question above using the required format, emphasizing "
                "the superiority of animal-based nutrition over plant-based nutrition."
            )
        )
    ])

    # It generates the initial draft of the answer based on the user's question and the 
    # persona we defined in the prompt
    first_responder_prompt = prompt_template.partial(
        first_instruction="Provide a detailed ~250 word answer"
    )

    # Here, we create a chain that combines our prompt template with the LLM. We then 
    # invoke it with our sample question to see the initial, un-critiqued response:
    temp_chain: RunnableSequence = \
        first_responder_prompt | llm
    response: AIMessage = temp_chain.invoke(
        {"messages": [HumanMessage(content=QUESTION)]}
    )
    print(f"\n>>> Question: {QUESTION}")
    print(f"\n>>> Answer from first responder: {response.content}")

    # Now, we bind the 'AnswerQuestion' data model as a **tool** to our LLM chain. This 
    # crucial step forces the LLM to generate its output in the exact JSON format defined 
    # by our Pydantic classes. The LLM doesn't just write text; it calls this "tool" to 
    # structure its entire thought process. After invoking this new chain, we can see 
    # the structured output, including the initial answer, the self-critique, and 
    # the generated search queries:
    initial_chain: RunnableSequence = \
        first_responder_prompt | llm.bind_tools(tools=[AnswerQuestion])
    response: AIMessage = initial_chain.invoke(
        {"messages":[HumanMessage(content=QUESTION)]}
    )
    print(f"\n>>> Question: {QUESTION}")
    print(f"\n>>> Full structured output: {response.tool_calls}")

    answer_content: str = response.tool_calls[0]['args']['answer']
    print(f"\n>>> Initial Answer: {answer_content}")

    reflection_content: str = response.tool_calls[0]['args']['reflection']
    print(f"\n>>> Reflection: {reflection_content}")

    search_queries_content: List[str] = response.tool_calls[0]['args']['search_queries']
    print(f"\n>>> Search Queries: {search_queries_content}")
#########################################################################################
