"""
For Youtube specific dependencies, run the following command:
uv pip install pytube==15.0.0 youtube-transcript-api==1.2.4 yt-dlp==2026.7.4
"""

# Initial imports
#########################################################################################
import json
import logging
import os
import re
import warnings
from typing import Dict, List, Union, Any

import yt_dlp
from IPython.display import JSON, display
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.tools import tool
from langchain_core.tools.structured import StructuredTool
from pytube import Search, YouTube
from youtube_transcript_api import YouTubeTranscriptApi

import mlflow

#########################################################################################

# Suppress warnings
#########################################################################################
# General warnings
warnings.filterwarnings("ignore")

# Suppress pytube errors
pytube_logger = logging.getLogger("pytube")
pytube_logger.setLevel(logging.ERROR)

# Suppress yt-dlp warnings
yt_dpl_logger = logging.getLogger("yt_dlp")
yt_dpl_logger.setLevel(logging.ERROR)
#########################################################################################

# Execution parameters
#########################################################################################
# OpenAI model
MODEL: str = "gpt-4o-mini"
MODEL_PROVIDER: str = "openai"

# Mlflow parameters
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
EXPERIMENT_NAME = "Youtube Agents"

# yt-dlp parameters
# Path to a Netscape-format cookies.txt exported from a logged-in YouTube session,
# used to reduce bot-detection blocks. There's no local browser inside this container,
# so cookies are exported on the host and mounted in (see secrets/youtube_cookies.txt)
# Set to None / unset to disable (anonymous requests, more likely to be blocked)
YT_DLP_COOKIES_FILE: str | None = os.environ.get(
    "YT_DLP_COOKIES_FILE", "/llm_app/secrets/youtube_cookies.txt"
)
#########################################################################################


# Tools
#########################################################################################
@tool
def extract_video_id(url: str) -> str:
    """
    Extracts the 11-character YouTube video ID from a URL.

    Parameters
    ----------
    url : str
        A YouTube URL containing a video ID

    Returns
    -------
    str
        Extracted video ID or error message if parsing fails
    """

    # Regex pattern to match video IDs
    pattern = r"(?:v=|be/|embed/)([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else "Error: Invalid YouTube URL"


@tool
def fetch_transcript(video_id: str, language: str = "en") -> str:
    """
    Fetches the transcript of a YouTube video.

    Parameters
    ----------
    video_id : str
        The 11-eleven YouTube video identifier (e.g., "dQw4w9WgXcQ")
    language : str
        Language code for the transcript (e.g., "en", "es")

    Returns
    -------
        str: The transcript text or an error message
    """

    try:
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id, languages=[language])

        return " ".join([snippet.text for snippet in transcript.snippets])

    except Exception as e:
        return f"Error: {str(e)}"


@tool
def search_youtube(query: str) -> List[Dict[str, str]]:
    """
    Search YouTube for videos matching the query.

    Parameters
    ----------
    query : str
        The search term to look for on YouTube

    Returns
    -------
    List[Dict[str, str]]
        List of dictionaries containing video titles and IDs in format:
        [{'title': 'Video Title', 'video_id': 'abc123', 'url': 'https://youtu.be...'}]
        Returns error message if search fails
    """
    try:
        s = Search(query)
        return [
            {
                "title": yt.title,
                "video_id": yt.video_id,
                "url": f"https://youtu.be/{yt.video_id}",
            }
            for yt in s.results
        ]
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_full_metadata(url: str) -> Dict[str, Union[str, Dict[str, str]]]:
    """
    Extract metadata given a YouTube URL, including title, views, duration, channel,
    likes, comments, and chapters.

    Parameters
    ----------
    url : str
        youtube URL

    Return
    ------
    Dict[str, Union[str, Dict[str, str]]]
        Comprehensive information about the video, including its title, view count,
        duration, channel name, like count, comment count, and any chapter markers
    """

    ydl_opts = {
        "quiet": True,
        "logger": yt_dpl_logger,
        "cookiefile": YT_DLP_COOKIES_FILE,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

        return {
            "title": info.get("title"),
            "views": info.get("view_count"),
            "duration": info.get("duration"),
            "channel": info.get("uploader"),
            "likes": info.get("like_count"),
            "comments": info.get("comment_count"),
            "chapters": info.get("chapters", []),
        }


@tool
def get_thumbnails(url: str) -> List[Dict]:
    """
    Get available thumbnails for a YouTube video using its URL.

    Parameters
    ----------
    url : str
        YouTube video URL (any format)

    Returns
    -------
    List[Dict]
        List of dictionaries with thumbnail URLs and resolutions in YouTube's native
        order
    """

    try:
        ydl_opts = {
            "quiet": True,
            "logger": yt_dpl_logger,
            "cookiefile": YT_DLP_COOKIES_FILE,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            thumbnails = []
            for t in info.get("thumbnails", []):
                if "url" in t:
                    thumbnails.append(
                        {
                            "url": t["url"],
                            "width": t.get("width"),
                            "height": t.get("height"),
                            "resolution": f"{t.get('width', '')}x{t.get('height', '')}".strip(
                                "x"
                            ),
                        }
                    )

            return thumbnails

    except Exception as e:
        return [{"error": f"Failed to get thumbnails: {str(e)}"}]


# Available tools
tools: List[StructuredTool] = [
    extract_video_id,
    fetch_transcript,
    search_youtube,
    get_full_metadata,
    get_thumbnails,
]

# This mapping will be useful later when you need to programmatically invoke specific
# tools based on their names
tool_mapping = {
    "get_thumbnails": get_thumbnails,
    "extract_video_id": extract_video_id,
    "fetch_transcript": fetch_transcript,
    "search_youtube": search_youtube,
    "get_full_metadata": get_full_metadata,
}
#########################################################################################


# Ancillar methods
#########################################################################################
def execute_tool(tool_call: Dict[str, Any]) -> ToolMessage:
    """
    Execute single tool call and return 'ToolMessage'
    """

    try:
        result = tool_mapping[tool_call["name"]].invoke(tool_call["args"])
        content = json.dumps(result) if isinstance(result, (dict, list)) else str(result)
    except Exception as e:
        content = f"Error: {str(e)}"

    return ToolMessage(content=content, tool_call_id=tool_call["id"])


def process_tool_calls(messages):
    """
    This function handles the core processing logic of your recursive chain.

    It takes the current conversation history and:
    1. Identifies the most recent message in the conversation
    2. Extracts all tool calls from that message and executes them in parallel using your
    'execute_tool' helper
    3. Updates the message history by adding the tool response messages
    4. Gets the next response from the language model based on the updated conversation
    5. Returns the complete updated message history with both tool responses and the new
    LLM response
    """

    # Most recent message in conversation
    last_message = messages[-1]

    # Execute all tool calls in parallel
    tool_messages = [execute_tool(tc) for tc in getattr(last_message, "tool_calls", [])]

    # Add tool responses to message history
    updated_messages = messages + tool_messages

    # Get next LLM response
    next_ai_response = llm_with_tools.invoke(updated_messages)

    return updated_messages + [next_ai_response]


def should_continue(messages):
    """
    Determines whether your recursive process should continue or terminate.

    It:

    1. Takes the current message history and examines the last message
    2. Checks if that message contains any tool calls using the getattr function
    (which safely handles cases where the attribute might not exist)
    3. Returns a boolean value - True if there are more tool calls to process, and
    False when you reach a point where the LLM has provided a final answer without
    requesting additional tools
    """

    last_message = messages[-1]

    return bool(getattr(last_message, "tool_calls", None))


def _recursive_chain(messages):
    """
    This function implements the actual recursion that powers your dynamic tool
    calling process.

    It:

    1. It first checks the stopping condition using the should_continue function to
    determine if more tools need to be called
    2. If more tool calls are needed, it processes those calls using your
    'process_tool_calls' function and then recursively calls itself with the updated
    messages
    3. If no more tool calls are needed, it returns the final message history, which
    contains the complete conversation, including the LLM's final response
    """

    if should_continue(messages):
        new_messages = process_tool_calls(messages)

        return _recursive_chain(new_messages)

    return messages
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

# Main code
#########################################################################################
# Initialize the language model from OpenAI
llm = init_chat_model(model=MODEL, model_provider=MODEL_PROVIDER)

# Enables the LLM to access and use your custom YouTube tools during conversations
llm_with_tools = llm.bind_tools(tools)

# After defining the recursive function, you'll wrap it in a RunnableLambda to make it
# compatible with LangChain's chain architecture
recursive_chain = RunnableLambda(_recursive_chain)

# Chain that can handle any type of query requiring any number of tool calls
universal_chain = (
    # The first step converts the user query into a properly formatted 'HumanMessage'
    # object
    RunnableLambda(lambda x: [HumanMessage(content=x["query"])])
    |
    # The second step sends this initial message to your tool-equipped LLM and adds the
    # LLM's first response to the message history
    RunnableLambda(lambda messages: messages + [llm_with_tools.invoke(messages)])
    |
    # The final step passes the conversation to your recursive chain, which will handle
    # all subsequent tool calls until the LLM provides a final answer
    recursive_chain
)

# Test it
QUERY_USER: Dict[str, str] = {
    "query": "Show top 3 more visited videos about Stoicism with metadata and thumbnails"
}

try:
    response = universal_chain.invoke(QUERY_USER)
    print("\nUS Trending Videos:\n", response[-1])

except Exception as e:
    print("Non-critical network error while fetching US trending videos:", e)
