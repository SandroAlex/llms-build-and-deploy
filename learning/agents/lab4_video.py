"""
Youtube specific dependencies:
+ pytube==15.0.0
+ youtube-transcript-api==1.2.4
+ yt-dlp==2026.7.4
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
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
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

# Main query given by user
QUERY: str = "I want to summarize youtube video: https://www.youtube.com/watch?v=aYhacHNHTEs in english"

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

# Mlflow setup
#########################################################################################
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# Captures LLM calls, tool calls, chain/graph steps as nested spans
mlflow.langchain.autolog()

# Optional: also trace raw OpenAI HTTP calls (token usage, latency)
mlflow.openai.autolog()
#########################################################################################


# Ancillar methods
#########################################################################################
def execute_tool(tool_call: Dict[str, Any]) -> ToolMessage:
    """
    Execute single tool call and return 'ToolMessage'
    """

    try:
        result = tool_mapping[tool_call["name"]].invoke(tool_call["args"])
        return ToolMessage(content=str(result), tool_call_id=tool_call["id"])
    except Exception as e:
        return ToolMessage(content=f"Error: {str(e)}", tool_call_id=tool_call["id"])


#########################################################################################

# Main code
#########################################################################################
# Initialize the language model from OpenAI
llm = init_chat_model(model=MODEL, model_provider=MODEL_PROVIDER)

# Enables the LLM to access and use your custom YouTube tools during conversations
llm_with_tools = llm.bind_tools(tools)
