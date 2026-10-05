# Agents

This folder contains hands-on examples for learning how to develop LLM agents — programs that combine a language model with tools to reason about tasks and take actions in a loop.

## Contents

### `tool_calls.py`

A bare-bones look at what tool calling actually does under the hood, with no agent framework involved:

- **Manual tool-call loop** — binds `add`, `subtract`, and `multiply` to a chat model, inspects the `tool_calls` on the model's response, invokes the matching Python function by hand, and appends a `ToolMessage` back into the conversation before asking the model for a final answer.
- **Self-test** — each tool is invoked directly and asserted against an expected result before the LLM is ever called.
- **MLflow tracing** — logs the run under the `"Interactive LLM Agents with Tools"` experiment via `mlflow.langchain.autolog()` and `mlflow.openai.autolog()`.

Good starting point for understanding the primitives (`tool_calls`, `ToolMessage`, `tool_call_id`) that higher-level agent constructors hide from you.

### `simple_math.py`

A minimal math agent that demonstrates the core building blocks of agent development:

- **LangChain tools** — a `calculate_power` tool that parses two numbers from natural-language input and returns `base ** exponent`.
- **Agent setup** — uses `create_agent` with an OpenAI chat model (`gpt-5-nano`), a system prompt, and the tool list.
- **MLflow tracing** — logs the full agent run (LLM calls, tool invocations, and graph steps) via `mlflow.langchain.autolog()` and `mlflow.openai.autolog()`, under the `"Math Agent Test"` experiment.

### `data_wizard.py`

A multi-tool "chat with your CSVs" agent, built on the `AgentExecutor` / `create_openai_tools_agent` API — a useful contrast to `simple_math.py`'s `create_agent`:

- **Tools** — `list_csv_files`, `preload_datasets`, `get_dataset_summaries`, `call_dataframe_method`, `evaluate_classification_dataset`, and `evaluate_regression_dataset`, letting the agent discover CSVs, cache them in memory, inspect their schema, and train a quick `RandomForest` classifier/regressor to judge whether a dataset looks like a classification or regression problem.
- **In-memory dataset cache** — a module-level `DATAFRAME_CACHE` dict so tools don't re-read CSVs from disk or re-describe them to the LLM on every turn.
- **Interactive loop** — runs as a REPL (`input()` / `print()`) rather than a single `.invoke()` call; type `exit` or `quit` to stop.
- **MLflow tracing** — logs under the `"Data Wizard Agent Test"` experiment.
- **Sample data** — `data/classification-dataset.csv` and `data/regression-dataset.csv` ship alongside it as ready-made examples.

### `youtuber.py`

A YouTube research agent that hand-rolls its own recursive tool-calling chain instead of using a prebuilt agent constructor:

- **Tools** — `extract_video_id`, `fetch_transcript` (via `youtube_transcript_api`), `search_youtube` (via `pytube`), and `get_full_metadata` / `get_thumbnails` (via `yt_dlp`).
- **Custom recursive chain** — `should_continue`, `process_tool_calls`, and `_recursive_chain` implement the agent loop by hand as `RunnableLambda`s: execute every tool call from a turn, feed the results back to the model, and repeat until it stops requesting tools.
- **yt-dlp cookie auth** — `get_full_metadata` and `get_thumbnails` pass `cookiefile=YT_DLP_COOKIES_FILE` (default `/llm_app/secrets/youtube_cookies.txt`) to reduce YouTube's bot-detection blocks. Export cookies from a logged-in browser session and keep the file out of git (see the `secrets/` entry in `.gitignore`).
- **MLflow tracing** — logs under the `"Youtube Agents"` experiment.
- **Extra dependencies** (see the module docstring): `uv pip install pytube==15.0.0 youtube-transcript-api==1.2.4 yt-dlp==2026.7.4`.

### `qa_workflow.py`

A question-answering agent built directly on `langgraph.StateGraph`, rather than a prebuilt agent constructor or a hand-rolled chain — a look at the lower-level graph API that `create_agent` and LangGraph's prebuilt agents sit on top of:

- **Shared state** — a `QAState` `TypedDict` (`question`, `context`, `answer`) is threaded through every node and updated incrementally.
- **Nodes** — `input_validation_node` checks the question isn't empty, `context_provider_node` supplies project context only if the question mentions "LangGraph"/"guided project", and `llm_qa_node` calls the LLM to answer when context is available (returning a fallback message otherwise).
- **Linear graph** — nodes are wired as `InputNode → ContextNode → QANode → END` via `add_edge`, then compiled with `.compile()` into a runnable app.
- **MLflow tracing** — logs under the `"QA Workflow Graph"` experiment.
- Runs as a script (`if __name__ == "__main__":`) with two example invocations: one off-topic question and one on-topic question, to show both branches of `context_provider_node`.

## Running the examples

- Set `OPENAI_API_KEY` in your environment (or `.env`, if using the project's docker-compose setup).
- An MLflow tracking server must be reachable at the configured `MLFLOW_TRACKING_URI` for the autologging calls to succeed. Traces can be viewed in the MLflow UI at [http://localhost:5000](http://localhost:5000).
- `youtuber.py` additionally needs the YouTube-related packages listed above and, if you hit bot-detection blocks, a cookies file as described.
