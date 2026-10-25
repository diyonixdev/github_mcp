
# AGENT.md — GitHub MCP Agent

## 1. Project Overview

This project is a Streamlit-based AI chat application that allows users to load a GitHub repository and ask questions about its code.

The application uses:
- Python
- Streamlit for the user interface
- LangGraph for agent orchestration
- LangChain for messages and tool integration
- MCP (Model Context Protocol) for communication with GitHub tools
- `langchain-mcp-adapters` to convert MCP tools into LangChain-compatible tools
- Groq as the LLM provider

The user selects a repository in `owner/repo` format and asks questions about its files. The agent retrieves relevant information through GitHub's remote MCP server and generates an answer using the LLM.

## 2. Project Structure

The main files and their responsibilities are:

- `app.py`: Streamlit interface, repository input, chat interaction, and logs display.
- `main.py`: LangGraph construction, assistant/tool execution loop, and answer extraction.
- `mcp_client.py`: Connects to GitHub's remote MCP server and filters available tools.
- `groq_client.py`: Initializes the Groq LLM.
- `prompt.py`: Contains the agent's system instructions.
- `state.py`: Defines the shared LangGraph state schema.

Inspect the actual repository before making changes. If filenames or responsibilities differ, follow the existing implementation and update this documentation when appropriate.

## 3. Application Workflow

The expected workflow is:

1. The application starts using Streamlit.
2. The user enters a GitHub repository in `owner/repo` format.
3. The selected repository is stored in Streamlit session state.
4. The user submits a question.
5. `app.py` calls the agent function in `main.py`.
6. The agent receives the question and selected repository.
7. LangGraph invokes the assistant node.
8. The LLM decides whether a GitHub tool is required.
9. If a tool call is requested, `ToolNode` executes it.
10. The tool result is returned to the assistant.
11. The assistant generates a final response.
12. The answer and execution logs are returned to Streamlit.

Loading a repository should not imply that all repository files have been downloaded or indexed. Repository information is retrieved through tools when needed.

## 4. MCP Integration

The project uses GitHub's remote MCP server:

`https://api.githubcopilot.com/mcp/`

The MCP client obtains the available tools and filters them using an explicit allowlist.

The intended initial allowlist is:

```python
GITHUB_TOOLS_ALLOWLIST = {"get_file_contents"}
```

Only tools included in the allowlist should be provided to the LLM.

Do not enable additional tools without a clear requirement and explicit implementation changes.

The current design is intended for reading repository file contents. Do not assume that the agent can create issues, push commits, modify repositories, or execute arbitrary GitHub actions.

### MCP implementation rules

- Keep MCP connection logic in `mcp_client.py`.
- Use `langchain-mcp-adapters` for MCP-to-LangChain integration.
- Handle connection failures and missing tools clearly.
- Do not silently continue if the required tool is unavailable.
- Do not expose GitHub access tokens in logs or error messages.
- Verify actual tool names and schemas rather than assuming them.

## 5. LangGraph Architecture

The expected graph structure is:

```text
START
  |
  v
assistant
  |
  v
Does the AI response contain tool calls?
  | Yes                 | No
  v                     v
tools                   END
  |
  v
assistant
```

The assistant node should:

1. Read the current graph state.
2. Obtain the user's question and selected repository.
3. Include repository context in the LLM request.
4. Apply the configured tool-result length guard.
5. Invoke the LLM with the system instructions and conversation messages.
6. Return the resulting AI message.

The tools node executes requested MCP tools and appends the results as tool messages.

After tool execution, control should return to the assistant so it can interpret the results and produce a final answer.

Use the existing LangGraph routing implementation, such as `tools_condition`, where appropriate.

Ensure the graph terminates when the assistant produces a response without tool calls.

## 6. State Management

The shared graph state is expected to contain:

- `messages`: A list of LangChain messages, managed using the appropriate message reducer.
- `repo`: The selected repository in `owner/repo` format, or `None` when unavailable.

Inspect `state.py` before changing the schema.

### Stateless agent behavior

Each invocation of the agent should use a fresh message list containing the current question.

The repository should be supplied separately on every request.

Unless explicitly requested, do not introduce:

- Persistent LangGraph checkpoints.
- Cross-request agent memory.
- Automatic reuse of previous user questions in the LLM context.
- Persistent conversation storage.

Streamlit may maintain chat history for display without passing the entire history to the agent.

Caching the compiled graph at process level is acceptable, provided request messages and repository context remain correctly isolated.

## 7. LLM Configuration

The Groq integration should remain centralized in `groq_client.py`.

The existing `get_llm()` function should initialize the configured chat model and support tool binding.

Rules:

- Read credentials from environment variables or the project's existing secret-management system.
- Never hard-code API keys or tokens.
- Never print credentials.
- Verify that the configured model exists and supports tool calling.
- Handle authentication errors, rate limits, unavailable models, and network failures.
- Avoid unnecessary changes to the `get_llm()` interface.
- Keep provider-specific code isolated from the graph implementation.

If a different LLM provider is requested, modify the provider integration while preserving the rest of the architecture wherever possible.

Do not assume that a model identifier remains available indefinitely.

## 8. Token and Context Limits

The reference implementation uses the following tool-message limit:

```python
MAX_TOOL_MESSAGE_CHARS = 6000
```

Tool messages exceeding this limit should be truncated before being sent back to the LLM.

Requirements:

- Apply truncation to tool messages only.
- Do not truncate the user's original question unnecessarily.
- Make truncation behavior clear and predictable.
- Test messages below, at, and above the configured limit.
- Avoid loading unnecessary MCP tool schemas.
- Prefer targeted file retrieval over unnecessarily large tool responses.

If the limit changes, update the relevant tests and documentation.

## 9. Streamlit UI Guidelines

The Streamlit application should provide:

- A GitHub repository input.
- A button or explicit action to select/load the repository.
- A chat interface for asking questions.
- A readable assistant response.
- An expandable logs section for debugging.

Requirements:

- Store the selected repository in session state.
- Validate empty or malformed repository input.
- Provide useful error messages.
- Keep the interface responsive during agent execution.
- Avoid displaying secrets or sensitive environment values.
- Do not imply that loading a repository indexes every file.
- Preserve existing UI behavior unless a requested feature requires a change.

## 10. Error Handling

Handle errors at the relevant layer.

Potential failure points include:

- Missing environment variables.
- Missing GitHub authentication.
- MCP server connection failures.
- Empty MCP tool results.
- Missing allowed tools.
- Invalid or inaccessible repositories.
- LLM authentication failures.
- Unsupported model identifiers.
- Provider rate limits.
- Tool execution errors.
- Unexpected message formats.
- Empty final answers.
- LangGraph execution failures.

Error messages should explain what failed and, when possible, how to fix it.

Do not silently swallow exceptions.

Logs should identify the failing stage without exposing secrets. If a fallback is used, make it explicit.

## 11. Logging and Debugging

Execution logs should help identify:

- Whether the graph was initialized.
- Whether MCP tools were discovered.
- Which tools passed the allowlist.
- Whether the assistant requested a tool.
- Whether tool execution succeeded.
- Whether the LLM generated a final answer.
- Whether an error occurred.

Do not log complete credentials, access tokens, or other sensitive configuration values.

The MCP client should be runnable independently to help diagnose tool discovery:

```bash
python mcp_client.py
```

The application should be started from the project root using:

```bash
streamlit run app.py
```

Use the existing virtual environment and dependency configuration.

## 12. Environment Variables and Security

- Store credentials in `.env`, Streamlit secrets, or another approved secret store.
- Ensure secret files are excluded from Git.
- Use placeholder values in `.env.example`.
- Never commit API keys or personal access tokens.
- Never include real secrets in documentation or generated code.
- Do not request that the user paste credentials into source files.
- Do not increase token permissions without explaining why.

Inspect the actual source code to confirm environment variable names before changing configuration or writing setup instructions.

## 13. Coding Rules for AI Agents

Before editing:

1. Read the relevant source files.
2. Inspect dependencies and configuration.
3. Trace the current request flow.
4. Identify the root cause of the problem.
5. Propose the smallest appropriate change.
6. Preserve existing working behavior.

While editing:

- Follow the existing code style.
- Keep responsibilities separated by module.
- Avoid unnecessary dependencies.
- Avoid duplicate initialization logic.
- Do not rewrite the entire project to fix a localized bug.
- Do not change public function signatures without checking their callers.
- Do not invent APIs or package methods.
- Do not change unrelated functionality.

After editing:

1. Run relevant tests.
2. Check Python syntax and imports where practical.
3. Verify affected behavior.
4. Report the exact files changed.
5. State which tests and commands were actually executed.
6. Clearly disclose any integration tests that could not be run because credentials or network access were unavailable.

Never claim that a test passed unless it was actually executed successfully.

## 14. Testing Expectations

Add or update tests when practical for:

- Graph routing with and without tool calls.
- Repository context injection.
- Fresh message state for each request.
- MCP tool filtering.
- Missing-tool behavior.
- Tool-message truncation.
- Empty LLM responses.
- API and network failures.
- Repository input validation.
- Final answer extraction.
- Log generation.

Mock the LLM and MCP tools for unit tests so core logic can be tested without live credentials.

Keep live integration tests separate from unit tests.

## 15. Scope Restrictions

Unless explicitly requested:

- Do not replace Streamlit.
- Do not replace LangGraph.
- Do not replace Groq.
- Do not replace the MCP adapter.
- Do not introduce persistent agent memory.
- Do not enable extra GitHub tools.
- Do not redesign the entire project.
- Do not modify repository contents through GitHub tools.
- Do not add unrelated features.
- Do not modify credentials or authentication scopes unnecessarily.

If a request conflicts with these restrictions, explain the conflict and ask for clarification when it materially affects security or expected behavior.

## 16. Definition of Done

A task is complete when:

- The requested behavior is implemented.
- The root cause has been addressed rather than hidden.
- The existing architecture remains intact unless a redesign was requested.
- Stateless request behavior is preserved.
- Secrets remain protected.
- Relevant error cases are handled.
- Relevant tests or checks have been run.
- Documentation is updated when setup or behavior changes.
- The final response clearly summarizes changes, tests, and remaining limitations.
