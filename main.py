import asyncio
import io
import contextlib
import json
import os
import time
from collections import Counter
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.messages import AIMessage,HumanMessage,ToolMessage
from langgraph.graph import END,START,StateGraph
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import ToolNode

from groqclient import get_llm
from mcp_client import get_mcp_tools
from prompt import SYSTEMMESSAGE
from state import State

load_dotenv()

#llm and tool setup
_graph = None
MAX_TOOL_MESSAGE_CHARS = 3000  # Reduced to stay under Groq TPM limit (8000 tokens)
MAX_MESSAGE_HISTORY = 10  # Keep only last N messages to control context size

#Hard limits so no single stage can block the Streamlit spinner forever.
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))          # one model call
TOOL_TIMEOUT = float(os.getenv("TOOL_TIMEOUT", "45"))        # one MCP tool call
AGENT_TIMEOUT = float(os.getenv("AGENT_TIMEOUT", "180"))     # whole request
# Each assistant -> tools round costs TWO supersteps (assistant + tools), so a
# multi-step GitHub investigation needs well over 10. LangGraph's default is 25.
GRAPH_RECURSION_LIMIT = int(os.getenv("GRAPH_RECURSION_LIMIT", "25"))
# Loop guard: how many times the model may repeat an identical tool call
# (same tool, same arguments) before the graph stops instead of spinning.
MAX_IDENTICAL_TOOL_CALLS = int(os.getenv("MAX_IDENTICAL_TOOL_CALLS", "2"))

#Timestamped trace for the current request. Printed to the console and surfaced
#in the Streamlit "State Logs" panel so a hang can be located exactly.
_TRACE: list[str] = []


def _log(message: str) -> str:
    stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{stamp}] {message}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:  # Windows cp1252 consoles: never let logging break the agent
        print(line.encode("ascii", "replace").decode("ascii"), flush=True)
    _TRACE.append(line)
    return line

#to control token

def _truncate_if_tool_message(msg): 
    """cap oversize tool outputs to protect Groq TPM limits"""  
    if not isinstance(msg, ToolMessage):#if not tool message nothing to be done
       return msg   
    content = str(msg.content)#convert message into string => for counting length   
    if len(content) <= MAX_TOOL_MESSAGE_CHARS:      
        return msg  
    truncated = content[:MAX_TOOL_MESSAGE_CHARS] + f"\n...(truncated,{len(content)}chars total)"    
    return ToolMessage(content=truncated,tool_call_id=msg.tool_call_id,name=msg.name)# covert back into and handover to langgraph

def _tool_call_signatures(message) -> list[tuple[str, str]]:
    """Stable (tool name, canonical args) signature for every tool call in a message.

    Used by the router to detect the model re-issuing a call it has already made
    with identical arguments, which would otherwise loop forever.
    """
    signatures: list[tuple[str, str]] = []
    for call in getattr(message, "tool_calls", None) or []:
        name = str(call.get("name", ""))
        args = call.get("args", {})
        try:
            key = json.dumps(args, sort_keys=True, default=str)
        except (TypeError, ValueError):
            key = str(args)
        signatures.append((name, key))
    return signatures

def _tool_error_to_message(error: Exception) -> str:
    """Convert a tool runtime failure into a ToolMessage the model can read.

    ToolNode's default handler re-raises anything that is not an argument
    error, which would crash the whole graph instead of letting the agent
    report the problem to the user.
    """
    detail = getattr(error, "message", None) or str(error)
    _log(f"tools: failed with {type(error).__name__}: {detail}")
    return (
        f"Tool execution failed ({type(error).__name__}): {detail} "
        f"- Do not issue the identical tool call again; answer using what you "
        f"already have and state the limitation."
    )

#build graph function
async def _build_graph():   
    _log("build: connecting to MCP server to load tools ...")
    t_build = time.perf_counter()
    tools = await get_mcp_tools()
    _log(f"build: {len(tools)} tool(s) loaded in {time.perf_counter()-t_build:.2f}s")
    print(f"\nLoaded tools: {len(tools)}")
    for tool in tools:  
        print(f"- {tool.name}") 

    _log("build: binding tools to LLM ...")
    llm_with_tools = get_llm().bind_tools(tools) ##reading tools to the llm => making it agent
    _log("build: tools bound")

    #1.Assistant Node
    async def assistant(state:State):   
        repo = state.get("repo")    # 1 key is repo
        messages= list(state["messages"])  #2 is message/user question


        # Inject repository context into the latest human prompt if available
        if messages and repo and isinstance(messages[-1], HumanMessage): #user ke question ke saath GitHub repository ki information bhi jodna, taaki AI ko pata ho ki kis repository ke baare mein answer dena hai.
            original_question = messages[-1].content    
            messages[-1] = HumanMessage(
                content=f"Repository:\n{repo}\n\nQuestion:\n{original_question}"

            )

        # Truncate tool response to save tokens
        messages = [_truncate_if_tool_message(msg) for msg in messages]

        _log(f"assistant: LLM invoke start (timeout={LLM_TIMEOUT}s, {len(messages)} msg(s))")
        t_llm = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                llm_with_tools.ainvoke(
                    [SYSTEMMESSAGE] + messages
                ),
                timeout=LLM_TIMEOUT,
            )
        except asyncio.TimeoutError as exc:
            _log(f"assistant: LLM invoke TIMED OUT after {LLM_TIMEOUT}s")
            raise TimeoutError(
                f"The LLM did not respond within {LLM_TIMEOUT}s "
                f"(model: check GROQ_API_KEY, rate limits and connectivity)."
            ) from exc
        _log(
            f"assistant: LLM invoke done in {time.perf_counter()-t_llm:.2f}s "
            f"-> {len(response.tool_calls)} tool call(s)"
        )

        print("\nLLM Tool Calls:")  
        print(response.tool_calls)  

        return{"messages": [response]}





    
    # 2 Routing logic

    def route_assistant(state:State):  #Is code ka kaam hai decide karna ki AI ke response ke baad ab kya karna hai - tool chalana hai ya process khatam karna hai. 
        last_message = state["messages"][-1]
        pending = _tool_call_signatures(last_message)

        # Normal termination: the model produced a plain answer -> END.
        if not pending:
            return "end"

        # Loop guard: count identical (tool, args) calls already issued by
        # earlier assistant turns. Routing to tools again with the very same
        # arguments cannot make progress, so stop at END instead.
        already_issued: Counter = Counter()
        for previous in state["messages"][:-1]:
            already_issued.update(_tool_call_signatures(previous))

        repeated = [sig for sig in pending if already_issued[sig] >= MAX_IDENTICAL_TOOL_CALLS]
        if repeated:
            _log(
                f"route: repeated tool call {repeated[0][0]} issued "
                f"{already_issued[repeated[0]] + 1}x -> stopping at END"
            )
            return "end"

        return "tools" 




    #2b Tool execution wrapper: timestamps the call and caps it so a stalled
    #MCP/network response cannot block the request indefinitely.
    tool_node = ToolNode(tools, handle_tool_errors=_tool_error_to_message)

    async def tools_node(state: State, config):
        last = state["messages"][-1]
        names = [tc.get("name") for tc in getattr(last, "tool_calls", []) or []]
        _log(f"tools: executing {names} (timeout={TOOL_TIMEOUT}s)")
        t_tool = time.perf_counter()
        try:
            # config must be forwarded: ToolNode needs LangGraph's runtime
            # config, otherwise it fails before the tool even runs.
            result = await asyncio.wait_for(
                tool_node.ainvoke(state, config), timeout=TOOL_TIMEOUT
            )
        except asyncio.TimeoutError as exc:
            _log(f"tools: TIMED OUT after {TOOL_TIMEOUT}s -> {names}")
            raise TimeoutError(
                f"MCP tool {names} did not finish within {TOOL_TIMEOUT}s. "
                f"Check GitHub API connectivity or raise TOOL_TIMEOUT."
            ) from exc
        _log(f"tools: done in {time.perf_counter()-t_tool:.2f}s")
        return result

    #3compile Graph 
    builder= StateGraph(State)  
    builder.add_node("assistant",assistant) 
    builder.add_node("tools", tools_node)  

    builder.add_edge(START,"assistant") 
    builder.add_conditional_edges(
        "assistant",    
       route_assistant,  
       {"tools" : "tools", "end": END},
    )   
    builder.add_edge("tools","assistant")

    _log("build: LangGraph compiled")
    return builder.compile()


def get_state_logs(result):
    output = io.StringIO() #ek khaali notebook samjho, jisme tum agent ke kaam ka record likhogi.

    with contextlib.redirect_stdout(output):#Whatever the program prints on the screen, save it inside the output variable instead.
        print("\n---- STATE LOGS ----")

        if _TRACE:
            print("TIMESTAMPED TRACE:")
            for line in _TRACE:
                print(f"  {line}")

        if result.get("repo"):
            print(f"Repository: {result['repo']}")

        for i, msg in enumerate(result["messages"]): #enumerate() → Adds a number to each message.
            content = getattr(msg, "content", "No Content")#Try to get the actual text inside the message. If there is no content attribute, use "No Content" instead.
            print(f"[{i}] {msg.__class__.__name__}: {content}")#Print the message number, its type, and its content together.

            if hasattr(msg, "tool_calls") and msg.tool_calls:#Checks whether the message has a tool_calls attribute
                print(f"Tool calls: {msg.tool_calls}")#Checks whether that attribute contains any tool calls.

        print("-------------------------\n")

    return output.getvalue()


def ask_github_mcp_agent(repo: str, question: str):
    async def _run():
        global _graph #Your _graph contains the agent's workflow — the steps that tell it how to process a question and use GitHub tools. global keywords lets it get updated

        _TRACE.clear()
        t_start = time.perf_counter()
        _log(f"request start: repo={repo!r} question={question[:80]!r}")

        if _graph is None:
            _log("graph not built yet -> building")
            _graph = await _build_graph()
        else:
            _log("using cached graph")

        # STRICTLY STATELESS: Only pass the repo and current question.
        # Do not accumulate or accept chat_history lists.
        _log(
            f"graph.ainvoke start (limit={AGENT_TIMEOUT}s, "
            f"recursion_limit={GRAPH_RECURSION_LIMIT})"
        )
        t_invoke = time.perf_counter()
        try:
            result = await _graph.ainvoke(  #It sends your question and repository name into the agent's workflow.
                {
                    "messages": [HumanMessage(content=question)],
                    "repo": repo,
                },
                config={"recursion_limit": GRAPH_RECURSION_LIMIT},
            )
        except asyncio.TimeoutError as exc:
            _log(f"graph.ainvoke TIMED OUT after {time.perf_counter()-t_invoke:.2f}s")
            raise TimeoutError(
                f"Agent exceeded the {AGENT_TIMEOUT}s request budget. "
                f"Increase AGENT_TIMEOUT if the repository is large."
            ) from exc
        except GraphRecursionError as exc:
            # Should now be rare: the router stops repeated tool calls itself.
            _log(f"graph.ainvoke hit the recursion limit of {GRAPH_RECURSION_LIMIT}")
            raise RuntimeError(
                f"The agent did not reach a final answer within "
                f"{GRAPH_RECURSION_LIMIT} steps. Rephrase the question as smaller "
                f"steps, or raise GRAPH_RECURSION_LIMIT."
            ) from exc
        _log(f"graph.ainvoke done in {time.perf_counter()-t_invoke:.2f}s")

        logs = get_state_logs(result)

        answer = "No answer generated."

        for msg in reversed(result["messages"]):#Goes through messages backwards, starting with the latest one
            if isinstance(msg, AIMessage) and msg.content:#Checks whether the message is from the AI and contains content.
                answer = msg.content #Saves the AI's message as the answer.
                break

        if answer == "No answer generated.":
            # The loop guard stopped the run before the model emitted prose.
            # Fall back to the most recent tool result so the user still gets
            # something useful instead of a bare placeholder.
            for msg in reversed(result["messages"]):
                if isinstance(msg, ToolMessage) and msg.content:
                    answer = (
                        "I stopped after the model repeatedly issued the same "
                        "tool call, to avoid an endless loop. The most recent "
                        f"tool result was:\n\n{msg.content}"
                    )
                    _log("answer: fell back to last ToolMessage after loop guard")
                    break

        _log(f"request complete in {time.perf_counter()-t_start:.2f}s")
        return answer, logs #Returns two things: the AI's answer and the execution logs.

    #Hard outer bound: even if an inner await ignores its own timeout, this
    #guarantees ask_github_mcp_agent() returns or raises instead of blocking
    #the Streamlit script thread forever.
    return asyncio.run(asyncio.wait_for(_run(), timeout=AGENT_TIMEOUT))


 

