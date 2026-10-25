from typing import Annotated, Optional, TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict): 
    """ 
    state shared across LangGraph workflow. 

    messages:   
        Conversation managed by LangGraph.

    repo:   
       The currently selected Github repository for this conversation.  
       Example: "github/github-mcp-server"
    """ 
    messages: Annotated[list,add_messages]
    repo: Optional[str]