import streamlit as st
from main import ask_github_mcp_agent

st.set_page_config(
    page_title="GitHub MCP Agent",  
    page_icon="🐙",
)

st.title("🐙 GitHub MCP Agent")
st.caption("Paste a Github repository once, than chat with it")

#State Session
if "history" not in st.session_state:   
    st.session_state.history = []   

if "repo" not in st.session_state:
    st.session_state.repo = ""

if "logs" not in st.session_state:
    st.session_state.logs = []

@st.dialog("🪵 State Logs",width="large")   
def show_logs_dialog(): 
    st.code("\n\n".join(st.session_state.logs),language="text") 

#sidebar

with st.sidebar:    

    st.subheader("Repository")  

    repo_input = st.text_input(
        "Github Repository",    
        value=st.session_state.repo,    
        placeholder="github/github-mcp-server", 
        help="Example: github/github-mcp-server",
    )

if st.button("Load Repository"):    
    st.session_state.repo = repo_input.strip()  
    st.success(f"Loaded: {st.session_state.repo}")

st.divider()

if st.button("Start New Conversation"): 
    st.session_state.history = []   
    st.session_state.logs = []  
    st.rerun()  

st.divider()

with st.expander("🪵State Logs"):   
    if st.session_state.logs:   
        if st.button("Expand Logs",use_container_width=True):   
            show_logs_dialog()

    else:   
        st.caption("No logs yet.Ask a question to see state logs here")

#Display Chat History (Visual UI Bubbles)

for msg in st.session_state.history:    
    with st.chat_message(msg["role"]):  
        st.write(msg["content"])

#chat Input & Execution(Stateless backend call)


user_prompt = st.chat_input("Ask a question about the repository")

if user_prompt:
    if not st.session_state.repo:
        st.warning("Please load a GitHub repository first.")
        st.stop()

    # Save the user's message
    st.session_state.history.append({
        "role": "user",
        "content": user_prompt,
    })

    with st.chat_message("user"):
        st.write(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, logs = ask_github_mcp_agent(
                    repo=st.session_state.repo,
                    question=user_prompt,
                )
            except Exception as exc:
                # Never leave the spinner hanging: surface the failure instead.
                error_text = f"Request failed - {type(exc).__name__}: {exc}"
                st.session_state.logs.append(error_text)
                st.session_state.history.append({
                    "role": "assistant",
                    "content": error_text,
                })
                st.error(error_text)
                st.stop()

        st.session_state.logs.append(logs)
        st.write(answer)

    # Save the assistant's response
    st.session_state.history.append({
        "role": "assistant",
        "content": answer,
    })

    st.rerun()
