import os
from langchain_groq import ChatGroq
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"),override=True)

def get_llm(model: str = "openai/gpt-oss-20b",temperature=0 ) -> ChatGroq: 
    api_key = os.getenv("GROQ_API_KEY") 
    if not api_key: 
        raise ValueError("GROQ_API_KEY not found.please make sure it ia in you .env file")

    # langchain-groq defaults request_timeout to None, which is forwarded verbatim to
    # httpx as timeout=None and disables connect/read/write/pool timeouts entirely.
    # A stalled response then blocks forever, leaving the UI on "Thinking...".
    request_timeout = float(os.getenv("GROQ_REQUEST_TIMEOUT", "60"))
    max_retries = int(os.getenv("GROQ_MAX_RETRIES", "2"))

    llm = ChatGroq(
        model=model,
        temperature=temperature,
        api_key=api_key,
        request_timeout=request_timeout,
        max_retries=max_retries,
    )
    print(f"model loaded {model} (timeout={request_timeout}s, retries={max_retries})")
    return llm

if __name__ == "__main__":  
    get_llm()