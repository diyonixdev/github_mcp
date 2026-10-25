import os
import asyncio
import time

from dotenv import load_dotenv
from pathlib import Path
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv(Path(__file__).with_name(".env"), override=True)

# each Github tool schema is tool-heavy and
#groq 's free tier has a small toekns per miniute(TPM) ceiling
#mcp_client.py
GITHUB_TOOLS_ALLOWLIST = (
    "get_file_contents",
)
# The GitHub MCP endpoint is a remote HTTP server; without a cap a stalled
# handshake blocks tool loading forever.
MCP_TOOLS_TIMEOUT = float(os.getenv("MCP_TOOLS_TIMEOUT", "30"))
def get_mcp_client() -> MultiServerMCPClient:   #Defines a regular (synchronous) function that returns a MultiServerMCPClient instance. The -> MultiServerMCPClient is a type hint — it tells you what the function returns.
        github_pat = os.getenv("GITHUB_MCP_PAT")
        if not github_pat:  
           raise ValueError("GITHUB_MCP_PAT not found.set it to your .env file.")

        return MultiServerMCPClient(    
            {
                "github": {
                    "url": "https://api.githubcopilot.com/mcp/",    
                    "transport": "streamable_http",
                    "headers": {
                        "Authorization": f"Bearer {github_pat}",
                        },                
                  },  
             },     
        )

async def get_mcp_tools():  
      client = get_mcp_client() #client = get_mcp_client() is the line where you create and initialize your connection to the MCP server.
      t0 = time.monotonic()
      try:
            all_tools=await asyncio.wait_for(client.get_tools(), timeout=MCP_TOOLS_TIMEOUT) #Asynchronously fetches the list of all tools available from the GitHub MCP server. await pauses here until the network call completes.
      except asyncio.TimeoutError as exc:
            raise TimeoutError(
                  f"MCP server did not return tools within {MCP_TOOLS_TIMEOUT}s "
                  f"(transport: streamable_http). Check network/proxy access to "
                  f"api.githubcopilot.com."
            ) from exc
      print(f"[mcp] tools listed in {time.monotonic()-t0:.2f}s (limit {MCP_TOOLS_TIMEOUT}s)")

      #print name and description of each tool
      print(f"total count of tools {len (all_tools)}")    
      print("Available tools from all servers:")    
      for t in all_tools:   
            print(f"- **{t.name}**:{t.description}")

      filtered = [t for t in all_tools if t.name in GITHUB_TOOLS_ALLOWLIST ]    

      print("\nTools actually loaded:")
      for t in filtered:    
           print(f" - **{t.name}**: {t.description}")

      return filtered

if __name__ == "__main__": 
    asyncio.run(get_mcp_tools())
   
