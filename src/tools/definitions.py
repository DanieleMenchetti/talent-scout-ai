import json
import os
import re
from datetime import datetime
from typing import Optional

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

_ddg_search = DuckDuckGoSearchRun()


@tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo and return the results."""
    try:
        result = _ddg_search.run(query)
        return result or "No result found."
    except Exception as e:
        return f"Search error: {str(e)}"

@tool
def save_report(filename: str, content: str) -> str:
    """Save a report to the results directory."""
    output_dir = "./results/"
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return json.dumps({"status": "saved", "path": path})


# Tool lists per ogni agente (ogni agente riceve solo i tool di cui ha bisogno)
SEARCHER_TOOLS = [web_search]
REPORTER_TOOLS = [save_report]
