import os
import sys
from dotenv import load_dotenv

# Force reload dotenv with override
load_dotenv(override=True)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_chain import get_llm
from config import OLLAMA_CLOUD_PREFIX

# UI imports
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.prompt import Prompt
except ImportError:
    print("Please install rich to see the beautiful CLI: pip install rich")
    sys.exit(1)

console = Console()

def simulate_chat():
    model_id = f"{OLLAMA_CLOUD_PREFIX}gemma4:31b-cloud"
    llm = get_llm(model=model_id, streaming=True) # Enabled streaming
    
    console.rule("[bold cyan]🤖 AI CLI Test Interface")
    
    user_query = "Say the word 'Hello' and nothing else."
    
    console.print(f"[bold cyan]User:[/bold cyan] {user_query}")
    
    try:
        with console.status("[bold green]Agent is thinking...", spinner="dots"):
            response = llm.invoke(user_query)
            
        md = Markdown(response.content)
        panel = Panel(md, title="🤖 Agent Response", border_style="green", expand=False)
        console.print(panel)
        
    except Exception as e:
        console.print(f"[bold red]Error invoking model:[/bold red] {e}")

if __name__ == "__main__":
    simulate_chat()
