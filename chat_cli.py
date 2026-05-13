import os
import sys
from dotenv import load_dotenv

# Force reload dotenv
load_dotenv(override=True)

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.live import Live
except ImportError:
    print("Please install rich to use the CLI: pip install rich")
    sys.exit(1)

# Import from your existing backend
from rag_chain import get_llm
from config import OLLAMA_CLOUD_PREFIX

console = Console()

def main():
    console.rule("[bold cyan]🧠 AI Terminal Agent[/bold cyan]")
    console.print("[dim]Type 'quit' or 'exit' to stop. Type 'clear' to reset screen.[/dim]\n")
    
    # Initialize your model
    model_id = f"{OLLAMA_CLOUD_PREFIX}gemma4:31b-cloud"
    llm = get_llm(model=model_id, streaming=True)
    
    while True:
        try:
            # 1. Beautiful User Prompt
            user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")
            
            if user_input.lower() in ['quit', 'exit']:
                console.print("[dim]Goodbye![/dim]")
                break
            elif user_input.lower() == 'clear':
                console.clear()
                continue
            elif not user_input.strip():
                continue
                
            # 2. Status Spinner while waiting for first token
            with console.status("[bold green]Agent is thinking...", spinner="dots"):
                # We can use streaming to show the response being generated
                response = llm.stream(user_input)
                
            # 3. Render the output nicely
            full_response = ""
            for chunk in response:
                full_response += chunk.content
                
            # Render Markdown into a Panel
            md = Markdown(full_response)
            panel = Panel(md, title="🤖 Agent Response", border_style="green", expand=False)
            console.print(panel)
            
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            console.print("\n[dim]Interrupted. Type 'quit' to exit.[/dim]")
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {e}")

if __name__ == "__main__":
    main()
