from bot_utils.tools import set_open_api_key, setup_openai_model, SimpleChatBot
from langchain_core.tools import tool

from rich.prompt import Prompt
from rich.text import Text
from rich.console import Console


@tool
def simple_arithmetic(operation: str, a: float, b: float) -> float:
    """Perform simple arithmetic operations.
    
    Args:
        operation: The operation to perform (add, subtract, multiply, divide)
        a: First number
        b: Second number
        
    Returns:
        The result of the arithmetic operation
    """
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    else:
        raise ValueError(f"Unsupported operation: {operation}")


def display_bot_message(console: Console, message: str):
    text = Text()
    text.append("ChaiGPT: ", style="bold yellow3")
    text.append(message, style="white")
    console.print(text)


def display_exit_message(console: Console, message: str):
    text = Text()
    text.append(message, style="bold red")
    console.print(text)


def prompt_user():
    return Prompt.ask("[bold cyan]User[/bold cyan]")


def load_prompt_from_file(file_path):
    md_text = ''
    with open(file_path, "r", encoding="utf-8") as f:
        md_text = f.read()
    
    if not md_text.strip():
        raise ValueError("Markdown file is empty or contains only whitespace.")

    return md_text


def main():
    console = Console()

    err = set_open_api_key(config_file_name="keys-config.yml")
    if err:
        display_exit_message(console, str(err))
        return

    err, llm = setup_openai_model()
    if err:
        display_exit_message(console, str(err))
        return

    # Bind the arithmetic tool to the model
    # llm_with_tools = llm.bind_tools([simple_arithmetic])

    system_prompt = "You are ChaiGPT v3, a helpful assistant with access to arithmetic tools. You can perform basic math calculations when asked."
    greeting_message = "Hello! I am ChaiGPT v3 with arithmetic capabilities. I can help with math calculations and chai preparation. How can I help you today?"
    
    # Create bot with tool-enabled model
    bot = SimpleChatBot(llm, system_prompt, greeting_message, [simple_arithmetic])

    human_input = ""
    display_bot_message(console, greeting_message)
    while True:
        human_input = prompt_user()
        human_input = human_input.strip()
        if human_input == '/exit':
            break
        if not len(human_input):
            continue
        reply = bot.invoke(human_input)
        display_bot_message(console, reply)


if __name__ == '__main__':
    main()