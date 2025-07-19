from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from bot_utils.tools import set_google_api_key, setup_google_model, Conversation

from rich.prompt import Prompt
from rich.text import Text
from rich.console import Console


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
    # Styled prompt with Monokai-style bold cyan
    return Prompt.ask("[bold cyan]User[/bold cyan]")


def load_prompt_from_file(file_path):
    # Load the raw Markdown content
    md_text = ''
    with open(file_path, "r", encoding="utf-8") as f:
        md_text = f.read()
    
    if not md_text.strip():
        raise ValueError("Markdown file is empty or contains only whitespace.")

    return md_text


class ChaiGPT:
    def __init__(self, llm: BaseChatModel, system_prompt: str, greeting: str) -> None:
        system_message = SystemMessage(system_prompt)
        self.conversation = Conversation(system_message)
        self.conversation.add(AIMessage(greeting))
        self.llm = llm

    def invoke(self, user_input: str) -> str:
        human_msg = HumanMessage(user_input)
        self.conversation.add(human_msg)
        response = self.llm.invoke(self.conversation.get_messages())
        ai_msg = AIMessage(response.content)
        self.conversation.add(ai_msg)
        return response.content


def main():
    #0. Set up terminal user interface.
    console = Console()

    #1. Set up key and model.
    err = set_google_api_key(config_file_name="keys-config.yml")
    if err:
        display_exit_message(console, str(err))
        return

    err, llm = setup_google_model()
    if err:
        display_exit_message(console, str(err))
        return

    #2. Set up chatbot
    system_prompt = load_prompt_from_file("chai_gpt_sys_prompt.md")
    greeting_message = "Hello! I am ChaiGPT, your personal assistant for preparing delicious chai. How can I help you today?"
    bot = ChaiGPT(llm, system_prompt, greeting_message)



    #3. Start conversations.
    human_input = ""
    display_bot_message(console, greeting_message)
    while human_input != '/exit':
        human_input = prompt_user()
        if human_input == '/exit':
            break
        reply = bot.invoke(human_input)
        display_bot_message(console, reply)


if __name__ == '__main__':
    main()