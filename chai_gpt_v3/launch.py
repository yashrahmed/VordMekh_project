from bot_utils.tools import set_open_api_key, setup_openai_model, Conversation
from langchain_core.messages import SystemMessage, HumanMessage

from rich.prompt import Prompt
from rich.text import Text
from rich.console import Console

from pydantic import BaseModel, Field
from typing import Optional

class ChaiOrderState(BaseModel):
    selected_chai_recipe: Optional[str] = Field(
        None,
        description=(
            "Mapped chai type: 'Masala Chai', 'Adrak Chai', 'Sulaimani Chai', "
            "'Kashmiri Chai', 'Kahwah', 'Unsupported chai type', or null if unknown."
        )
    )
    number_of_servings: Optional[float] = Field(
        None,
        description="Number of chai servings as a float. Null if unknown."
    )
    at_campsite: Optional[bool] = Field(
        None,
        description="True if confirmed at a campsite, false if confirmed not at one, null if unknown."
    )
    has_missc_content: bool = Field(
        False,
        description="True if unrelated content is included, false otherwise."
    )
    does_user_want_to_make_chai: bool = Field(
        False,
        description="True if the user explicitly states that they want to make chai or ask for help making it. False otherwise"
    )

def build_messages_for_order_parsing(current_state: ChaiOrderState, user_input: str):
    system_prompt = f"""
        You are a structured output parser for chai making.  
        Your job is to read the user’s natural language text and fill in the following state object fields.  
        You must return **only** a JSON object matching the provided schema. Do not include extra commentary.

        The schema represents the current state of the chai making.  
        Some fields may already be known (non-null). If a value is already filled in, do not overwrite it unless the new input clearly contradicts the previous value.
        The currently known value are specified in the current state description below.
        
        ### Field Requirements:

        1. **selected_chai_recipe** (String)  
        - Map the chai type from the user input to one of the following values:  
            "Masala Chai", "Adrak Chai", "Sulaimani Chai", "Kashmiri Chai", "Kahwah", "Unsupported chai type",
            or `null` if unknown.  
        - Correct spelling errors and infer from context (e.g., "ginger tea" → `"Adrak Chai"`).  
        - If user mentions a chai type not in the list or does not specify a specific chai type, set to `"Unsupported chai type"`.

        2. **number_of_servings** (float)  
        - Extract the numeric value of servings.  
        - If no serving count is given, set to `null`.

        3. **at_campsite** (boolean | null)  
        - `true` if the user explicitly confirms they are at a campsite.  
        - `false` if explicitly confirmed they are NOT at a campsite.  
        - `null` if location is unknown.

        4. **has_missc_content** (boolean)  
        - `true` if the user includes content unrelated to making chai in their message, and message is not a blank message.  
        - `false` otherwise.

        5. **does_user_want_to_make_chai** (boolean)  
        - `true` if the user explicitly states that they want to make chai or ask for help making it.  
        - `false` otherwise.


        ### Current State:
        {current_state.model_dump_json()}

        ### Instructions:
        - Parse the user’s new input and update only the fields that can be determined.  
        - Keep the rest unchanged.  
        - Return only the updated JSON state.
    """
    user_input = user_input or " "
    return [SystemMessage(system_prompt), HumanMessage(user_input)]


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

def validate_user_input_step(current_state: ChaiOrderState):
    problems_with_state = []

    if not current_state.does_user_want_to_make_chai:
        problems_with_state.append("User hasn't stated that he wants to make chai! User needs to confirm this.")

    if current_state.selected_chai_recipe is None:
        problems_with_state.append("Selected chai type is unknown. User must select a chai type that is valid.")
    elif current_state.selected_chai_recipe == "Unsupported chai type":
        problems_with_state.append("User has selected an unsupported chai type. You cannot help with that. User must select a chai type that is valid.")
    
    if current_state.number_of_servings is None:
        problems_with_state.append("Number of servings is unknown and the user needs to select a valid number of servings between 1 and 6 so that the recipe can be correctly calculated.")
    elif current_state.number_of_servings < 1 or current_state.number_of_servings > 6:
        problems_with_state.append("Number of servings is invalid and the user needs to select a valid number of servings between 1 and 6 so that the recipe can be correctly calculated.")
    
    if current_state.at_campsite is None:
        problems_with_state.append("It is unknown whether is user is at a campsite. This information is requried to properly determine the tools required for chai prep.")
    
    all_relevant_value_known = len(problems_with_state) == 0 # All relevant values are known given that there are no problems with the state at this point.

    if current_state.has_missc_content:
        problems_with_state.append("User has added unrelated content in their input.")

    print(current_state)
    return all_relevant_value_known, problems_with_state


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
    
    llm = llm.with_structured_output(ChaiOrderState)
    current_state = ChaiOrderState(selected_chai_recipe="Masala Chai")
    
    user_input = ''

    display_bot_message(console, "Hi there! I am Chai Assistant. Pleased to meet you.")
    while user_input != '/exit':
        user_input = prompt_user()
        cleaned_user_input = user_input.strip().lower()
        if cleaned_user_input == '/exit':
            break
        if not len(cleaned_user_input):
            display_bot_message(console, "I didn't quite get that. Could you try saying that again please?")
            continue

        # Step 1 - Parsing the user input.
        current_state = llm.invoke(build_messages_for_order_parsing(current_state, user_input))
        x, y = validate_user_input_step(current_state)
        print(x)
        print(y)
        print('---------')



if __name__ == '__main__':
    main()