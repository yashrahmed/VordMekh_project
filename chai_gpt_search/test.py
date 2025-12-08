from pathlib import Path
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

from bot_utils.tools import set_open_api_key, setup_openai_model

from chai_gpt_search.models import CookingActions
from chai_gpt_search.db_search import load_db


@dataclass
class LLMHandle:
    llm: BaseChatModel

llm_handle = LLMHandle

# ATTRIBUTE_PROMPT_PATH = Path(__file__).resolve().parent / "resources" / "attribute-prompt.txt"
# try:
#     ATTRIBUTE_PROMPT = ATTRIBUTE_PROMPT_PATH.read_text(encoding="utf-8").strip()
# except FileNotFoundError:
#     ATTRIBUTE_PROMPT = ""

def parse_actions(user_query):
    ## Build a human message object.
    ## recreate the LLM object with structured output enabled.
    base_llm = llm_handle.llm
    structured_llm = base_llm.with_structured_output(CookingActions)

    system_prompt = (
        "You are an extraction agent that maps natural language queries about cooking"
        "to the CookingActions schema. Mark an action True when the query implies the "
        "cook needs that skill, False only when the text explicitly rules it out, and "
        "leave it null when it is not mentioned."
    )
    system_message = SystemMessage(system_prompt)
    human_message = HumanMessage(
        "Identify which actions are required for the following request. "
        "Only rely on the provided actions list.\n\nRequest:\n" + user_query.strip()
    )

    return structured_llm.invoke([system_message, human_message])

def launch():
    _ = set_open_api_key(config_file_name="keys-config.yml")
    _, llm = setup_openai_model(model_name="gpt-5-chat-latest")
    llm_handle.llm = llm
    prompt = """
        I need stir fry some chicken in order to get the color.
      """
    cooking_actions = parse_actions(prompt)
    print(cooking_actions.pretty_print())


if __name__ == '__main__':
    print(load_db())
    # launch()
