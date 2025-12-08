from pathlib import Path
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

from bot_utils.tools import set_open_api_key, setup_openai_model

from chai_gpt_search.models import CookingActions
from chai_gpt_search.db_search import load_db, search_db_given_actions


@dataclass
class LLMHandle:
    llm: BaseChatModel

llm_handle = LLMHandle

def parse_actions(user_query):
    ## Build a human message object.
    ## recreate the LLM object with structured output enabled.
    base_llm = llm_handle.llm
    structured_llm = base_llm.with_structured_output(CookingActions)

    system_prompt = (
        "You are an extraction agent that maps natural language queries about cooking"
        "to the CookingActions schema. Mark an action True when the query implies that the "
        "cooking action is required, False only when the text explicitly rules it out, and "
        "leave it null when it is not mentioned. Users may post queries about specific steps involved in preparation."
        "In such cases, the only the actions relevant to the steps must be specified."
    )
    system_message = SystemMessage(system_prompt)
    human_message = HumanMessage(
        "Identify which actions are required for the following request.\n"
        + user_query.strip()
    )

    return structured_llm.invoke([system_message, human_message])

def launch():
    _ = set_open_api_key(config_file_name="keys-config.yml")
    _, llm = setup_openai_model(model_name="gpt-5.1")
    db = load_db()
    llm_handle.llm = llm
    prompt = """
        How do I prepare the spices before cooking when making Indian chicken curry?
      """
    cooking_actions = parse_actions(prompt)
    print(cooking_actions.pretty_print())
    print('######')
    search_result = search_db_given_actions(db, cooking_actions)
    for result in search_result:
        print(result.pretty_print())
        print('_____________')



if __name__ == '__main__':
    launch()
