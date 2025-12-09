from pathlib import Path
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

from bot_utils.tools import set_open_api_key, setup_openai_model

from chai_gpt_search.models import CookingActions
from chai_gpt_search.db_search import load_db, search_db_given_actions, CookingActionsDictionary


@dataclass
class LLMHandle:
    llm: BaseChatModel

llm_handle = LLMHandle

def parse_actions_flow(user_query, debug=False):
    ## Build a human message object.
    ## recreate the LLM object with structured output enabled.
    actions_dict = CookingActionsDictionary()
    chat_messages = []
    base_llm = llm_handle.llm
    structured_llm = base_llm.with_structured_output(CookingActions)

    # Step 1 - Add details to the user's query.
    system_prompt = (
        "You are an AI cooking assistant with knowledge about a wide range of cooking techniques and recipes."
        "The system that you are a part of is a kitchen equipment search engine."
        "The query from the user is provided in natural language. it may be a statement, question, a sceanrio or a recipe."
        "Respond to the user's query and provide extra details where necessary especially about required cooking actions like peeling, cutting, sauteeing etc."
        "If the query has nothing to do with cooking, then respond by saying 'Not Applicable! Not related to Cooking!"
        "Do not end your reply with follow up questions."
    )
    system_message = SystemMessage(system_prompt)
    chat_messages.append(system_message)
    add_details_message = HumanMessage(user_query.strip())
    chat_messages.append(add_details_message)
    step_1_op = base_llm.invoke(chat_messages)
    chat_messages.append(step_1_op) # Append the LLM response to the chat

    # Step 2 - Extract actions details.
    extract_cmd_message = HumanMessage(f"Based on the above, list all the actions that apply and populate the object. Choose only from the following list.\n {actions_dict.pretty_print()}")
    chat_messages.append(extract_cmd_message)

    if debug:
        for msg in chat_messages:
            print(f"[{type(msg)}]")
            print(msg.content)
            print()

    return structured_llm.invoke(chat_messages)

def launch():
    _ = set_open_api_key(config_file_name="keys-config.yml")
    _, llm = setup_openai_model(model_name="gpt-5-chat-latest")
    db = load_db()
    llm_handle.llm = llm
    prompt = """
        How do I prepare the spices before cooking when making Indian chicken curry?
    """
    cooking_actions = parse_actions_flow(prompt, True)
    print(cooking_actions.actions)
    # print('######')
    # search_result = search_db_given_actions(db, cooking_actions)
    # for result in search_result:
    #     print(result.pretty_print())
    #     print('_____________')



if __name__ == '__main__':
    launch()
