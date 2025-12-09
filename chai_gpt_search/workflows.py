from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

from chai_gpt_search.db_search import CookingActionsDictionary
from chai_gpt_search.models import CookingActions



def parse_actions_flow(base_llm: BaseChatModel, user_query: str, debug: bool = False) -> CookingActions:
    """Multi-step workflow that enriches a query then extracts structured actions."""
    actions_dict = CookingActionsDictionary()
    chat_messages = []

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
    chat_messages.append(step_1_op)  # Append the LLM response to the chat

    # Step 2 - Extract actions details.
    extract_cmd_message = HumanMessage(
        "Based on the above, list all the actions that apply and populate the object. "
        f"Choose only from the following list.\n {actions_dict.pretty_print()}"
    )
    chat_messages.append(extract_cmd_message)

    if debug:
        for msg in chat_messages:
            print(f"[{type(msg)}]")
            print(msg.content)
            print()

    return structured_llm.invoke(chat_messages)
