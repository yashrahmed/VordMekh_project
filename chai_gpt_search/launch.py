from bot_utils.tools import set_open_api_key, setup_openai_model
from dataclasses import dataclass
from langchain_core.language_models.chat_models import BaseChatModel

from chai_gpt_search.db_search import load_db, search_db_given_actions
from chai_gpt_search.workflows import parse_actions_flow


@dataclass
class LLMHandle:
    llm: BaseChatModel

llm_handle = LLMHandle

def launch():
    _ = set_open_api_key(config_file_name="keys-config.yml")
    _, llm = setup_openai_model(model_name="gpt-5-chat-latest")
    db = load_db()
    llm_handle.llm = llm
    prompt = """
        How do I prepare the chicken before cooking when making Indian chicken curry?
    """
    cooking_actions = parse_actions_flow(llm_handle.llm, prompt)
    print(cooking_actions.actions)
    print('######')
    search_result = search_db_given_actions(db, cooking_actions)
    for result in search_result:
        print(result.pretty_print())
        print('_____________')

if __name__ == '__main__':
    launch()
