from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

from bot_utils.tools import set_open_api_key, setup_openai_model
from dataclasses import dataclass


@dataclass
class LLMHandle:
    llm: BaseChatModel

llm_handle = LLMHandle

def launch():
    key_err = set_open_api_key(config_file_name="keys-config.yml")
    err, llm = setup_openai_model(model_name="gpt-5-chat-latest")
    llm_handle.llm = llm
    msg = HumanMessage("Hi there! What day is it today?")
    response = llm_handle.llm.invoke([msg])
    print(response.content)


if __name__ == '__main__':
    launch()