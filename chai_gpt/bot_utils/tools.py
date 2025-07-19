import os
import yaml

from langchain.chat_models import init_chat_model
from typing import List, Union
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

Message = Union[SystemMessage, HumanMessage, AIMessage, ToolMessage]

GOOGLE_API_KEY_NAME = "GOOGLE_API_KEY"

class Conversation:
    def __init__(self, system_message: SystemMessage) -> None:
        self._messages: List[Message] = [system_message]

    def add(self, message: Message) -> None:
        self._messages.append(message)

    def get_messages(self):
        return self._messages


def load_yaml_config(filepath):
    """
    Loads a YAML file and returns its contents as a dictionary.
    
    :param filepath: Path to the YAML file
    :return: dict containing the YAML file's contents
    """
    try:
        with open(filepath, 'r') as file:
            data = yaml.safe_load(file)
        return None, data
    except FileNotFoundError:
        return ValueError(f"Could not Config Key from {filepath}."), None


def set_api_key(key_name, config_file_name):
   ## Load config + read and set api key as an environment variable.
    err, config = load_yaml_config(config_file_name)
    if err:
        return ValueError(f"Could not load config file {config_file_name}.")
    if key_name not in config:
        return ValueError(f"Could not find Config Key in the config file.")
    # By convention key_name in the config file will be the same as the name of the environment variable.
    # See Langchain docs for the correct env variable names.
    api_key = config[key_name]
    os.environ[key_name] = api_key
    return None


def set_google_api_key(config_file_name = "keys-config.yml"):
    """
    Sets the api key as an env variable and then
    """
    return set_api_key(GOOGLE_API_KEY_NAME, config_file_name)


def setup_google_model(model_name="gemini-2.5-flash", model_provider="google_genai"):
    """
    Sets the api key as an env variable and then
    Initializes a langchain anthropic chat model
    """
    if not os.environ.get(GOOGLE_API_KEY_NAME, "").strip():
        return ValueError(f"Set the API Key `{GOOGLE_API_KEY_NAME}` for Google Gemini API in as an environment variable to use the chat bot."), None
    return None, init_chat_model(model_name, model_provider=model_provider)