from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

from bot_utils.tools import set_open_api_key, setup_openai_model
from dataclasses import dataclass

from .frames import ChaiPreparationIngredientsActionsFrame

@dataclass
class LLMHandle:
    llm: BaseChatModel

llm_handle = LLMHandle

def run_recipe_parse_test():
    recipe = """
    Ingredients (for 2 small “cutting” glasses)
	•	¾ cup water
	•	¾ cup full-fat milk
	•	2 teaspoons loose black tea (Assam or strong CTC)
	•	2 teaspoons sugar (adjust to taste)
	•	1 small piece fresh ginger, crushed
	•	2–3 green cardamom pods, lightly crushed
	•	1 small clove (optional)

    Steps
        1.	In a saucepan, bring the water, ginger, cardamom, and clove to a simmer.
        2.	Add the tea leaves and boil for 30–40 seconds so the color deepens and aroma rises.
        3.	Pour in the milk and sugar; keep stirring to prevent boil-over.
        4.	Let the mixture come up to a full rolling boil once, then reduce heat and simmer another minute.
        5.	“Pull” the chai a couple of times by pouring between two pots to aerate and develop froth (classic street-style).
        6.	Strain into small glasses and serve piping hot.

    """
    msg_body = f"""
    {recipe}
    """
    messages = [HumanMessage(msg_body)]
    parser_llm = llm_handle.llm.with_structured_output(ChaiPreparationIngredientsActionsFrame)
    frame = parser_llm.invoke(messages)
    print(frame)
    print('__________')
    print(frame.generate_description())


if __name__ == "__main__":
    key_err = set_open_api_key(config_file_name="keys-config.yml")
    err, llm = setup_openai_model(model_name="gpt-5")
    llm_handle.llm = llm
    run_recipe_parse_test()
