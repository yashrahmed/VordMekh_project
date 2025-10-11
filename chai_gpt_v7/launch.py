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
    Here’s one version of a single-serving recipe for Yemeni / Adeni chai (often called “Adeni Chai” or “Chai Adani”) based on sources:

    ⸻

    Ingredients (for 1 cup / serving)
        •	¾ cup milk (or combination of water + milk)  ￼
        •	¾ cup water  ￼
        •	1 teaspoon loose black tea  ￼
        •	1 rolled cinnamon stick (broken)  ￼
        •	3 green cardamom pods (slightly crushed / peeled)  ￼
        •	5 cloves  ￼
        •	¼ teaspoon ground ginger (optional)  ￼
        •	Sugar (to taste)  ￼

    ⸻

    Steps
        1.	In a small saucepan, combine milk + water, along with cinnamon, cardamom pods, cloves, ground ginger. Cover and heat over medium flame.  ￼
        2.	When the mixture is nearly boiling, add sugar and tea leaves. Stir, watching carefully to avoid boil-over.  ￼
        3.	Let it come to a brief boil for about a minute.  ￼
        4.	Turn off the heat. Strain into a cup, removing spices and leaves.  ￼

    ⸻

    """
    msg_body = f"""Extract all the relevant details about ingredients and the actions that I need to take from the given chai recipe.
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
