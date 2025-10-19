from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

from bot_utils.tools import set_open_api_key, setup_openai_model
from dataclasses import dataclass

from .workflow import parse_recipe_step, get_recipe_step

@dataclass
class LLMHandle:
    llm: BaseChatModel

llm_handle = LLMHandle

def run_recipe_parse_test():
    recipe = """
    Ingredients
        •	1 cup water
        •	½ teaspoon Kashmiri green tea leaves (or other mild green tea)
        •	2 green cardamom pods, lightly crushed
        •	1 small pinch of saffron (2–3 strands)
        •	2–3 almond slices (or 1 almond, blanched and slivered)
        •	½ inch piece of cinnamon stick
        •	1 clove (optional)
        •	½ teaspoon honey or sugar (to taste)

    ⸻

    Steps
        1.	In a small saucepan, combine water, cardamom, cinnamon, clove, and saffron.
    Bring to a gentle boil, then simmer for about 2 minutes to release the aromas.
        2.	Turn off the heat, add green tea leaves, and steep for 1–2 minutes.
    Avoid boiling after adding tea—it should stay clear and fragrant.
        3.	Strain the tea into a cup.
        4.	Add sliced almonds and sweeten with honey or sugar.
    Stir lightly and serve hot.

    """
    recipe = get_recipe_step(llm_handle.llm, "Give me the recipe for a serving of Kashmiri chai. Make it on the sweeter side and use sliced almonds!")
    print(recipe)
    print('___________')
    frame = parse_recipe_step(llm_handle.llm, recipe)

    print(frame)
    print('__________')
    print(frame.generate_description())


if __name__ == "__main__":
    from time import time
    key_err = set_open_api_key(config_file_name="keys-config.yml")
    start_ts = time()
    err, llm = setup_openai_model(model_name="gpt-5-chat-latest")
    llm_handle.llm = llm
    run_recipe_parse_test()
    end_ts = time()
    print(f"Time taken: {end_ts - start_ts:.4f} seconds")

    # run_recipe_parse_test()
