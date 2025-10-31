from pathlib import Path
from typing import Any

from langchain.messages import HumanMessage

import yaml

from bot_utils.tools import set_open_api_key, setup_openai_model  # type: ignore  # noqa: E402
from .frames import (  # type: ignore  # noqa: E402
    ChaiPreparationIngredientsActionsFrame,
    ChaiPrepToolingFrame,
    ChaiRecipe,
)
from .workflow import (  # type: ignore  # noqa: E402
    generate_full_nl_description_step,
    generate_full_scene_descriptor_step,
    get_recipe_step,
    infer_chai_prep_tools_step,
    parse_recipe_step,
)



def load_yaml_dict(path: str | Path) -> dict[str, Any]:
    """Load a YAML file, ensuring the top-level document is a mapping."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"YAML file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as yaml_file:
        data = yaml.safe_load(yaml_file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {file_path}, got {type(data).__name__}")
    return data


def load_llm(model_name: str, config_path = 'keys-config.yml'):
    key_err = set_open_api_key(config_file_name=config_path)
    if key_err:
        raise RuntimeError(key_err)
    err, llm = setup_openai_model(model_name=model_name)
    if err:
        raise RuntimeError(err)
    return llm

def main():
    llm = load_llm(model_name="gpt-5-chat-latest")
    serving_situation_options = [
        'Preparing chai for a large group of people i.e > 10.',
        'Preparing chai for a small number of people i.e. 1-4'
    ]
    recipes = load_yaml_dict("chai_gpt_v7/recipe-exploration/customization-1.yml")
    for chai_name, customizations in recipes.items():
        for cust_item in customizations:
            for serving_situation in serving_situation_options:
                recipe_query = f"""
                    I wish to prepare {chai_name}.
                    I wish to customize it like so -
                    {cust_item}
                    Here is the preparation context -
                    {serving_situation}

                    Give me the recipe for prearing the above.
                    Take into account the following -
                    1. I do not have much help handling heavy tools/utensils.
                    2. I am preparing this solo.

                    - The recipe must be structured as follows - 
                        Ingredients:
                            1.
                            2. .... ingredients and their quantities.
                        Preparation steps:
                            1. 
                            2. .... preparation steps
                    
                    - Output only the recipe.
                    - No follow up questions.
                    - Do NOT respond if the user's request has nothing to do with a chai recipe. 
                """
                messages = [HumanMessage(recipe_query)]
                output = llm.invoke(messages)
                print('+++++++++++++++++++++++')
                print(output.content)
                return
    


if __name__ == "__main__":
    raise SystemExit(main())
