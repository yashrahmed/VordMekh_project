from pathlib import Path
from typing import Any

from langchain.messages import HumanMessage

import yaml
from itertools import chain

from bot_utils.tools import set_open_api_key, setup_openai_model  # type: ignore  # noqa: E402
from .frames import (  # type: ignore  # noqa: E402
    ChaiPreparationIngredientsActionsFrame,
    ChaiRecipe,
)
from .workflow import (  # type: ignore  # noqa: E402
    generate_full_nl_description_step,
    generate_full_scene_descriptor_step,
    get_recipe_step,
    infer_chai_prep_tools_step,
    parse_recipe_step,
)

SEPARATOR = '\n\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n'


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


def write_text_file(path: str | Path, contents: str) -> None:
    """Persist plain-text content to a file, creating parents if required."""
    file_path = Path(path)
    if file_path.parent and not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        handle.write(contents)


def load_llm(model_name: str, config_path = 'keys-config.yml'):
    key_err = set_open_api_key(config_file_name=config_path)
    if key_err:
        raise RuntimeError(key_err)
    err, llm = setup_openai_model(model_name=model_name, temperature=1.0)
    if err:
        raise RuntimeError(err)
    return llm

def main():
    llm = load_llm(model_name="gpt-5-chat-latest")
    serving_situation_options = [
        'Preparing chai for a large group of people i.e > 10.',
        'Preparing chai for a small number of people i.e. 1-4'
    ]
    chai_customizations = load_yaml_dict("chai_gpt_v7/recipe-exploration/customization-1.yml")
    recipe_responses = []
    item_num = 1
    total = len(list(chain.from_iterable(chai_customizations.values()))) * 2
    limit = -1 #-1 indicates no limit; Used for testing when I do not want to run the llm for all customizations.
    for chai_name, customizations in chai_customizations.items():
        if limit > -1 and item_num > limit: break
        for cust_item in customizations:
            if limit > -1 and item_num > limit: break
            for serving_situation in serving_situation_options:
                if limit > -1 and item_num > limit: break
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
                recipe_response = f"""Recipe for {chai_name}\n\n{output.content}"""
                recipe_responses.append(recipe_response)
                print(f"{item_num} / {total} done!")
                item_num += 1
                    
    write_text_file("chai_gpt_v7/recipe-exploration/recipe-responses.txt", SEPARATOR.join(recipe_responses))
                    


if __name__ == "__main__":
    raise SystemExit(main())
