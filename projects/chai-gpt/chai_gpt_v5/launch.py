from flask import Flask, jsonify, request, render_template
from flask_cors import CORS  # optional but preferred
import os
import yaml
import re
import json

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

from bot_utils.tools import set_open_api_key, setup_openai_model

from dataclasses import dataclass

@dataclass
class LLMHandle:
    llm: BaseChatModel

app = Flask(__name__)
CORS(app)
llm_handle = LLMHandle


def build_preparation_tips(plans):
    # Step 1 - Collect required item for all the scenarios.
    items_for_scenarios = []
    regex = r"Locate\(([\w\s\d]+)\)"
    for plan in plans:
        required_items = []
        scenario_items = {
            'scenario': plan.get('description', 'N/A'),
            'items': required_items
        }
        for step in plan.get('steps', []):
            matches = re.findall(regex, step)
            if len(matches):
                required_items.extend(matches)
        items_for_scenarios.append(scenario_items)
    # Step 2 - Find items common to all scenarios.
    common_items = set(items_for_scenarios[0]['items']) if len(items_for_scenarios) else set()
    for scenario_items in items_for_scenarios:
        common_items = common_items.intersection(set(scenario_items['items']))
    # Step 3 - Find items unique to each scenarios.
    # -- Edit the items directly so as to have only the unique entries
    for scenario_items in items_for_scenarios:
        items = scenario_items['items']
        unique_items = list(set(items).difference(common_items))
        items.clear()
        items.extend(unique_items)
    # Step 4 - Craft response
    response = {
        'common_items': list(common_items),
        'scenario_specific_items': items_for_scenarios
    }
    
    return response


def build_system_prompt():
    system_prompt = f"""
    You are chai-gpt. An expert in chai making.

    The user is planning to prepare chai. A list of things that he/she may need will be provided.
    The items therein will fall into two main groups. They are as follows.
    - common items which are useful in almost any scenarios.
    - Items specific to certain scenarios that the user may not have thought about.

    Note that there are multiple scenarios that the user has planned for.

    This data is in JSON form. Express that data in natural language with the following structure.

    Required items -
    ...
    ...

    Additonal items you may need -
    ... scenario specific items - An explanation of where and when it may be required depending on the scenarios it is included in.

    The scenario descriptions may be quite specific. I want you to describe them in a more general tone.
    """
    return SystemMessage(system_prompt)


def build_user_message(chai_type, scene, prep_details):
    user_message = f"""
    Goal - To prepare {chai_type} at {scene}

    The JSON list is as follows -

    {json.dumps(prep_details, indent=2)}
    """
    return HumanMessage(user_message)


def load_yaml(file_name):
    try:
        recipes_path = os.path.join(os.path.dirname(__file__), file_name)
        with open(recipes_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise Exception(f'Failed to load recipes: {str(e)}')


@app.route("/recipe", methods=["GET"])
def recipe():
    chai_type = request.args.get("type", type=str)
    if not chai_type:
        return jsonify({"error": "Missing required query parameter: type"}), 400

    data = load_yaml(file_name="recipes.yaml")
    recipes = data.get("chai_recipes", {})
    recipe_obj = recipes.get(chai_type) 
    if recipe_obj is None:
        return jsonify({"error": f"Recipe '{chai_type}' not found"}), 404

    # Ensure consistent JSON shape; steps are sourced from recipes.yaml as required
    payload = {
        "name": chai_type,
        "ingr": recipe_obj.get("ingr", []),
        "steps": recipe_obj.get("steps", []),
    }
    return jsonify(payload), 200


@app.route("/preparation", methods=["GET"])
def prepare():
    chai_type = request.args.get("type", type=str)
    prep_scene = request.args.get("scene", type=str)
    if not chai_type or not prep_scene:
        return jsonify({"error": "Missing required query parameter"}), 400
    
    valid_scenes = ['home', 'campsite']
    if prep_scene not in valid_scenes:
        return jsonify({"error": f"Scene must be one of {' or '.join(valid_scenes)}"}), 400
    
    data = load_yaml(file_name="chai_prep_grounded_plans.yml")
    plans = data.get(chai_type, {})

    relevant_plans = [p for _, p in plans.items() if p['scene'] == prep_scene]

    prep_tips = build_preparation_tips(relevant_plans)
    chat_messages = [
                        build_system_prompt(), 
                        build_user_message(chai_type, prep_scene, prep_tips)
                    ]
    llm_response = llm_handle.llm.invoke(chat_messages, 
                                         reasoning={"effort": "minimal"},
                                         text={"format": {"type": "text"}, "verbosity": "medium"})
    return jsonify(llm_response.content, 200)


@app.route("/")
def show_app():
    return render_template("index.html")


if __name__ == "__main__":
    key_err = set_open_api_key(config_file_name="keys-config.yml")
    err, llm = setup_openai_model(model_name="gpt-5")
    llm_handle.llm = llm
    port = int(os.getenv("PORT", 5050))
    app.run(host="127.0.0.1", port=port, debug=True)
