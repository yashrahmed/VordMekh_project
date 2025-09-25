from flask import Flask, jsonify, request
from flask_cors import CORS  # optional but preferred
import os
import yaml
import re

app = Flask(__name__)
CORS(app)

def load_yaml(file_name):
    try:
        recipes_path = os.path.join(os.path.dirname(__file__), file_name)
        with open(recipes_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise Exception(f'Failed to load recipes: {str(e)}')


def build_preparation_tips(chai_type, plans):
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

    return jsonify(build_preparation_tips(chai_type, relevant_plans), 200)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=True)
