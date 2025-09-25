from flask import Flask, jsonify, request
from flask_cors import CORS  # optional but preferred
import os
import yaml

app = Flask(__name__)
CORS(app)

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


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=True)
