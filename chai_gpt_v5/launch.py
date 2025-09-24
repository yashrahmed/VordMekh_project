from flask import Flask, jsonify, request
from flask_cors import CORS  # optional but preferred
import os
import yaml

app = Flask(__name__)
CORS(app)

def load_recipes():
    try:
        recipes_path = os.path.join(os.path.dirname(__file__), 'recipes.yaml')
        with open(recipes_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise Exception(f'Failed to load recipes: {str(e)}')


def _lookup_recipe(chai_type: str):
    """Return the recipe dict for a chai type from v5, falling back to v4."""
    data = load_recipes()
    recipes = data.get("chai_recipes", {})
    if chai_type in recipes:
        return recipes.get(chai_type) or {}
    return None


@app.after_request
def _add_cors_headers(resp):
    # Allow local file:// or other origins to call this endpoint during dev
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Accept, Authorization"
    return resp


@app.route("/recipe", methods=["GET", "OPTIONS"])
def recipe():
    if request.method == "OPTIONS":
        # Preflight support
        response = app.make_response(("", 204))
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Accept"
        return response

    chai_type = request.args.get("type", type=str)
    if not chai_type:
        return jsonify({"error": "Missing required query parameter: type"}), 400

    recipe_obj = _lookup_recipe(chai_type)
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
