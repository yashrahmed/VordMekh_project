from flask import Flask, request, jsonify
from flask_cors import CORS
import yaml
import os

app = Flask(__name__)
CORS(app)

def load_recipes():
    try:
        recipes_path = os.path.join(os.path.dirname(__file__), 'recipes.yaml')
        with open(recipes_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise Exception(f'Failed to load recipes: {str(e)}')

@app.route('/get-recipe', methods=['POST'])
def get_recipe():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        num_servings = data.get('num_servings')
        selected_chai_recipe = data.get('selected_chai_recipe')
        heating_equipment = data.get('heating_equipment')
        
        if num_servings is None or selected_chai_recipe is None or heating_equipment is None:
            return jsonify({'error': 'Missing required fields: num_servings, selected_chai_recipe, heating_equipment'}), 400
        
        if not isinstance(num_servings, int):
            return jsonify({'error': 'num_servings must be an integer'}), 400
        
        if not isinstance(selected_chai_recipe, str):
            return jsonify({'error': 'selected_chai_recipe must be a string'}), 400
        
        if not isinstance(heating_equipment, str):
            return jsonify({'error': 'heating_equipment must be a string'}), 400
        
        recipes_data = load_recipes()
        chai_recipes = recipes_data.get('chai_recipes', {})
        
        if selected_chai_recipe not in chai_recipes:
            available_recipes = list(chai_recipes.keys())
            return jsonify({
                'error': f'Recipe "{selected_chai_recipe}" not found',
                'available_recipes': available_recipes
            }), 404
        
        recipe = chai_recipes[selected_chai_recipe]
        
        response_data = {
            'num_servings': num_servings,
            'selected_chai_recipe': selected_chai_recipe,
            'heating_equipment': heating_equipment,
            'recipe': {
                'ingredients': recipe.get('ingr', []),
                'tools': recipe.get('tools', []),
                'steps': recipe.get('steps', [])
            },
            'status': 'success'
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)