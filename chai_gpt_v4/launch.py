from flask import Flask, request, jsonify
from flask_cors import CORS
import yaml
import os
import random
from langchain_core.messages import HumanMessage, AIMessage

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

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        messages = data.get('messages')
        form_state = data.get('form_state', {})
        
        if messages is None:
            return jsonify({'error': 'Missing required field: messages'}), 400
        
        if not isinstance(messages, list):
            return jsonify({'error': 'messages must be a list'}), 400
        
        # Convert messages to Langchain objects
        langchain_messages = []
        for i, message in enumerate(messages):
            if not isinstance(message, str):
                return jsonify({'error': f'Message at index {i} must be a string'}), 400
            
            # First message is AIMessage, then alternate (AI, Human, AI, Human, ...)
            if i % 2 == 0:
                langchain_messages.append(AIMessage(content=message))
            else:
                langchain_messages.append(HumanMessage(content=message))
        
        # Print the langchain messages and form state to the log
        print("Langchain Messages:")
        for i, msg in enumerate(langchain_messages):
            print(f"  {i}: {type(msg).__name__}: {msg.content}")
        
        print("Form State:")
        print(f"  Selected Recipe: {form_state.get('selected_chai_recipe', 'Not selected')}")
        print(f"  Servings: {form_state.get('num_servings', 'Not specified')}")
        print(f"  Heating Equipment: {form_state.get('heating_equipment', 'Not selected')}")
        
        # Generate a randomized AI response
        random_responses = [
            "I understand! Let me help you with that chai recipe question.",
            "That's a great question about chai preparation. Here's what I'd suggest...",
            "Based on your message, I can provide some guidance on chai making.",
            "Interesting! For chai recipes, I always recommend starting with quality ingredients.",
            "I see you're asking about chai. Let me share some helpful tips.",
            "That's a wonderful question! Chai making is both an art and a science.",
            "Great to chat with you! For the best chai experience, consider these points...",
            "I'm here to help with your chai questions! Here's my take on that..."
        ]
        
        ai_response = AIMessage(content=random.choice(random_responses))
        
        # Print the AI response to the log
        print(f"AI Response: {ai_response.content}")
        
        return jsonify({
            'response': ai_response.content,
            'message_count': len(langchain_messages),
            'form_state_received': form_state,
            'status': 'success'
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)