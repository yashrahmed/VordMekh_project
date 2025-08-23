from flask import Flask, request, jsonify
from flask_cors import CORS
import yaml
import os
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

from bot_utils.tools import set_open_api_key, setup_openai_model

from dataclasses import dataclass
from rich.prompt import Prompt
from rich.text import Text
from rich.console import Console

@dataclass
class LLMHandle:
    llm: BaseChatModel


app = Flask(__name__)
CORS(app)
llm_handle = LLMHandle

def create_system_prompt_for_chai_customization():
    system_prompt = f"""
    You are ChaiGPT, a helpful LLM assistant and an expert chai chef.
    You will be given a recipe, the number of servings and the heating equiment that is being used to prepare chai.
    
    Your task is to provide a recipe that is tailored to the number of servings and the available heating equipment.
    
    Use your expertise to adjust the recipe.

    Remember to craft your response in such a way that it is addressed directly to the user as opposed to the program relaying it to you. The reply must be such that the user does not know that his inputs are being relayed.
    """
    return SystemMessage(system_prompt)


def create_system_prompt_for_chatbot():
    system_prompt = f"""
    You are ChaiGPT, a helpful LLM assistant and an expert chai chef.
    You are located on an html form where user's can search for customized recipes by helping them fill out the form correctly.

    
    The form has the following form fields - 
    1. SelectedRecipe - A recipe that the user has selected from the dropdown. Supported options are - 
        - Not selected
        - Masala Chai
        - Adrak Chai
        - Sulaimani Chai
        - Kashmiri Chai
        - Kahwah
    2. Number of Servings - A dropdown with options of int values between 1 and 6 and "Not selected".
    3. Selected heating equipment - A dropdown of values that describes the heating equipment and the power source available to the user. Supported options are -
        - Not selected
        - Electric stove (Induction or Coil heating)
        - Propane stove (w Propane Tank)
        - Butane stove (w Butane Tank)

    You will be relayed the user's message along with the current form state in each of his messages.
    Your goals are the following - 
    1. Help users by answering questions that are relevant to chai making including general knowledge questions about chai making, ingredients, health effects, buying ingredients, historical context etc.
    2. Do not help users with tasks that are completely unrelated to making chai. Decline their requests politely if the conversation strays too far from the task of making chai.
    3. Help user set the form state. If user asks to set a form state for one of the above fields then return a command with your response. A command sentence must describe ALL form values even if user asks to set a specific one. e.g if the current form state is the following -
    SelectedRecipe=Adrak Chai | Number of Servings=Not Selected | Selected heating equipment=Electric stove (Induction or Coil heating)

    and the user's message is to select Masala chai then your output must be as follows -

    [...A helpful message stating that you understand the request...]
    CMD_SET SelectedRecipe=Masala Chai | Number of Servings=Not Selected | Selected heating equipment=Electric stove (Induction or Coil heating)

    The UI will use this structure to separate your reply from the command to execute. Remember that the command will ALWAYS be at the end of your response.
    If the user's request is not supported by the form do not use the command phrase and relay to the user that the operation is unsupported.

    4. Remember to craft your response in such a way that it is addressed directly to the user as opposed to the program relaying it to you. The reply must be such that the user does not know that his inputs are being relayed.

    5. Given that purpose of the form is to provide customized recipes, do not provide recipes until explictly asked for it. And if you do provide it, inform the user that the system will generate the customized recipe and that your recipe is an educated guess.
    """
    return SystemMessage(system_prompt)


def display_exit_message(console: Console, message: str):
    text = Text()
    text.append(message, style="bold red")
    console.print(text)


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
            return jsonify({
                'error': f'Recipe "{selected_chai_recipe}" not found',
            }), 404
        
        recipe = chai_recipes[selected_chai_recipe]

        customization_req_message_str = f"""
            #Recipe -
                {selected_chai_recipe}

            #Number of Servings - {num_servings}

            #Heating Equipment - {heating_equipment}

            Help the user customize the recipe.
        """
        messages = [create_system_prompt_for_chai_customization(), HumanMessage(customization_req_message_str)]

        response = llm_handle.llm.invoke(messages)
        
        response_data = {
            'response': response.content,
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
        
        if messages is None:
            return jsonify({'error': 'Missing required field: messages'}), 400
        
        if not isinstance(messages, list):
            return jsonify({'error': 'messages must be a list'}), 400
        
        # Convert messages to Langchain objects
        langchain_messages = [create_system_prompt_for_chatbot()]
        for i, message in enumerate(messages):
            if not isinstance(message, str):
                return jsonify({'error': f'Message at index {i} must be a string'}), 400
            
            # First message is AIMessage, then alternate (AI, Human, AI, Human, ...)
            if i % 2 == 0:
                langchain_messages.append(AIMessage(content=message))
            else:
                langchain_messages.append(HumanMessage(content=message))
        
        ai_response = llm_handle.llm.invoke(langchain_messages)
        
        # Print the AI response to the log
        # print(f"AI Response: {ai_response.content}")
        return jsonify({
            'response': ai_response.content,
            'message_count': len(langchain_messages),
            'status': 'success'
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


def main():
    console = Console()
    err = set_open_api_key(config_file_name="keys-config.yml")
    if err:
        display_exit_message(console, str(err))
        return
    err, llm = setup_openai_model()
    llm_handle.llm = llm
    if err:
        display_exit_message(console, str(err))
        return
    app.run(debug=True)


if __name__ == '__main__':
    main()