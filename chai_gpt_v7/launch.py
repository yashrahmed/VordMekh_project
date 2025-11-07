from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

from bot_utils.tools import set_open_api_key, setup_openai_model
from dataclasses import dataclass

from .frames import (
    ChaiRecipe,
    ChaiPreparationIngredientsActionsFrame as CPIAF
)

from .workflow import (
    parse_recipe_step,
    get_recipe_step,
    infer_chai_prep_tools_step,
    generate_full_scene_descriptor_step,
    generate_full_nl_description_step,
)

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import os
import re
import html

@dataclass
class LLMHandle:
    llm: BaseChatModel

llm_handle = LLMHandle

app = Flask(__name__)
CORS(app)

def _sanitize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"```[\s\S]*?```", "", t)
    t = t.replace("`", "")
    t = re.sub(
        r"(__\w+__|\bimport\b|\bfrom\b|\beval\b|\bexec\b|\bos\.|\bsys\.|\bsubprocess\b|\bshutil\b|\bopen\(|\bbase64\b|<\/?script[^>]*>)",
        "",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", t)
    t = html.escape(t)
    return t[:4000]

@app.route("/")
def show_app():
    return render_template("index.html")

@app.route("/get-recipe", methods=["POST"])
def get_recipe():
    recipe_query = None
    if request.is_json:
        body = request.get_json(silent=True) or {}
        recipe_query = body.get("query")
    if recipe_query is None:
        recipe_query = request.get_data(as_text=True)
    if not recipe_query:
        return jsonify({"error": "Missing text payload"}), 400
    recipe_query = _sanitize_text(recipe_query)

    recipe: ChaiRecipe = get_recipe_step(llm_handle.llm, recipe_query)

    if recipe.is_valid and recipe.recipe_text:
        return jsonify({"recipe": recipe.recipe_text}), 200
    if not recipe.recipe_text:
        return jsonify({"recipe": "N/A", "error": "Recipe could not be fetched"}), 400
    return jsonify({"recipe": "N/A", "error": "Your query isn't relevant to chai preparation!"}), 400


@app.route("/get-prep-tools", methods=["POST"])
def get_prep_tools():
    recipe_text = None
    if request.is_json:
        body = request.get_json(silent=True) or {}
        recipe_text = body.get("recipe_text")
    if not recipe_text:
        return jsonify({"error": "Missing text payload"}), 400
    recipe_text = _sanitize_text(recipe_text)

    recipe_frame: CPIAF = parse_recipe_step(llm_handle.llm, recipe_text)
    tooling_description: str = infer_chai_prep_tools_step(recipe_frame)
    combined_scene_description = generate_full_scene_descriptor_step(recipe_frame, tooling_description)
    nl_output = generate_full_nl_description_step(llm_handle.llm, combined_scene_description)
    return jsonify({"sanitized_text": nl_output}), 200


@app.route("/hello", methods=["GET"])
def hello():
    return "hello there", 200

if __name__ == "__main__":
    key_err = set_open_api_key(config_file_name="keys-config.yml")
    err, llm = setup_openai_model(model_name="gpt-5-chat-latest")
    llm_handle.llm = llm
    port = int(os.getenv("PORT", 5051))
    app.run(host="127.0.0.1", port=port, debug=True)
