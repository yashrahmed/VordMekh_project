from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

from bot_utils.tools import set_open_api_key, setup_openai_model
from dataclasses import dataclass

from .workflow import (
    parse_recipe_step,
    get_recipe_step,
    infer_chai_prep_tools_step,
    generate_full_scene_descriptor_step,
    generate_full_nl_description_step,
)

from flask import Flask, jsonify, request
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


@app.route("/get-recipe", methods=["POST"])
def get_recipe():
    text = None
    if request.is_json:
        body = request.get_json(silent=True) or {}
        text = body.get("query")
    if text is None:
        text = request.get_data(as_text=True)
    if not text:
        return jsonify({"error": "Missing text payload"}), 400
    sanitized = _sanitize_text(text)
    return jsonify({"recipe": sanitized}), 200


@app.route("/get-prep-tools", methods=["POST"])
def get_prep_tools():
    scene_type = None
    text = None
    if request.is_json:
        body = request.get_json(silent=True) or {}
        scene_type = body.get("scene_type")
        text = body.get("recipe")
    if not scene_type:
        return jsonify({"error": "Missing scene_type"}), 400
    valid_scenes = ["home", "campsite"]
    if scene_type not in valid_scenes:
        return jsonify({"error": f"scene_type must be one of {' or '.join(valid_scenes)}"}), 400
    if not text:
        return jsonify({"error": "Missing text payload"}), 400
    sanitized = _sanitize_text(text)
    return jsonify({"scene_type": scene_type, "sanitized_text": sanitized}), 200


@app.route("/hello", methods=["GET"])
def hello():
    return "hello there", 200

def run_recipe_parse_test():
    recipe = """
    Ingredients
        •	1 cup water
        •	½ teaspoon Kashmiri green tea leaves (or other mild green tea)
        •	2 green cardamom pods, lightly crushed
        •	1 small pinch of saffron (2–3 strands)
        •	2–3 almond slices (or 1 almond, blanched and slivered)
        •	½ inch piece of cinnamon stick
        •	1 clove (optional)
        •	½ teaspoon honey or sugar (to taste)

    ⸻

    Steps
        1.	In a small saucepan, combine water, cardamom, cinnamon, clove, and saffron.
    Bring to a gentle boil, then simmer for about 2 minutes to release the aromas.
        2.	Turn off the heat, add green tea leaves, and steep for 1–2 minutes.
    Avoid boiling after adding tea—it should stay clear and fragrant.
        3.	Strain the tea into a cup.
        4.	Add sliced almonds and sweeten with honey or sugar.
    Stir lightly and serve hot.

    """
    recipe = get_recipe_step(llm_handle.llm, "Give me the recipe for a serving of Kashmiri chai. Make it on the sweeter side")
    # recipe = get_recipe_step(llm_handle.llm, "How do I change my car tyre?")

    print(recipe.is_valid)
    print('___________')
    if recipe.is_valid:
        frame = parse_recipe_step(llm_handle.llm, recipe.recipe_text)
        chai_tool_frame = infer_chai_prep_tools_step(frame)
        # print('################')
        combined_scene_description = generate_full_scene_descriptor_step("home", frame, chai_tool_frame)
        # print(combined_scene_description)
        # print('################')
        nl_output = generate_full_nl_description_step(llm_handle.llm, combined_scene_description)
        print(nl_output)

    else:
        print("Invalid request.....")


if __name__ == "__main__":
    key_err = set_open_api_key(config_file_name="keys-config.yml")
    err, llm = setup_openai_model(model_name="gpt-5-chat-latest")
    llm_handle.llm = llm
    port = int(os.getenv("PORT", 5051))
    app.run(host="127.0.0.1", port=port, debug=True)
