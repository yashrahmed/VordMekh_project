from dataclasses import dataclass
from typing import Optional

from flask import Flask, jsonify, request
from flask_cors import CORS
from langchain_core.language_models.chat_models import BaseChatModel

from bot_utils.tools import set_open_api_key, setup_openai_model
from chai_gpt_search.db_search import load_db, search_db_given_actions
from chai_gpt_search.workflows import answer_then_parse_actions_flow


@dataclass
class LLMHandle:
    llm: Optional[BaseChatModel] = None

llm_handle = LLMHandle()
items_db = []
app = Flask(__name__)
CORS(app)


def search_equipment(user_prompt):
    if not user_prompt:
        raise ValueError("User prompt cannot be empty.")
    if llm_handle.llm is None:
        raise RuntimeError("LLM not initialized. Call setup_openai_model first.")

    ans, cooking_actions = answer_then_parse_actions_flow(llm_handle.llm, user_prompt)
    search_result = search_db_given_actions(items_db, cooking_actions)
    return ans, search_result


@app.route("/search", methods=["POST"])
def search_endpoint():
    """REST endpoint that surfaces the kitchen equipment search workflow."""
    payload = request.get_json(silent=True) or {}
    user_prompt = payload.get("query")

    if not user_prompt:
        return jsonify({"error": "Missing 'query' parameter or body field."}), 400

    try:
        answer, matches = search_equipment(user_prompt)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Search failed: {exc}"}), 500

    response_payload = [
        {
            "key": item.key,
            "name": item.name,
            "description": item.description,
            "llmAnswer": answer.content
            # "actions": item.actions,
        }
        for item in matches
    ]
    return jsonify({"results": response_payload, "count": len(response_payload)}), 200


def launch():
    _ = set_open_api_key(config_file_name="keys-config.yml")
    _, llm = setup_openai_model(model_name="gpt-5-chat-latest")
    items_db[:] = load_db()
    llm_handle.llm = llm
    app.run(host="127.0.0.1", port=4050, debug=True)


if __name__ == '__main__':
    launch()
