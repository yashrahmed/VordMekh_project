
from bot_utils.tools import set_open_api_key, setup_openai_model

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END

from typing import Literal
from typing_extensions import TypedDict, Annotated
import operator

llm_handle = None

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

def ask_for_chai_type_node(state: MessagesState):
    ai_msg_str = "What chai would you like to prepare today?"
    user_input_msg = input(ai_msg_str)
    return {
      "messages": [
        AIMessage(content=ai_msg_str),
        HumanMessage(user_input_msg)
    ]
    } 

def get_recipe_step_node(state: MessagesState):
    """Find the chai recipe"""
    messages = state["messages"] + [HumanMessage(content="Look up the recipe for this chai")]
    response = llm_handle.invoke(messages)    
    return {
        "messages": [response]
    }

def prepare_and_run_flow():
    agent_builder = StateGraph(MessagesState)
    # Add nodes
    agent_builder.add_node("get-chai-type", ask_for_chai_type_node)
    agent_builder.add_node("get-recipe-node", get_recipe_step_node)

    # Add edges to connect nodes
    agent_builder.add_edge(START, "get-chai-type")
    agent_builder.add_edge("get-chai-type", "get-recipe-node")
    agent_builder.add_edge("get-recipe-node", END)

    # Compile the agent
    agent = agent_builder.compile()
    return agent.invoke({
        "messages": []
    })


def main():
    set_open_api_key()
    global llm_handle
    _, llm_handle = setup_openai_model("gpt-5-nano")
    messages = prepare_and_run_flow()
    for m in messages["messages"]:
        m.pretty_print()

if __name__ == '__main__':
    main()