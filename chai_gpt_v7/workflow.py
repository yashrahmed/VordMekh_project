from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

from .frames import ChaiPreparationIngredientsActionsFrame as CPIAF, ChaiPrepToolingFrame as CPF, CookingEquipmentInASceneFrame as CESF
from typing import List

def get_recipe_step(llm: BaseChatModel, user_request: str) -> str:
    "User asks for recipe for a certain type of chai (single serving) and goes back and forth with the customization."
    return ""

def parse_recipe_step(llm: BaseChatModel, user_request: str) -> CPIAF:
    "Using the recipe, the LLM is prompted to extract ingredients, their quantities and actions like crushing, grating, stirring, aerating etc."
    return CPIAF()

def infer_chai_prep_tools_step(ingredient_prep_frame: CPIAF) -> CPF:
    "Lookup the chai prep tools frame table using the above step's output as the input."
    return CPF()

def generate_full_scene_descriptor_step(ingredient_prep_frame: CPIAF, tooling_prep_frame: CPF, scenario_frames: List[CESF]):
    "Combine the results from the previous steps with the cooking scenario descriptions like in V6."
    pass

def generate_full_nl_description_step(llm: BaseChatModel, scene_and_tooling_description: str) -> str:
    "Have the LLM generate a tool/equipment description."
    return ""

