from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
import re
from .frames import (
    ChaiPreparationIngredientsActionsFrame as CPIAF,
    ChaiPrepToolingFrame as CPF,
    SCENE_FRAMES_VARIANTS,
    ChaiRecipe,
    generate_chai_tooling
)

def get_recipe_step(llm: BaseChatModel, user_request: str) -> ChaiRecipe:
    "User asks for recipe for a certain type of chai (single serving) and goes back and forth with the customization."
    messages= [AIMessage(f"""As an expert chai chef, help me find a recipe that is customized to my needs. 
        - The recipe must be structured as follows - 
            Ingredients:
                1.
                2. .... ingredients and their quantities.
            Stpes:
                1. 
                2. .... preparation steps
        
        - Output only the recipe.
        - No follow up questions.
        - Do NOT respond if the user's request has nothing to do with a chai recipe. 
    Here is my request -
    {user_request}
    """)] 
    messages.append(HumanMessage(user_request))
    parser_llm = llm.with_structured_output(ChaiRecipe)
    return parser_llm.invoke(messages)

def parse_recipe_step(llm: BaseChatModel, recipe_text: str) -> CPIAF:
    "Using the recipe, the LLM is prompted to extract ingredients, their quantities and actions like crushing, grating, stirring, aerating etc."
    msg_body = f"""{recipe_text}"""
    messages = [HumanMessage(msg_body)]
    parser_llm = llm.with_structured_output(CPIAF)
    return parser_llm.invoke(messages)

def infer_chai_prep_tools_step(ingredient_prep_frame: CPIAF) -> CPF:
    "Lookup the chai prep tools frame table using the above step's output as the input."
    return generate_chai_tooling(ingredient_prep_frame)

def generate_full_scene_descriptor_step(scene_type: str, ingredient_prep_frame: CPIAF, tooling_prep_frame: CPF):
    "Combine the results from the previous steps with the cooking scenario descriptions like in V6."
    equipment_frames = SCENE_FRAMES_VARIANTS.get_scenes(scene_type)
    equipment_frames_descriptions = '____________________\n'.join([scene.generate_description() for scene in equipment_frames])
    ingredient_frame_description = ingredient_prep_frame.generate_description()
    tooling_frame_description = tooling_prep_frame.generate_description()
    full_description = f"""
        Here are the details about preparing chai at {scene_type}.

        Ingredients and actions -
        
        {ingredient_frame_description}

        Chai preparation equipment - 

        {tooling_frame_description}

        Preparation contexts and the required equipment -
        
        {equipment_frames_descriptions}
    """
    full_description = '\n'.join([re.sub(r'^[\s^\n]+', '' ,line) for line in full_description.split('\n')])
    return full_description

def generate_full_nl_description_step(llm: BaseChatModel, scene_and_tooling_description: str) -> str:
    "Have the LLM generate a tool/equipment description."
    user_prompt = f"""
        You will be given a description of a chai recipe and the details about the equipment and scenarios involved.

        The description document will have four sections - 
        1. Ingredients.
        2. Preparation actions.
        3. Chai preparation equipment
        4. Preparation contexts and the required equipment.

        #4 describes the different pieces of equipment when making chai under different circumstances.

        You task is to generate the following "required tools" document. The document must have the following structure.

        1. Ingredients - Contains details about ingredients and their quantities.
        2. Required Equipment.

        The "Required Equipment" section must be formatted like the following.
        1. [Tool 1]
            - Why it is needed (Reason #1)
            - Why it is needed (Reason #2)
            ......
        2. ....
        3. ....

        When generating the "Required Equipment" section, dedup the tools from all different preparation contexts with details about different usage contexts
        listed under that tool's entry.

        The document must be written in natural language and must NOT sound formulaic.

        The description document is given below -
        {scene_and_tooling_description}
    """
    user_message = [HumanMessage(user_prompt)]
    return llm.invoke(user_message).content

