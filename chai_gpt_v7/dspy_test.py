from bot_utils.tools import set_open_api_key
import dspy
import json
from .frames import ChaiPreparationIngredientsActionsFrame


def call_an_llm():
    llm = dspy.LM("gpt-5-nano", temperature=1.0, max_tokens=16000)
    dspy.configure(lm=llm)
    messages = [
        {
            "role": "user",
            "content": "Hi there! Who are you?"
        }
    ]
    response = llm(messages=messages)
    print(response[0])

def llm_cot():
    llm = dspy.LM("gpt-5-nano", temperature=1.0, max_tokens=16000)
    dspy.configure(lm=llm)
    math_bot = dspy.ChainOfThought("question -> answer: float")
    response = math_bot(question="Two dice are tossed. What is the probability that the sum equals two?")
    print(response)

def llm_predict_test():
    llm = dspy.LM("gpt-5-nano", temperature=1.0, max_tokens=16000)
    dspy.configure(lm=llm)
    # Prompt generated within the chat adapter.
    # There is no restriction on the values used in the text signature as long as the LLM can make sense of it after the chat message is rendered.
    # Inputs are to be supplied with the same kwarg in the callable see 'car' below.
    # Outputs can be anything as long as the LLM can make sense of it i.e using car -> xyz won't work since in the actual chat message, the llm cannot tell what xyz is.
    predict = dspy.Predict("car -> length_in_meters:float")
    response = predict(car="Lamborghini Murciélago")
    print('___________________')
    print(response)
    print(dspy.inspect_history())

def llm_chai_recipe_parse_test():

    recipe_text = """
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

    SCHEMA_HINT = (
    "Return ONLY valid JSON for ChaiPreparationIngredientsActionsFrame. "
    "Do not add prose or extra keys. If unsure, use null. "
    "Never infer grinding for pre-ground spices.\n\n"
    + json.dumps(ChaiPreparationIngredientsActionsFrame.model_json_schema(), indent=2)
    )

    class RecipeParser(dspy.Signature):
        """Parse a recipe into ingredients + action flags. Output must be strict JSON."""
        recipe_text: str = dspy.InputField(desc="Full recipe text: ingredients + steps.")
        format_instructions: str = dspy.InputField(desc="JSON schema and formatting rules.")
        parsed_recipe_json: str = dspy.OutputField(desc="Strict JSON for ChaiPreparationIngredientsActionsFrame.")
        

    llm = dspy.LM("gpt-5", temperature=1.0, max_tokens=16000)
    dspy.configure(lm=llm)
    predict = dspy.Predict(RecipeParser)
    raw = predict(recipe_text=recipe_text, format_instructions=SCHEMA_HINT, reasoning={ "effort": "low" })
    parsed = ChaiPreparationIngredientsActionsFrame.model_validate_json(raw.parsed_recipe_json)
    print(parsed.generate_description())




def main():
    set_open_api_key()
    # call_an_llm()
    # llm_cot()
    # llm_predict_test()
    llm_chai_recipe_parse_test()

if __name__ == '__main__':
    main()