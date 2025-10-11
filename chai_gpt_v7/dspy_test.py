from bot_utils.tools import set_open_api_key
import dspy



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


def main():
    set_open_api_key()
    # call_an_llm()
    llm_cot()

if __name__ == '__main__':
    main()