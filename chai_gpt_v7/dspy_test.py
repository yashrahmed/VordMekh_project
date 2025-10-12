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


def main():
    set_open_api_key()
    # call_an_llm()
    # llm_cot()
    llm_predict_test()

if __name__ == '__main__':
    main()