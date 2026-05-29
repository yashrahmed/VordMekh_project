# The ВордМех [VordMekh] Project hosts multple experiments.

## Chai GPT.
- Experiments on getting chatbots to steer conversation in order to fill in missing information at a desired level of detail.
- 8 experiments were conducted trying out variants and some new ideas.

### Experimental details
- ChaiGPT v1: Prompt-only state-machine chat that gathers recipe, servings, inventory, context, equipment, and power source.
- ChaiGPT v2: SOAR-style production rules that track intent, inventory, missing items, heat sources, and scaled ingredients.
- ChaiGPT v3: Typed state extraction with structured outputs, explicit validation, and separate response-generation steps.
- ChaiGPT v4: "Talking form" UI where the LLM reads form state and emits `CMD_SET` commands to manipulate fields.
- ChaiGPT v5: Grounded planning with hand-authored scenario plans to surface common and scene-specific prep items.
- ChaiGPT v6: Frame-based scenario reasoning that derives preparation advice from recipe, scene, and equipment frames.
- ChaiGPT v7: Tool-search edition that parses recipe text into action frames and generates prep-tool/scene descriptions.
- ChaiGPT Search: Equipment search that expands cooking queries, extracts structured actions, and ranks matching tools.

### Findings
- LLMs (as of 2026) are bad at figuring out the level of detail. This needs to be explicitly modeling or specified via prompts.
- A chat window is a poor interface for tasks that require a human to query about or manipulate multiple variables. It is far better to have them work on a form.
- Based on the above, the idea of talking forms came to me. The LLM could (but wouldn't unless asked to) manipulate form elements. I.e. the LLM, which would be aware of the form state and user's interactions (and its own as well) with the form and a KB, would **assist** the user in filling out the form instead of directly maintaing all the details in its context. This requires the form state to be injected in the context along with the interaction events (Obviously).
- The idea of talking forms took a while to settle in my mind (This section was added a long time after the project was completed). At the time, I wasn't satisfied. I wanted to reach a conclusion. I wanted a concrete answer to the question "What WAS ChaiGPT at the end of it all?". This lead to the last experiment which was a simple search engine based on semantic parsing.

## Grasp embeddings for 3d and 2d shapes — link with graph embeddings.

### Idea - Use hand based grasps and different to charecterize parts of MNIST shapes
- Imagine that you hold the part of an alphabet or a numeral with your hand. Based on the position of your hand and fingers AND the position on the alphabet there are a few configurations to grasp that part. These configurations are topologically linked to each other and can be used as embeddings.
- The above approach has some resemblance to shape similarities.
- **In THEORY**, this approach must work in noisy environments, in presence of multiple shapes and would allow switching between scales.
