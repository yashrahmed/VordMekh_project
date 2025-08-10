# ChaiGPT V1

## Description

ChaiGPT V1 is a conversational assistant that helps you prepare a variety of traditional chai recipes. It operates as a state-machine, guiding you through a checklist of questions to gather all the necessary information—such as the desired recipe, number of servings, and your available equipment—before generating a customized recipe.

This version ensures all required ingredients and tools are confirmed before providing step-by-step preparation instructions.

## Setup

Before launching the assistant, you need to provide your Google API key.

1.  In the root directory of the `VordMekh_project`, create a file named `keys-config.yml`.
2.  Add your Google API key to this file in the following format:

    ```yaml
    google_api_key: "YOUR_API_KEY_HERE"
    ```

## Launch Instructions

To run ChaiGPT V1, execute the following command from the root directory of the `VordMekh_project`:

```bash
python3 -m chai_gpt.launch
```
