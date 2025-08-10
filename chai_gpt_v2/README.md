# ChaiGPT V2

## Description

ChaiGPT V2 is an advanced conversational assistant designed to help you prepare chai from a curated list of recipes. This version uses a more flexible, intent-based model to understand your needs. It tracks your desired chai, serving size, and available inventory to provide a tailored recipe.

Key features include:
-   Inferring user intent to add/remove items from an inventory.
-   Calculating required ingredients and identifying what's missing.
-   Guiding you if you don't have a proper heat source.
-   Generating adjusted preparation steps based on your specific inputs.

## Setup

Before launching the assistant, you need to provide your Google API key.

1.  In the root directory of the `VordMekh_project`, create a file named `keys-config.yml`.
2.  Add your Google API key to this file in the following format:

    ```yaml
    google_api_key: "YOUR_API_KEY_HERE"
    ```

## Launch Instructions

To run ChaiGPT V2, execute the following command from the root directory of the `VordMekh_project`:

```bash
python3 -m chai_gpt_v2.launch
```
