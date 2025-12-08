from pathlib import Path
from typing import List

import yaml

from chai_gpt_search.models import Item

def load_db(file_path = "./chai_gpt_search/resources/items.yml"):
    # Function loads the yml data from a file and returns a list of Item objects
    file_path_obj = Path(file_path)

    if not file_path_obj.exists():
        raise FileNotFoundError(f"Items database not found: {file_path_obj}")

    with file_path_obj.open("r", encoding="utf-8") as fh:
        raw_items = yaml.safe_load(fh) or []

    if not isinstance(raw_items, list):
        raise ValueError("Items database must be a list of item definitions.")

    return [Item(**item_data) for item_data in raw_items]
