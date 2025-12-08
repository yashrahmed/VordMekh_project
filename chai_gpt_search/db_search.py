from pathlib import Path
from typing import List

import yaml

from chai_gpt_search.models import Item, CookingActions

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

def search_db_given_actions(db_items: List[Item], actions: CookingActions):
    # Given cooking actions using only the flags that are true, find the items that match in the db_items list.
    # Items must be ranked by the number of matches with the items with max # of matching actions at the top.
    active_actions = {name for name, value in actions.model_dump().items() if value}
    if not active_actions:
        return []

    scored_items = []
    for item in db_items:
        matches = len(active_actions.intersection(item.actions))
        if matches > 0:
            scored_items.append((matches, item))

    scored_items.sort(key=lambda entry: entry[0], reverse=True)
    return [item for _, item in scored_items]
