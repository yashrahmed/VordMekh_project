from pathlib import Path
from typing import List

import yaml

from chai_gpt_search.models import Item, CookingActions

#: Structured checklist of kitchen tasks for lookup or downstream prompts.
class CookingActionsDictionary:
    def __init__(self) -> None:
        self.actions_lookup =  {
            "peel_round_produce": "Remove the outer skin from spherical or rounded fruits and vegetables while maintaining their curvature.",
            "extract_stem_core": "Surgically remove the woody stem or hard core from a fruit or vegetable.",
            "remove_surface_blemish": "Excise small rotten spots, eyes, or imperfections from the surface of produce.",
            "sculpt_decorative_shape": "Cut vegetables into specific geometric shapes for visual presentation (e.g., tourné cut).",
            "peel_irregular_produce": "Remove skin from ingredients with uneven surfaces like ginger or Jerusalem artichokes requiring intricate navigation.",
            "slice_small_produce": "Cut small items like garlic, shallots, or strawberries into thin, consistent pieces.",
            "mince_small_aromatics": "Chop herbs, garlic, or ginger into extremely fine pieces using a rocking or pivot motion.",
            "trim_meat_fat": "Remove small, precise areas of fat, silverskin, or veins from a cut of meat.",
            "pierce_check_texture": "Insert the tip of a tool into food to assess tenderness or cooking progress.",
            "sever_binding_material": "Cut through twine, string, or netting used to truss meat.",
            "carve_cooked_protein": "Slice warm, large roasted meats into serving pieces.",
            "slice_cured_fish": "Cut very thin, translucent sheets of cold fish like smoked salmon.",
            "slice_uniform_portions": "Cut terrines, pâtés, or loaves into identical serving sizes for presentation.",
            "portion_boneless_meat": "Divide large raw muscle groups into steaks or cutlets.",
            "debone_raw_protein": "Separate raw meat from the bone structure with minimal waste.",
            "disjoint_carcass": "Cut through ligaments and cartilage to separate limbs from the main body of poultry or small game.",
            "trim_connective_tissue": "Remove tough silverskin, gristle, or sinew from the surface of meat muscles.",
            "extract_buried_bone": "Reach deep into meat to cut around and remove a specific bone.",
            "butterfly_thick_protein": "Slice a thick cut of meat horizontally almost all the way through to open it like a book.",
            "slice_uniform_produce": "Create perfectly even, consistent slices of fruits or vegetables at a set thickness.",
            "julienne_vegetables": "Cut vegetables into long, thin, uniform matchsticks.",
            "cut_french_fries": "Cut potatoes or root vegetables into thick, uniform batons.",
            "cut_waffle_pattern": "Create lattice-shaped cuts by rotating the vegetable 90 degrees between slices on a fluted blade.",
            "shave_delicate_produce": "Create translucent, paper-thin slices of ingredients like truffles or radishes.",
            "emulsify_liquid_fat": "Vigorously mix two immiscible liquids into a stable, homogeneous mixture.",
            "smooth_heavy_sauce": "Agitate thick mixtures like béchamel to remove lumps.",
            "access_pan_corners": "Reach into the sharp angle between the base and sidewall of a pan to incorporate ingredients.",
            "dissolve_starch_thickener": "Mix starch-based agents into liquid without clumping.",
            "chop_through_bone": "Use weight and momentum to sever dense bone and cartilage.",
            "crush_aromatics": "Smash ingredients like ginger or lemongrass to release essential oils.",
            "tenderize_meat": "Strike meat with a blunt surface to break down muscle fibers.",
            "slice_midsize_produce": "Cut ingredients that are too large for a paring knife but small for a chef's knife (e.g., tomatoes).",
            "portion_sandwich_cheese": "Cut through prepared sandwiches or blocks of soft cheese.",
            "segment_citrus": "Cut between the membranes of citrus fruit to remove the flesh sections (supremes).",
            "rock_chop_herbs": "Mince ingredients by anchoring the knife tip and rocking the heel up and down.",
            "push_cut_vegetables": "Propel the blade forward and down to chop through firm vegetables cleanly.",
            "slice_large_produce": "Cut through large items like melons or roasts using long, continuous strokes.",
            "crush_garlic_clove": "Smash a garlic clove with the flat side of a blade to loosen the skin.",
            "beat_eggs": "Agitate eggs rapidly to mix the yolk and white.",
            "blend_batter": "Mix wet and dry ingredients into a semi-liquid mixture.",
            "whip_cream_whites": "Beat high-fat cream or egg whites to trap air bubbles and increase volume.",
            "peel_produce_skin": "Remove the exterior layer of fruits and vegetables using a slotted blade.",
            "shave_hard_cheese": "Cut thin, wide strips from blocks of hard cheese like Parmesan.",
            "shave_vegetable_ribbons": "Create long, thin strips of vegetables like zucchini or carrots.",
            "roll_dough_flat": "Compress dough into a thin, even sheet.",
            "crush_crumbs_nuts": "Pulverize dry ingredients like crackers or nuts into smaller particles.",
            "shape_pastry": "Manipulate dough into specific forms (e.g., shells, cookies).",
            "simmer_stock": "Maintain a liquid just below boiling point for long-duration extraction of flavor.",
            "boil_large_volume": "Bring a large quantity of liquid to a rolling boil.",
            "stew_meat_vegetables": "Cook solid ingredients submerged in liquid slowly.",
            "reduce_sauce_liquid": "Simmer a liquid in an open vessel to evaporate water and concentrate flavor.",
            "simmer_grains": "Cook rice, quinoa, or other grains in gently bubbling liquid.",
            "blanch_vegetables": "Briefly submerge vegetables in boiling water to cook partially.",
            "heat_small_liquid": "Warm small quantities of sauces or milk.",
            "boil_eggs": "Cook eggs in their shells submerged in boiling water.",
            "poach_eggs": "Cook shelled eggs gently in simmering liquid.",
            "saute_vegetables": "Cook food quickly in a small amount of fat over relatively high heat.",
            "sear_protein": "Brown the surface of meat at high temperature to develop flavor.",
            "flip_contents_motion": "Toss food in the pan using a wrist flick motion.",
            "poach_whole_fish": "Cook an entire fish gently in barely simmering liquid.",
            "steam_delicate_items": "Cook food using the vapor produced by boiling water.",
            "measure_spices": "Portion dry flavorings in small, specific volumes.",
            "scoop_dry_ingredients": "Dig into a container to retrieve and level a specific volume of powder or grains.",
            "measure_liquid_volume": "Assess the quantity of a fluid using graduated markings.",
            "pour_controlled_stream": "Dispense liquid at a steady rate.",
            "measure_internal_temp": "Determine the heat level inside a food item.",
            "monitor_cooking_progress": "Continuously track temperature changes over time.",
            "drain_pasta_water": "Separate cooked pasta from the boiling liquid.",
            "rinse_produce": "Wash fruits or vegetables under running water while allowing drainage.",
            "strain_large_solids": "Separate bones or vegetables from a stock or liquid.",
            "rice_cooked_potatoes": "Force cooked potatoes through small holes to create a fluffy texture.",
            "press_soft_vegetables": "Squeeze water out of cooked greens or soft vegetables.",
            "create_smooth_puree": "Process food into a paste free of lumps.",
            "strain_fine_sauce": "Pass liquid through a very fine mesh to remove minute particles.",
            "filter_impurities": "Remove sediments or curds from clarified liquids or custards.",
            "dust_powdered_sugar": "Sprinkle a fine, even layer of powder over a dish.",
            "roast_large_poultry": "Cook whole birds (chicken, turkey) using dry heat.",
            "roast_vegetables": "Cook cut vegetables in an oven to brown and caramelize them.",
            "catch_drippings": "Collect fat and juices released from meat during cooking.",
            "bake_bread_loaf": "Cook yeast-leavened dough in a deep rectangular form.",
            "mold_meatloaf": "Shape ground meat mixtures into a rectangular block while cooking.",
            "bake_pound_cake": "Cook dense batters in a deep, rectangular mold.",
            "bake_individual_portion": "Cook single-serving items.",
            "mold_souffle": "Provide straight vertical sides for egg-based dishes to rise.",
            "serve_condiments": "Present sauces or dips in small vessels.",
            "puree_soup": "Blend cooked ingredients into a smooth liquid soup.",
            "crush_ice": "Break ice cubes into slush or small chips.",
            "emulsify_dressing": "Blend oil and acid into a creamy salad dressing.",
            "liquefy_fruit": "Turn solid fruit into juice or smoothies.",
            "knead_bread_dough": "Work dough to develop gluten strands.",
            "whip_heavy_cream": "Aerate cream using high-speed mechanical rotation.",
            "mix_cake_batter": "Combine flour, sugar, and wet ingredients until just incorporated.",
            "cream_butter_sugar": "Beat fat and sugar together until light and fluffy.",
        }

    def pretty_print(self) -> str:
        """Return a newline separated list of `action: description` rows sorted by name."""
        rows = []
        for action, description in sorted(self.actions_lookup.items()):
            rows.append(f"{action}: {description}")
        return "\n".join(rows) if rows else "No actions defined."
        

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
    # Given the requested cooking actions, find the items that match in the db_items list.
    # Items must be ranked by the number of matches with the items with max # of matching actions at the top.
    active_actions = set(actions.actions or [])
    if not active_actions:
        return []

    scored_items = []
    for item in db_items:
        matches = len(active_actions.intersection(item.actions))
        if matches > 0:
            scored_items.append((matches, item))

    scored_items.sort(key=lambda entry: entry[0], reverse=True)
    return [item for _, item in scored_items]
