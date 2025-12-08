from typing import Optional, List

from pydantic import BaseModel, Field

class Item(BaseModel):
    """Inventory item describing a tool, vessel, or appliance plus its supported actions."""

    key: str = Field(..., description="Unique machine-friendly identifier that matches items.yml.")
    name: str = Field(..., description="Human-readable label for the item.")
    description: str = Field(..., description="Text describing construction, form factor, or usage context.")
    actions: List[str] = Field(..., description="List of CookingActions keys this item can perform.")

class CookingActions(BaseModel):
    """Structured checklist of kitchen tasks for LLM parsing to flag relevant actions."""

    peel_round_produce: Optional[bool] = Field(
        default=None,
        description="Remove the outer skin from spherical or rounded fruits and vegetables while maintaining their curvature.",
    )
    extract_stem_core: Optional[bool] = Field(
        default=None,
        description="Surgically remove the woody stem or hard core from a fruit or vegetable.",
    )
    remove_surface_blemish: Optional[bool] = Field(
        default=None,
        description="Excise small rotten spots, eyes, or imperfections from the surface of produce.",
    )
    sculpt_decorative_shape: Optional[bool] = Field(
        default=None,
        description="Cut vegetables into specific geometric shapes for visual presentation (e.g., tourné cut).",
    )
    peel_irregular_produce: Optional[bool] = Field(
        default=None,
        description="Remove skin from ingredients with uneven surfaces like ginger or Jerusalem artichokes requiring intricate navigation.",
    )
    slice_small_produce: Optional[bool] = Field(
        default=None,
        description="Cut small items like garlic, shallots, or strawberries into thin, consistent pieces.",
    )
    mince_small_aromatics: Optional[bool] = Field(
        default=None,
        description="Chop herbs, garlic, or ginger into extremely fine pieces using a rocking or pivot motion.",
    )
    trim_meat_fat: Optional[bool] = Field(
        default=None,
        description="Remove small, precise areas of fat, silverskin, or veins from a cut of meat.",
    )
    pierce_check_texture: Optional[bool] = Field(
        default=None,
        description="Insert the tip of a tool into food to assess tenderness or cooking progress.",
    )
    sever_binding_material: Optional[bool] = Field(
        default=None,
        description="Cut through twine, string, or netting used to truss meat.",
    )
    carve_cooked_protein: Optional[bool] = Field(
        default=None,
        description="Slice warm, large roasted meats into serving pieces.",
    )
    slice_cured_fish: Optional[bool] = Field(
        default=None,
        description="Cut very thin, translucent sheets of cold fish like smoked salmon.",
    )
    slice_uniform_portions: Optional[bool] = Field(
        default=None,
        description="Cut terrines, pâtés, or loaves into identical serving sizes for presentation.",
    )
    portion_boneless_meat: Optional[bool] = Field(
        default=None,
        description="Divide large raw muscle groups into steaks or cutlets.",
    )
    debone_raw_protein: Optional[bool] = Field(
        default=None,
        description="Separate raw meat from the bone structure with minimal waste.",
    )
    disjoint_carcass: Optional[bool] = Field(
        default=None,
        description="Cut through ligaments and cartilage to separate limbs from the main body of poultry or small game.",
    )
    trim_connective_tissue: Optional[bool] = Field(
        default=None,
        description="Remove tough silverskin, gristle, or sinew from the surface of meat muscles.",
    )
    extract_buried_bone: Optional[bool] = Field(
        default=None,
        description="Reach deep into meat to cut around and remove a specific bone.",
    )
    butterfly_thick_protein: Optional[bool] = Field(
        default=None,
        description="Slice a thick cut of meat horizontally almost all the way through to open it like a book.",
    )
    slice_uniform_produce: Optional[bool] = Field(
        default=None,
        description="Create perfectly even, consistent slices of fruits or vegetables at a set thickness.",
    )
    julienne_vegetables: Optional[bool] = Field(
        default=None,
        description="Cut vegetables into long, thin, uniform matchsticks.",
    )
    cut_french_fries: Optional[bool] = Field(
        default=None,
        description="Cut potatoes or root vegetables into thick, uniform batons.",
    )
    cut_waffle_pattern: Optional[bool] = Field(
        default=None,
        description="Create lattice-shaped cuts by rotating the vegetable 90 degrees between slices on a fluted blade.",
    )
    shave_delicate_produce: Optional[bool] = Field(
        default=None,
        description="Create translucent, paper-thin slices of ingredients like truffles or radishes.",
    )
    emulsify_liquid_fat: Optional[bool] = Field(
        default=None,
        description="Vigorously mix two immiscible liquids into a stable, homogeneous mixture.",
    )
    smooth_heavy_sauce: Optional[bool] = Field(
        default=None,
        description="Agitate thick mixtures like béchamel to remove lumps.",
    )
    access_pan_corners: Optional[bool] = Field(
        default=None,
        description="Reach into the sharp angle between the base and sidewall of a pan to incorporate ingredients.",
    )
    dissolve_starch_thickener: Optional[bool] = Field(
        default=None,
        description="Mix starch-based agents into liquid without clumping.",
    )
    chop_through_bone: Optional[bool] = Field(
        default=None,
        description="Use weight and momentum to sever dense bone and cartilage.",
    )
    crush_aromatics: Optional[bool] = Field(
        default=None,
        description="Smash ingredients like ginger or lemongrass to release essential oils.",
    )
    tenderize_meat: Optional[bool] = Field(
        default=None,
        description="Strike meat with a blunt surface to break down muscle fibers.",
    )
    slice_midsize_produce: Optional[bool] = Field(
        default=None,
        description="Cut ingredients that are too large for a paring knife but small for a chef's knife (e.g., tomatoes).",
    )
    portion_sandwich_cheese: Optional[bool] = Field(
        default=None,
        description="Cut through prepared sandwiches or blocks of soft cheese.",
    )
    segment_citrus: Optional[bool] = Field(
        default=None,
        description="Cut between the membranes of citrus fruit to remove the flesh sections (supremes).",
    )
    rock_chop_herbs: Optional[bool] = Field(
        default=None,
        description="Mince ingredients by anchoring the knife tip and rocking the heel up and down.",
    )
    push_cut_vegetables: Optional[bool] = Field(
        default=None,
        description="Propel the blade forward and down to chop through firm vegetables cleanly.",
    )
    slice_large_produce: Optional[bool] = Field(
        default=None,
        description="Cut through large items like melons or roasts using long, continuous strokes.",
    )
    crush_garlic_clove: Optional[bool] = Field(
        default=None,
        description="Smash a garlic clove with the flat side of a blade to loosen the skin.",
    )
    beat_eggs: Optional[bool] = Field(
        default=None,
        description="Agitate eggs rapidly to mix the yolk and white.",
    )
    blend_batter: Optional[bool] = Field(
        default=None,
        description="Mix wet and dry ingredients into a semi-liquid mixture.",
    )
    whip_cream_whites: Optional[bool] = Field(
        default=None,
        description="Beat high-fat cream or egg whites to trap air bubbles and increase volume.",
    )
    peel_produce_skin: Optional[bool] = Field(
        default=None,
        description="Remove the exterior layer of fruits and vegetables using a slotted blade.",
    )
    shave_hard_cheese: Optional[bool] = Field(
        default=None,
        description="Cut thin, wide strips from blocks of hard cheese like Parmesan.",
    )
    shave_vegetable_ribbons: Optional[bool] = Field(
        default=None,
        description="Create long, thin strips of vegetables like zucchini or carrots.",
    )
    roll_dough_flat: Optional[bool] = Field(
        default=None,
        description="Compress dough into a thin, even sheet.",
    )
    crush_crumbs_nuts: Optional[bool] = Field(
        default=None,
        description="Pulverize dry ingredients like crackers or nuts into smaller particles.",
    )
    shape_pastry: Optional[bool] = Field(
        default=None,
        description="Manipulate dough into specific forms (e.g., shells, cookies).",
    )
    simmer_stock: Optional[bool] = Field(
        default=None,
        description="Maintain a liquid just below boiling point for long-duration extraction of flavor.",
    )
    boil_large_volume: Optional[bool] = Field(
        default=None,
        description="Bring a large quantity of liquid to a rolling boil.",
    )
    stew_meat_vegetables: Optional[bool] = Field(
        default=None,
        description="Cook solid ingredients submerged in liquid slowly.",
    )
    reduce_sauce_liquid: Optional[bool] = Field(
        default=None,
        description="Simmer a liquid in an open vessel to evaporate water and concentrate flavor.",
    )
    simmer_grains: Optional[bool] = Field(
        default=None,
        description="Cook rice, quinoa, or other grains in gently bubbling liquid.",
    )
    blanch_vegetables: Optional[bool] = Field(
        default=None,
        description="Briefly submerge vegetables in boiling water to cook partially.",
    )
    heat_small_liquid: Optional[bool] = Field(
        default=None,
        description="Warm small quantities of sauces or milk.",
    )
    boil_eggs: Optional[bool] = Field(
        default=None,
        description="Cook eggs in their shells submerged in boiling water.",
    )
    poach_eggs: Optional[bool] = Field(
        default=None,
        description="Cook shelled eggs gently in simmering liquid.",
    )
    saute_vegetables: Optional[bool] = Field(
        default=None,
        description="Cook food quickly in a small amount of fat over relatively high heat.",
    )
    sear_protein: Optional[bool] = Field(
        default=None,
        description="Brown the surface of meat at high temperature to develop flavor.",
    )
    flip_contents_motion: Optional[bool] = Field(
        default=None,
        description="Toss food in the pan using a wrist flick motion.",
    )
    poach_whole_fish: Optional[bool] = Field(
        default=None,
        description="Cook an entire fish gently in barely simmering liquid.",
    )
    steam_delicate_items: Optional[bool] = Field(
        default=None,
        description="Cook food using the vapor produced by boiling water.",
    )
    measure_spices: Optional[bool] = Field(
        default=None,
        description="Portion dry flavorings in small, specific volumes.",
    )
    scoop_dry_ingredients: Optional[bool] = Field(
        default=None,
        description="Dig into a container to retrieve and level a specific volume of powder or grains.",
    )
    measure_liquid_volume: Optional[bool] = Field(
        default=None,
        description="Assess the quantity of a fluid using graduated markings.",
    )
    pour_controlled_stream: Optional[bool] = Field(
        default=None,
        description="Dispense liquid at a steady rate.",
    )
    measure_internal_temp: Optional[bool] = Field(
        default=None,
        description="Determine the heat level inside a food item.",
    )
    monitor_cooking_progress: Optional[bool] = Field(
        default=None,
        description="Continuously track temperature changes over time.",
    )
    drain_pasta_water: Optional[bool] = Field(
        default=None,
        description="Separate cooked pasta from the boiling liquid.",
    )
    rinse_produce: Optional[bool] = Field(
        default=None,
        description="Wash fruits or vegetables under running water while allowing drainage.",
    )
    strain_large_solids: Optional[bool] = Field(
        default=None,
        description="Separate bones or vegetables from a stock or liquid.",
    )
    rice_cooked_potatoes: Optional[bool] = Field(
        default=None,
        description="Force cooked potatoes through small holes to create a fluffy texture.",
    )
    press_soft_vegetables: Optional[bool] = Field(
        default=None,
        description="Squeeze water out of cooked greens or soft vegetables.",
    )
    create_smooth_puree: Optional[bool] = Field(
        default=None,
        description="Process food into a paste free of lumps.",
    )
    strain_fine_sauce: Optional[bool] = Field(
        default=None,
        description="Pass liquid through a very fine mesh to remove minute particles.",
    )
    filter_impurities: Optional[bool] = Field(
        default=None,
        description="Remove sediments or curds from clarified liquids or custards.",
    )
    dust_powdered_sugar: Optional[bool] = Field(
        default=None,
        description="Sprinkle a fine, even layer of powder over a dish.",
    )
    roast_large_poultry: Optional[bool] = Field(
        default=None,
        description="Cook whole birds (chicken, turkey) using dry heat.",
    )
    roast_vegetables: Optional[bool] = Field(
        default=None,
        description="Cook cut vegetables in an oven to brown and caramelize them.",
    )
    catch_drippings: Optional[bool] = Field(
        default=None,
        description="Collect fat and juices released from meat during cooking.",
    )
    bake_bread_loaf: Optional[bool] = Field(
        default=None,
        description="Cook yeast-leavened dough in a deep rectangular form.",
    )
    mold_meatloaf: Optional[bool] = Field(
        default=None,
        description="Shape ground meat mixtures into a rectangular block while cooking.",
    )
    bake_pound_cake: Optional[bool] = Field(
        default=None,
        description="Cook dense batters in a deep, rectangular mold.",
    )
    bake_individual_portion: Optional[bool] = Field(
        default=None,
        description="Cook single-serving items.",
    )
    mold_souffle: Optional[bool] = Field(
        default=None,
        description="Provide straight vertical sides for egg-based dishes to rise.",
    )
    serve_condiments: Optional[bool] = Field(
        default=None,
        description="Present sauces or dips in small vessels.",
    )
    puree_soup: Optional[bool] = Field(
        default=None,
        description="Blend cooked ingredients into a smooth liquid soup.",
    )
    crush_ice: Optional[bool] = Field(
        default=None,
        description="Break ice cubes into slush or small chips.",
    )
    emulsify_dressing: Optional[bool] = Field(
        default=None,
        description="Blend oil and acid into a creamy salad dressing.",
    )
    liquefy_fruit: Optional[bool] = Field(
        default=None,
        description="Turn solid fruit into juice or smoothies.",
    )
    knead_bread_dough: Optional[bool] = Field(
        default=None,
        description="Work dough to develop gluten strands.",
    )
    whip_heavy_cream: Optional[bool] = Field(
        default=None,
        description="Aerate cream using high-speed mechanical rotation.",
    )
    mix_cake_batter: Optional[bool] = Field(
        default=None,
        description="Combine flour, sugar, and wet ingredients until just incorporated.",
    )
    cream_butter_sugar: Optional[bool] = Field(
        default=None,
        description="Beat fat and sugar together until light and fluffy.",
    )

    def pretty_print(self) -> str:
        """Return newline separated action states, skipping unspecified values."""
        rows = []
        for field_name, value in self.model_dump().items():
            if value is None:
                continue
            label = field_name.replace("_", " ").title()
            rows.append(f"{label}: {'Yes' if value else 'No'}")
        return "\n".join(rows) if rows else "No actions detected."
