from typing import Optional

from pydantic import BaseModel, Field


class CookingActions(BaseModel):
    """Structured checklist of kitchen tasks for LLM parsing to flag relevant actions."""
    peel_round_produce: Optional[bool] = Field(
        default=None,
        description=
        "Removing the skin from spherical or curved items (e.g., apples, potatoes, turnips) where a curved blade or specific motion is required to maintain the shape.",
    )
    peel_irregular_produce: Optional[bool] = Field(
        default=None,
        description=
        "Removing skin from knobby, uneven, or small items (e.g., ginger, turmeric) where a straight, nimble blade is needed to navigate bumps.",
    )
    extract_stem_core: Optional[bool] = Field(
        default=None,
        description=
        "Using a pointed tip to scoop out the inedible stem or hard core from a fruit or vegetable (e.g., hulling strawberries, coring tomatoes) without cutting the item in half.",
    )
    remove_surface_blemish: Optional[bool] = Field(
        default=None,
        description="Surgically removing small spots, potato eyes, or bruises without wasting the surrounding flesh.",
    )
    sculpt_decorative_shape: Optional[bool] = Field(
        default=None,
        description="Carving vegetables for aesthetics rather than simple consumption, such as the 7-sided Tourné cut, mushroom fluting, or radish roses.",
    )
    slice_small_produce: Optional[bool] = Field(
        default=None,
        description="General cutting or slicing of small, hand-held items (e.g., garlic cloves, berries, cherry tomatoes).",
    )
    mince_small_aromatics: Optional[bool] = Field(
        default=None,
        description="Finely chopping small ingredients (e.g., garlic, shallots, herbs) on a cutting board to release flavor.",
    )
    slice_uniform_produce: Optional[bool] = Field(
        default=None,
        description="Creating identical, repeatable slices of hard vegetables (e.g., potato chips, cucumber rounds) with mechanical precision, usually for even cooking or presentation.",
    )
    julienne_vegetables: Optional[bool] = Field(
        default=None,
        description="Cutting vegetables into long, thin, uniform strips (matchsticks), typically for salads or quick sautéing.",
    )
    cut_french_fries: Optional[bool] = Field(
        default=None,
        description="Cutting potatoes or tubers into thick, uniform sticks (batonnets).",
    )
    cut_waffle_pattern: Optional[bool] = Field(
        default=None,
        description="Creating a corrugated, grid-like \"gaufrette\" cut that requires a specific ridged blade and rotation technique.",
    )
    shave_delicate_produce: Optional[bool] = Field(
        default=None,
        description="Cutting extremely thin, translucent shavings of firm ingredients (e.g., radishes, truffles, fennel).",
    )
    debone_raw_protein: Optional[bool] = Field(
        default=None,
        description="The act of separating raw meat from the bone structure with minimal waste.",
    )
    disjoint_carcass: Optional[bool] = Field(
        default=None,
        description="Cutting through cartilage and ligaments to separate a whole animal (like a chicken) into parts, without sawing through bone.",
    )
    trim_connective_tissue: Optional[bool] = Field(
        default=None,
        description="Sliding a blade under tough silverskin, sinew, or gristle to remove it from a muscle group (e.g., cleaning a tenderloin).",
    )
    trim_meat_fat: Optional[bool] = Field(
        default=None,
        description="General trimming of excess fat caps or loose skin from small or large cuts of meat.",
    )
    extract_buried_bone: Optional[bool] = Field(
        default=None,
        description="Maneuvering a blade deep inside a piece of meat to cut around and remove a bone (e.g., removing the aitch bone).",
    )
    butterfly_thick_protein: Optional[bool] = Field(
        default=None,
        description="Slicing a thick piece of meat (like a chicken breast or pork chop) horizontally almost all the way through to open it like a book.",
    )
    carve_cooked_protein: Optional[bool] = Field(
        default=None,
        description="Slicing large cuts of hot, cooked meat (e.g., turkey, roast beef, leg of lamb) for serving.",
    )
    slice_cured_fish: Optional[bool] = Field(
        default=None,
        description="Cutting thin, delicate sheets from cold, cured, or smoked fish (e.g., salmon, gravlax) without tearing the flesh.",
    )
    slice_uniform_portions: Optional[bool] = Field(
        default=None,
        description="Using long, single strokes to cut soft, cohesive foods (e.g., terrines, pâtés, roulades) into even servings.",
    )
    portion_boneless_meat: Optional[bool] = Field(
        default=None,
        description="Slicing raw, boneless sub-primals (e.g., a loin) into individual steaks or portions.",
    )
    emulsify_liquid_fat: Optional[bool] = Field(
        default=None,
        description="Forcefully combining liquids that usually separate (e.g., oil and vinegar, eggs and oil) into a stable mixture.",
    )
    smooth_heavy_sauce: Optional[bool] = Field(
        default=None,
        description="Agitating thick, heavy mixtures (e.g., gravy, béchamel, custard) to remove lumps and ensure a silky texture.",
    )
    dissolve_starch_thickener: Optional[bool] = Field(
        default=None,
        description="Breaking up clumps of flour (roux) or cornstarch against the bottom of a pan.",
    )
    access_pan_corners: Optional[bool] = Field(
        default=None,
        description="Reaching the tight 90-degree edges of a straight-sided sauté pan or pot to prevent sticking or burning.",
    )
    pierce_check_texture: Optional[bool] = Field(
        default=None,
        description="Stabbing a dense item (e.g., a boiling potato or baking cake) to test for tenderness or doneness.",
    )
    sever_binding_material: Optional[bool] = Field(
        default=None,
        description="Cutting non-food kitchen items, such as butcher's twine, vacuum seal bags, or packaging.",
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
