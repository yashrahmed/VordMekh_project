from pydantic import BaseModel, Field
from typing import Optional, List, Set, Tuple


# ============================================================================
# COOKING EQUIPMENT CONSTANTS
# ============================================================================

# Heat Source Constants
HEAT_GAS_COOKTOP = "gas cooktop"
HEAT_INDUCTION_COOKTOP = "induction cooktop"
HEAT_PROPANE_STOVE = "propane stove"
HEAT_BUTANE_STOVE = "butane stove"
HEAT_INDUCTION_STOVE = "induction stove"

# Cooking Vessel Constants
VESSEL_POT = "pot"
VESSEL_INDUCTION_POT = "induction pot"
VESSEL_STOCKPOT_SPIGOT = "stockpot with spigot"

# Cooking Vessel Handling tool constants
HANDLING_TOOL_MITTEN = "mittens"
HANDLING_TOOL_CLAMP = "clamps/tongs"

# Cooking Vessel Accessory Constants
ACCESSORY_POT_LID = " Lid for the pot or the cup"

# Cooking Platform Constants
PLATFORM_PORTABLE_TABLE = "portable table"

# Ignition Tool Constants
IGNITION_LIGHTER = "lighter"
IGNITION_MATCHES = "matches"

# Fuel/Power Attachment Constants
FUEL_BUTANE_CANISTER = "butane canister"
FUEL_PROPANE_CANISTER = "propane canister"
FUEL_PROPANE_TANK = "propane tank"
FUEL_BATTERY = "battery"

# Fuel Connector Constants
CONNECTOR_PROPANE_HOSE_ADAPTER = "propane hose adapter"

# Wind Mitigation Equipment Constants
WIND_WINDSHIELD = "windshield"

# Rain Mitigation Equipment Constants
RAIN_RAINFLY = "rainfly"
RAIN_CANOPY = "canopy"

# Lighting Equipment Constants
LIGHTING_LANTERN = "lantern"
LIGHTING_HEADLAMP = "headlamp"


# Chai Preparation Tool Constants
TOOL_MORTAR_PESTLE = "mortar and pestle"
TOOL_ROLLING_PIN = "rolling pin"
TOOL_SPICE_GRINDER = "spice grinder"
TOOL_COFFEE_GRINDER = "coffee grinder"
TOOL_PEELER = "peeler"
TOOL_PARING_KNIFE = "paring knife"
TOOL_KNIFE = "knife"
TOOL_MANDOLINE = "mandoline slicer"
TOOL_GRATER = "grater"
TOOL_MICROPLANE = "microplane"
TOOL_CHEFS_KNIFE = "chef's knife"
TOOL_CLEAVER = "small cleaver"
TOOL_SPOON = "spoon"
TOOL_LADLE = "ladle"
TOOL_STRAINER = "strainer"
TOOL_MUSLIN_CLOTH = "muslin cloth"
TOOL_FROTHER = "frother"
TOOL_CITRUS_SQUEEZER = "citrus squeezer"
TOOL_REAMER = "reamer"
TOOL_FORK = "fork"
TOOL_TOOTHPICK = "toothpick"
TOOL_SKEWER = "skewer"
TOOL_ZESTER = "zester"

# ============================================================================
# CHAI PREPARATION AND SCENARIO FRAMES
# ============================================================================

class CookingSceneConditionFrame(BaseModel):
    """
    Frame capturing environmental and preparation details for a cooking scene.

    Fill procedure (step-by-step):
    1) Heat Source Options:
       - By default, all heat sources are available (gas cooktop, induction cooktop, propane stove, butane stove, induction stove).
       - If the text explicitly mentions specific heat sources (e.g., "using a propane stove", "on a gas cooktop"),
         set heat_source_options to only include those mentioned.
       - If no heat source is mentioned, keep the default list of all options.
       - Examples:
         * "brewing chai on a propane stove" → heat_source_options = ["propane stove"]
         * "using either gas or induction cooktop" → heat_source_options = ["gas cooktop", "induction cooktop"]
         * "making chai outdoors" (no specific heat source) → keep default list

    2) Ignition Requirement (is_ignition_needed):
       - Set to True if the text mentions needing to light/ignite the heat source.
       - Keywords: "light the stove", "ignite", "use lighter", "use matches", "strike a match"
       - Set to False if using electric/battery-powered heat sources.
       - Leave None if not specified.

    3) Environmental Conditions:
       - is_windy: Set to True if text mentions wind, breeze, gusts, or wind protection needs.
         Examples: "windy conditions", "protect from wind", "breeze blowing", "need windshield"

       - is_rainy: Set to True if text mentions rain, wet conditions, or rain protection needs.
         Examples: "rainy weather", "protect from rain", "drizzling", "need cover", "use rainfly"

       - is_dark: Set to True if text mentions darkness, low light, nighttime, or lighting needs.
         Examples: "brewing at night", "dark conditions", "need light", "can't see well", "early morning darkness"

    4) Ground and Surface Conditions (has_uneven_groumd):
       - Set to True if text mentions uneven, rocky, sloped, or unstable ground/surface.
       - Keywords: "uneven ground", "rocky terrain", "sloped surface", "need stable surface", "unstable ground"
       - Set to False if explicitly mentions flat/level surface.
       - Leave None if not specified.

    5) Vessel Characteristics (does_vessel_have_handle):
       - Set to True if text mentions the pot/vessel has a handle or is easy to grip.
       - Set to False if text mentions no handle, need for clamps/tongs, or difficulty holding.
       - Keywords for False: "no handle", "need clamps", "use tongs", "difficult to hold", "hot to touch"
       - Leave None if not specified.

    6) Group Size (preparaing_for_a_large_group):
       - Set to True if text mentions preparing for many people, a large group, crowd, or gathering.
       - Keywords: "large group", "many people", "crowd", "20+ servings", "party", "gathering", "event"
       - Set to False if explicitly mentions small group (1-4 people) or personal use.
       - Leave None if not specified.

    Summary:
       - This model helps LLMs extract cooking scene context from natural language descriptions.
       - It captures environmental challenges, equipment constraints, and preparation scale.
       - Boolean fields should be set based on explicit mentions or strong implications in the text.
    """
    heat_source_options: List[str] = [
        HEAT_GAS_COOKTOP,
        HEAT_INDUCTION_COOKTOP,
        HEAT_PROPANE_STOVE,
        HEAT_BUTANE_STOVE,
        HEAT_INDUCTION_STOVE
    ]
    is_ignition_needed: Optional[bool] = None
    is_windy: Optional[bool] = None
    is_rainy: Optional[bool] = None
    is_dark: Optional[bool] = None
    has_uneven_groumd: Optional[bool] = None
    does_vessel_have_handle: Optional[bool] = None
    preparaing_for_a_large_group: Optional[bool] = None

class ChaiPreparationIngredientsActionsFrame(BaseModel):
    """
    Frame to extract chai ingredients and infer preparation actions from the recipe text.

    Fill procedure (step-by-step):
    1) Identify chai type:
       - If the recipe names a style (e.g., “Masala Chai”), set `chai_type` to that.
       - Otherwise use a concise descriptive name (e.g., “Milk tea with spices”).

    2) Parse ingredient lines and place each of them into the ingredients list.
    
    3) Infer Action Flags:
       - An action flag should be set to True based on two conditions:
         a) Explicit Instruction: The recipe steps explicitly state the action (e.g., "stir the chai," "strain into a cup", "let steep covered for 5 minutes").
         b) Implied Action from Ingredient State: An ingredient's description implies the action. For example:
            - "peeled ginger" → `peel_ingredients=True`
            - "sliced lemon" → `slice_ingredients=True`
            - "crushed cardamom" → `crush_spices=True`
            - "grated turmeric" → `grate_ingredients=True`
            - "lemon juice" or "squeezed lime" → `squeeze_ingredients=True`
            - "pierced cardamom pods" → `pierce_ingredients=True`
            - "lemon zest" or "zested orange peel" → `zest_ingredients=True`
       - NOTE: Bruising and muddling herbs (e.g., "bruise mint leaves," "muddle tulsi," "muddle lemongrass") should be classified as crushing. Set `crush_spices=True` for these actions.
       - EXCEPTION: Do not infer an action for pre-processed dry goods. The most common case is "ground" spices (e.g., "ground ginger," "cinnamon powder"). These refer to a product form, not a preparation step. `grind_spices` should only be True if explicitly instructed (e.g., "grind the cloves").

    Summary:
       - This model infers the cook's preparation steps from both direct instructions and ingredient descriptions.
       - It assumes that forms like 'peeled' or 'sliced' are actions to be performed now, except for common pre-packaged forms like 'ground' powder.
    """

    chai_type: Optional[str] = Field(
        None,
        description="Named style if present (e.g., 'Masala Chai', 'Adrak Chai'); otherwise a concise descriptive label or null."
    )

    # All the ingredient line items
    ingredients: Optional[List[str]] = Field(None, description="A line item on a recipe list which includes the name, quantity, alternatives etc e.g. '0.5 teaspoon of Kashmiri green tea leaves (or other mild green tea)' or '1.0 cup of water'")
    
    # --- Actions (Now with INFERENTIAL logic) ---
    crush_spices: Optional[bool] = Field(
        None,
        description="True if crushing is instructed OR if a spice is listed as 'crushed'."
    )
    grind_spices: Optional[bool] = Field(
        None,
        description="True ONLY if the recipe explicitly instructs grinding now. Unlike other actions, this is NOT inferred from 'ground' ingredients."
    )
    peel_ingredients: Optional[bool] = Field(
        None,
        description="True if peeling is instructed OR if an ingredient is listed as 'peeled'."
    )
    slice_ingredients: Optional[bool] = Field(
        None,
        description="True if slicing is instructed OR if an ingredient is listed as 'sliced'."
    )
    grate_ingredients: Optional[bool] = Field(
        None,
        description="True if grating is instructed OR if an ingredient is listed as 'grated'."
    )
    chop_ingredients: Optional[bool] = Field(
        None,
        description="True if chopping is instructed OR if an ingredient is listed as 'chopped'."
    )
    stir_chai: Optional[bool] = Field(
        None,
        description="True if the steps instruct stirring/mixing during cooking."
    )
    strain_chai: Optional[bool] = Field(
        None,
        description="True if the steps instruct straining/filtering before serving."
    )
    aerate_chai: Optional[bool] = Field(
        None,
        description="True if the recipe instructs pulling/frothing/aerating."
    )
    squeeze_ingredients: Optional[bool] = Field(
        None,
        description="True if squeezing is instructed OR if an ingredient is listed as 'squeezed' (e.g., lemon juice, lime juice)."
    )
    pierce_ingredients: Optional[bool] = Field(
        None,
        description="True if piercing is instructed OR if an ingredient is listed as 'pierced' (e.g., pierced cardamom pods to release flavor)."
    )
    zest_ingredients: Optional[bool] = Field(
        None,
        description="True if zesting is instructed OR if an ingredient is listed as 'zested' (e.g., lemon zest, lime zest, orange zest)."
    )
    infuse_flavors: Optional[bool] = Field(
        None,
        description="True if the recipe instructs steeping, infusing, or covering to let flavors develop (e.g., 'let steep for 5 minutes', 'cover and simmer', 'infuse flavors'). This step or the use of the lid must be specified explicitly. Set false otherwise."
    )
    use_clay_vessel: Optional[bool] = Field(
        None,
        description="True if the recipe instructs preparing or serving in a clay pot or clay cup (e.g., 'prepare in earthenware', 'serve in kulhad', 'use clay pot')."
    )

    def generate_description(self) -> str:
        """Generate a formatted description of ingredients and preparation actions."""
        description = "Ingredients:\n"
        ingredient_count = 1
        description_lines = []

        if self.ingredients:
            for ing in self.ingredients:
                description_line = f"{ing}\n"
                description_lines.append(description_line)
                ingredient_count += 1
        
        description += ''.join(description_lines)

        # Add preparation actions section
        description += "\nPreparation actions:\n"
        actions = []

        if self.crush_spices:
            actions.append("Crush spices")
        if self.grind_spices:
            actions.append("Grind spices")
        if self.peel_ingredients:
            actions.append("Peel ingredients (ginger, citrus, etc.)")
        if self.chop_ingredients:
            actions.append("Chop ingredients")
        if self.slice_ingredients:
            actions.append("Slice ingredients (ginger, citrus, etc.)")
        if self.grate_ingredients:
            actions.append("Grate ingredients (ginger, etc.)")
        if self.stir_chai:
            actions.append("Stir chai during preparation")
        if self.strain_chai:
            actions.append("Strain chai before serving")
        if self.aerate_chai:
            actions.append("Aerate chai (pull/froth)")
        if self.squeeze_ingredients:
            actions.append("Squeeze ingredients (lemon, lime, etc.)")
        if self.pierce_ingredients:
            actions.append("Pierce ingredients (cardamom pods, etc.)")
        if self.zest_ingredients:
            actions.append("Zest citrus (lemon, lime, orange, etc.)")
        if self.infuse_flavors:
            actions.append("Infuse flavors (steep covered to develop taste)")
        if self.use_clay_vessel:
            actions.append("Prepare in clay pot or cup")

        for i, action in enumerate(actions, 1):
            description += f"{i}. {action}\n"

        return description

# ============================================================================
# HELPER CLASSES AND FUNCTIONS
# ============================================================================

class ChaiRecipe(BaseModel):
    """Class to represent an LLMs response to a request for chai recipe"""
    recipe_text: Optional[str] = Field(None, description="The raw text of the recipe")
    is_valid: bool = Field(True, description="A boolean flag set to indicate if the response is valid. Flag is set to false if the system failed or used asked for a request that was unrelated to a chai recipe.")

def generate_chai_tooling(frame: ChaiPreparationIngredientsActionsFrame) -> str:
    """Generate a formatted description of preparation tools with their purposes."""

    tooling_rules = [
        ("crush_spices", [TOOL_MORTAR_PESTLE, TOOL_ROLLING_PIN], "for crushing spices"),
        ("grind_spices", [TOOL_SPICE_GRINDER, TOOL_COFFEE_GRINDER], "for grinding spices just before brewing"),
        ("slice_ingredients", [TOOL_KNIFE, TOOL_MANDOLINE], "for slicing ingredients"),
        ("chop_ingredients", [TOOL_CHEFS_KNIFE, TOOL_CLEAVER], "for chopping ingredients"),
        ("peel_ingredients", [TOOL_PEELER, TOOL_PARING_KNIFE], "for peeling"),
        ("grate_ingredients", [TOOL_GRATER, TOOL_MICROPLANE], "for grating aromatics"),
        ("stir_chai", [TOOL_SPOON, TOOL_LADLE], "for mixing and stirring"),
        ("strain_chai", [TOOL_STRAINER, TOOL_MUSLIN_CLOTH], "for filtering tea"),
        ("aerate_chai", [TOOL_LADLE, TOOL_FROTHER], "for creating froth and aeration"),
        ("squeeze_ingredients", [TOOL_CITRUS_SQUEEZER, TOOL_REAMER], "for squeezing citrus and other ingredients"),
        ("pierce_ingredients", [TOOL_FORK, TOOL_TOOTHPICK, TOOL_SKEWER], "for piercing ingredients to release flavor"),
        ("zest_ingredients", [TOOL_ZESTER, TOOL_MICROPLANE, TOOL_GRATER], "for zesting citrus"),
        ("infuse_flavors", [ACCESSORY_POT_LID], "for infusing flavors and trapping aromas"),
        ("use_clay_vessel", [HANDLING_TOOL_CLAMP], "for handling hot clay pots or cups"),
    ]

    description = "Preparation tools:\n"
    tool_count = 1

    for action_flag, tools, purpose in tooling_rules:
        if getattr(frame, action_flag, False):
            tools_str = " or ".join(tools)
            description += f"{tool_count}. {tools_str} - {purpose}\n"
            tool_count += 1

    return description

def get_tooling_for_scene(
    scene_conditions: CookingSceneConditionFrame,
) -> str:
    """Generate tooling requirements based on scene and environment context."""
    if scene_conditions is None:
        raise ValueError("scene_conditions must be provided.")

    def add_item(item: Optional[str], reason: str, items: List[Tuple[str, str]], seen: Set[str]) -> None:
        if item and item not in seen:
            items.append((item, reason))
            seen.add(item)

    tooling: List[Tuple[str, str]] = []
    seen_items: Set[str] = set()
    heat_source_options = scene_conditions.heat_source_options

    # Always include all heat source options
    if heat_source_options:
        for heat_source in heat_source_options:
            add_item(heat_source, "available heat source option for cooking", tooling, seen_items)

    if scene_conditions.does_vessel_have_handle is False:
        add_item(HANDLING_TOOL_CLAMP, "vessel has no handle", tooling, seen_items)
        add_item(HANDLING_TOOL_MITTEN, "vessel has no handle", tooling, seen_items)
    if scene_conditions.preparaing_for_a_large_group:
        add_item(VESSEL_STOCKPOT_SPIGOT, "preparing for a large group", tooling, seen_items)

    # Process each heat source option for specific equipment needs
    if heat_source_options:
        for heat_source in heat_source_options:
            if heat_source == HEAT_GAS_COOKTOP:
                add_item(VESSEL_POT, "gas cooktop option available", tooling, seen_items)
            if heat_source == HEAT_INDUCTION_COOKTOP:
                add_item(VESSEL_INDUCTION_POT, "induction cooktop option available", tooling, seen_items)
            if heat_source == HEAT_BUTANE_STOVE:
                add_item(FUEL_BUTANE_CANISTER, "butane stove option available", tooling, seen_items)
            if heat_source == HEAT_PROPANE_STOVE:
                add_item(FUEL_PROPANE_CANISTER, "propane stove option available", tooling, seen_items)
                add_item(FUEL_PROPANE_TANK, "propane stove can connect to refillable tank", tooling, seen_items)
                add_item(CONNECTOR_PROPANE_HOSE_ADAPTER, "propane tank setup may require hose adapter", tooling, seen_items)
            if heat_source == HEAT_INDUCTION_STOVE:
                add_item(VESSEL_INDUCTION_POT, "induction stove option available", tooling, seen_items)
                add_item(FUEL_BATTERY, "induction stove requires portable power", tooling, seen_items)

    if scene_conditions.is_windy:
        add_item(WIND_WINDSHIELD, "conditions are windy", tooling, seen_items)
    if scene_conditions.is_rainy:
        add_item(RAIN_RAINFLY, "conditions are rainy", tooling, seen_items)
        add_item(RAIN_CANOPY, "conditions are rainy", tooling, seen_items)
    if scene_conditions.is_dark:
        add_item(LIGHTING_LANTERN, "conditions are dark", tooling, seen_items)
        add_item(LIGHTING_HEADLAMP, "conditions are dark", tooling, seen_items)
    if scene_conditions.is_ignition_needed:
        add_item(IGNITION_LIGHTER, "ignition is needed", tooling, seen_items)
        add_item(IGNITION_MATCHES, "ignition is needed", tooling, seen_items)
    if scene_conditions.has_uneven_groumd:
        add_item(PLATFORM_PORTABLE_TABLE, "ground is uneven", tooling, seen_items)

    description = "Tooling required based on the provided scene conditions:\n"
    if tooling:
        for idx, (item, reason) in enumerate(tooling, start=1):
            description += f"{idx}. {item} (because {reason})\n"
    else:
        description += "No additional tooling required based on the provided conditions.\n"

    return description
