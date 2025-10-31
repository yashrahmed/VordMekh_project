from pydantic import BaseModel, Field
from typing import Optional, List, Set


# ============================================================================
# COOKING EQUIPMENT CONSTANTS
# ============================================================================

# Scene Type Constants
SCENE_HOME = "home"
SCENE_CAMPSITE = "campsite"
SCENE_ANY = "anywhere"

# Condition Constants
CONDITION_NORMAL = "normal"
CONDITION_WINDY = "windy"
CONDITION_RAINY = "rainy"
CONDITION_DARK = "dark"
CONDITION_LARGE_GROUP = "preparing for a large group"

# Heat Source Constants
HEAT_GAS_COOKTOP = "gas cooktop"
HEAT_INDUCTION_COOKTOP = "induction cooktop"
HEAT_PROPANE_STOVE = "propane stove"
HEAT_BUTANE_STOVE = "butane stove"
HEAT_INDUCTION_STOVE = "induction stove"

# Cooking Vessel Constants
VESSEL_POT = "pot"
VESSEL_POT_NO_HANDLE = "pot with no handle"
VESSEL_INDUCTION_POT = "induction pot"
VESSEL_STOCKPOT_SPIGOT = "stockpot with spigot"

# Cooking Vessel Handling tool constants
HANDLING_TOOL_MITTEN = "mittens"
HANDLING_TOOL_CLAMP = "clamps/tongs"

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


# ============================================================================
# CHAI PREPARATION CONSTANTS
# ============================================================================

# Chai Type Constants
CHAI_MASALA = "Masala Chai"
CHAI_ADRAK = "Adrak Chai"
CHAI_SULAIMANI = "Sulaimani Chai"
CHAI_KASHMIRI = "Kashmiri Chai"
CHAI_KAHWAH = "Kahwah"

# Liquid Ingredient Constants
LIQUID_WATER = "Water"
LIQUID_WHOLE_MILK = "Whole Milk"

# Tea Leaf Constants
TEA_LOOSE_BLACK = "Loose Black Tea"
TEA_KASHMIRI_GREEN = "Kashmiri Green Tea Leaves"
TEA_GREEN = "Green Tea Leaves"

# Sweetener Constants
SWEETENER_JAGGERY_OR_SUGAR = "Jaggery or Sugar"
SWEETENER_HONEY_OR_JAGGERY = "Honey or Jaggery"
SWEETENER_HONEY_OR_SUGAR = "Honey or Sugar"

# Salt Constants
SALT = "Salt"

# Ground Spice Constants
SPICE_GROUND_GINGER = "Grated Fresh Ginger"
SPICE_GROUND_CINNAMON = "Ground Cinnamon"

# Whole Spice Constants
SPICE_WHOLE_CARDAMOM = "Green Cardamom Pods"
SPICE_WHOLE_CLOVES = "Whole Cloves"
SPICE_WHOLE_PEPPERCORNS = "Black Peppercorns"
SPICE_WHOLE_FENNEL = "Fennel Seeds"
SPICE_WHOLE_SAFFRON = "Saffron Threads"

# Herb Constants
HERB_MINT = "Fresh Mint Leaves"

# Floral Constants
FLORAL_ROSE_PETALS = "Dried Rose Petals"

# Citrus Constants
CITRUS_LEMON_JUICE = "Lemon Juice"

# Process Modifier Constants
PROCESS_BAKING_SODA = "Baking Soda"
PROCESS_ICE = "Ice"

# Garnish Constants
GARNISH_CRUSHED_NUTS = "Crushed Nuts"
GARNISH_ALMONDS = "Slivered Almonds"

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

# ============================================================================
# CHAI PREPARATION AND SCENARIO FRAMES
# ============================================================================

class CookingEquipmentInASceneFrame(BaseModel):
    """Frame to represent the *required* equipment for cooking."""

    # Core
    scene_type: str = Field(SCENE_HOME, description="Type of scene: home, campsite, or anywhere.")
    conditions: Optional[Set[str]] = Field(None, description="Environmental conditions (e.g., normal, windy, rainy, dark).")
    heat_source: Optional[str] = Field(None, description="Heat source for cooking (e.g., gas cooktop, propane stove).")
    cooking_vessel: Optional[str] = Field(None, description="Vessel used for cooking (e.g., pot, induction pot).")
    cooking_vessel_handling_tools: Optional[str] = Field(None, description="Tools for handling hot vessels (e.g., mittens, clamps).")
    cooking_platform: Optional[str] = Field(None, description="Platform for cooking setup (e.g., portable table).")
    ignition_tool: Optional[str] = Field(None, description="Tool for ignition (e.g., lighter, matches).")
    fuel_or_power_attachment: Optional[str] = Field(None, description="Fuel or power source (e.g., propane canister, battery).")
    fuel_connectors: Optional[str] = Field(None, description="Connectors for fuel (e.g., propane hose adapter).")

    # Environment Mitigation
    wind_mitigation_equipment: Optional[str] = Field(None, description="Equipment to mitigate wind (e.g., windshield).")
    rain_mitigation_equipment: Optional[str] = Field(None, description="Equipment to mitigate rain (e.g., rainfly, canopy).")
    lighting_equipment: Optional[str] = Field(None, description="Lighting equipment for dark conditions (e.g., lantern, headlamp).")

    def model_post_init(self, __context) -> None:
        """Set default values after initialization."""
        # Set default conditions
        if self.conditions is None:
            self.conditions = {CONDITION_NORMAL}

        # Set defaults based on scene type
        if self.scene_type in [SCENE_HOME, SCENE_ANY]:
            if not self.heat_source:
                self.heat_source = HEAT_GAS_COOKTOP
            if not self.cooking_vessel:
                self.cooking_vessel = VESSEL_POT

        if self.scene_type == SCENE_CAMPSITE:
            if not self.heat_source:
                self.heat_source = HEAT_PROPANE_STOVE
            if not self.cooking_vessel:
                self.cooking_vessel = VESSEL_POT
            if not self.fuel_or_power_attachment:
                self.fuel_or_power_attachment = FUEL_PROPANE_CANISTER

    def generate_description(self) -> str:
        """Generate a formatted description of cooking equipment for the scene.

        Casual relation will NOT be explicitly stated. E.g.

        CookingEquipmentInASceneFrame -
            scene_type='any'
            conditions={'normal'}
            heat_source='gas cooktop'
            cooking_vessel='pot with no handle'
            cooking_vessel_handling_tools='mittens'
            ........

        The fact about mittens being used as the pot has no handle will NOT be explicitly modeled.
        That will be left up to the LLMs to infer.
        """
        # Format scene type
        scene_str = f"at {self.scene_type}" if self.scene_type else "anywhere"

        # Format conditions
        if self.conditions:
            conditions_list = sorted(list(self.conditions))
            conditions_str = ", ".join(conditions_list)
        else:
            conditions_str = "normal"

        description = f"When cooking {scene_str}\n"
        description += f"Under {conditions_str} conditions\n"
        description += "Here are the tools and equipment that is needed -\n"

        # Collect all equipment
        equipment_list = []

        if self.heat_source:
            equipment_list.append(self.heat_source)
        if self.cooking_vessel:
            equipment_list.append(self.cooking_vessel)
        if self.cooking_vessel_handling_tools:
            equipment_list.append(self.cooking_vessel_handling_tools)
        if self.cooking_platform:
            equipment_list.append(self.cooking_platform)
        if self.ignition_tool:
            equipment_list.append(self.ignition_tool)
        if self.fuel_or_power_attachment:
            equipment_list.append(self.fuel_or_power_attachment)
        if self.fuel_connectors:
            equipment_list.append(self.fuel_connectors)
        if self.wind_mitigation_equipment:
            equipment_list.append(self.wind_mitigation_equipment)
        if self.rain_mitigation_equipment:
            equipment_list.append(self.rain_mitigation_equipment)
        if self.lighting_equipment:
            equipment_list.append(self.lighting_equipment)

        # Add numbered list
        for idx, equipment in enumerate(equipment_list, start=1):
            description += f"{idx}. {equipment}\n"

        return description
    
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
         a) Explicit Instruction: The recipe steps explicitly state the action (e.g., "stir the chai," "strain into a cup").
         b) Implied Action from Ingredient State: An ingredient's description implies the action. For example:
            - "peeled ginger" → `peel_ingredients=True`
            - "sliced lemon" → `slice_ingredients=True`
            - "crushed cardamom" → `crush_spices=True`
            - "grated turmeric" → `grate_ingredients=True`
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

        for i, action in enumerate(actions, 1):
            description += f"{i}. {action}\n"

        return description

class ChaiPrepToolingFrame(BaseModel):
    """Frame to represent the tools needed for chai preparation actions."""
    crushing_tools: Optional[list[str]] = Field(None, description="Tools for crushing spices: mortar and pestle, rolling pin, etc.")
    slicing_tools: Optional[list[str]] = Field(None, description="Tools for slicing ingredients.")
    chopping_tools: Optional[list[str]] = Field(None, description="Tools for chopping ingredients.")
    peeling_tools: Optional[list[str]] = Field(None, description="Tools for peeling ginger, citrus: peeler, knife, spoon, etc.")
    grinding_tools: Optional[list[str]] = Field(None, description="Tools for grinding spices just before brewing.")
    grating_tools: Optional[list[str]] = Field(None, description="Tools for grating aromatics like ginger or nutmeg.")
    stirring_tools: Optional[list[str]] = Field(None, description="Tools for mixing and stirring: spoon, ladle, whisk, etc.")
    straining_tools: Optional[list[str]] = Field(None, description="Tools for filtering tea: strainer, muslin cloth, tea filter, sieve, etc.")
    aerating_tools: Optional[list[str]] = Field(None, description="Tools for creating froth/aeration: whisk, deep ladle (for pulling), frother, etc.")

    def generate_description(self) -> str:
        """Generate a formatted description of preparation tools with their purposes."""
        description = "Preparation tools:\n"
        tool_count = 1

        if self.crushing_tools:
            tools_str = " or ".join(self.crushing_tools)
            description += f"{tool_count}. {tools_str} - for crushing spices\n"
            tool_count += 1

        if self.grinding_tools:
            tools_str = " or ".join(self.grinding_tools)
            description += f"{tool_count}. {tools_str} - for grinding spices just before brewing\n"
            tool_count += 1

        if self.slicing_tools:
            tools_str = " or ".join(self.slicing_tools)
            description += f"{tool_count}. {tools_str} - for slicing ingredients\n"
            tool_count += 1

        if self.chopping_tools:
            tools_str = " or ".join(self.chopping_tools)
            description += f"{tool_count}. {tools_str} - for chopping ingredients\n"
            tool_count += 1

        if self.peeling_tools:
            tools_str = " or ".join(self.peeling_tools)
            description += f"{tool_count}. {tools_str} - for peeling\n"
            tool_count += 1

        if self.grating_tools:
            tools_str = " or ".join(self.grating_tools)
            description += f"{tool_count}. {tools_str} - for grating aromatics\n"
            tool_count += 1

        if self.stirring_tools:
            tools_str = " or ".join(self.stirring_tools)
            description += f"{tool_count}. {tools_str} - for mixing and stirring\n"
            tool_count += 1

        if self.straining_tools:
            tools_str = " or ".join(self.straining_tools)
            description += f"{tool_count}. {tools_str} - for filtering tea\n"
            tool_count += 1

        if self.aerating_tools:
            tools_str = " or ".join(self.aerating_tools)
            description += f"{tool_count}. {tools_str} - for creating froth and aeration\n"
            tool_count += 1

        return description
   
class CookingEquipmentSceneFrameVariants:
    """Collection of cooking equipment scene variants."""

    def __init__(self) -> None:
        self.variants: List[CookingEquipmentInASceneFrame] = []
        self.init_default_variants()
        self.init_home_variants()
        self.init_campsite_variants()

    def init_default_variants(self) -> None:
        self.variants.append(CookingEquipmentInASceneFrame(
            scene_type=SCENE_ANY,
            cooking_vessel=VESSEL_POT_NO_HANDLE,
            cooking_vessel_handling_tools=HANDLING_TOOL_CLAMP
        ))
        self.variants.append(CookingEquipmentInASceneFrame(
            scene_type=SCENE_ANY,
            cooking_vessel=VESSEL_POT_NO_HANDLE,
            cooking_vessel_handling_tools=HANDLING_TOOL_MITTEN
        ))
        self.variants.append(CookingEquipmentInASceneFrame(
            scene_type=SCENE_ANY,
            conditions={CONDITION_LARGE_GROUP},
            cooking_vessel=VESSEL_STOCKPOT_SPIGOT
        ))

    def init_home_variants(self) -> None:
        """Initialize home cooking scenarios."""
        self.variants.append(CookingEquipmentInASceneFrame(
            scene_type=SCENE_HOME,
            heat_source=HEAT_GAS_COOKTOP,
            cooking_vessel=VESSEL_POT
        ))
        self.variants.append(CookingEquipmentInASceneFrame(
            scene_type=SCENE_HOME,
            heat_source=HEAT_INDUCTION_COOKTOP,
            cooking_vessel=VESSEL_INDUCTION_POT
        ))

    def init_campsite_variants(self) -> None:
        """Initialize campsite cooking scenarios."""
        self.variants.append(CookingEquipmentInASceneFrame(
            scene_type=SCENE_CAMPSITE,
            conditions={CONDITION_WINDY},
            wind_mitigation_equipment=WIND_WINDSHIELD
        ))
        self.variants.append(CookingEquipmentInASceneFrame(
            scene_type=SCENE_CAMPSITE,
            conditions={CONDITION_RAINY},
            rain_mitigation_equipment=RAIN_RAINFLY
        ))
        self.variants.append(CookingEquipmentInASceneFrame(
            scene_type=SCENE_CAMPSITE,
            conditions={CONDITION_RAINY},
            rain_mitigation_equipment=RAIN_CANOPY
        ))
        self.variants.append(CookingEquipmentInASceneFrame(
            scene_type=SCENE_CAMPSITE,
            conditions={CONDITION_DARK},
            lighting_equipment=LIGHTING_LANTERN
        ))
        self.variants.append(CookingEquipmentInASceneFrame(
            scene_type=SCENE_CAMPSITE,
            conditions={CONDITION_DARK},
            lighting_equipment=LIGHTING_HEADLAMP
        ))

        self.variants.append(CookingEquipmentInASceneFrame(
            scene_type=SCENE_CAMPSITE,
            heat_source=HEAT_BUTANE_STOVE,
            fuel_or_power_attachment=FUEL_BUTANE_CANISTER
        ))
        self.variants.append(CookingEquipmentInASceneFrame(
            scene_type=SCENE_CAMPSITE,
            heat_source=HEAT_PROPANE_STOVE,
            fuel_or_power_attachment=FUEL_PROPANE_TANK,
            fuel_connectors=CONNECTOR_PROPANE_HOSE_ADAPTER
        ))
        self.variants.append(CookingEquipmentInASceneFrame(
            scene_type=SCENE_CAMPSITE,
            heat_source=HEAT_PROPANE_STOVE,
            fuel_or_power_attachment=FUEL_PROPANE_CANISTER
        ))

        self.variants.append(CookingEquipmentInASceneFrame(
            scene_type=SCENE_CAMPSITE,
            heat_source=HEAT_INDUCTION_STOVE,
            cooking_vessel=VESSEL_INDUCTION_POT,
            fuel_or_power_attachment=FUEL_BATTERY
        ))

        self.variants.append(CookingEquipmentInASceneFrame(
            scene_type=SCENE_CAMPSITE,
            ignition_tool=IGNITION_LIGHTER
        ))
        self.variants.append(CookingEquipmentInASceneFrame(
            scene_type=SCENE_CAMPSITE,
            ignition_tool=IGNITION_MATCHES
        ))

        self.variants.append(CookingEquipmentInASceneFrame(
            scene_type=SCENE_CAMPSITE,
            cooking_platform=PLATFORM_PORTABLE_TABLE
        ))


    def get_scenes(self, scene_type: str) -> List[CookingEquipmentInASceneFrame]:
        """Get all scene variants for a specific scene type."""
        result = [
            frame for frame in self.variants if frame.scene_type in [scene_type, SCENE_ANY]
        ]

        if not len(result):
            raise LookupError(f"Invalid scene type specified: {scene_type}")

        return result

# ============================================================================
# HELPER CLASSES AND FUNCTIONS
# ============================================================================

class ChaiRecipe(BaseModel):
    """Class to represent an LLMs response to a request for chai recipe"""
    recipe_text: Optional[str] = Field(None, description="The raw text of the recipe")
    is_valid: bool = Field(True, description="A boolean flag set to indicate if the response is valid. Flag is set to false if the system failed or used asked for a request that was unrelated to a chai recipe.")

def generate_chai_tooling(frame: ChaiPreparationIngredientsActionsFrame) -> ChaiPrepToolingFrame:
    """Generate a ChaiPrepToolingFrame instance with tool lists based on the detected actions."""

    # Mapping from action flag → (frame field, tools)
    action_to_field_and_tools = {
        "crush_spices": ("crushing_tools", [TOOL_MORTAR_PESTLE, TOOL_ROLLING_PIN]),
        "grind_spices": ("grinding_tools", [TOOL_SPICE_GRINDER, TOOL_COFFEE_GRINDER]),
        "peel_ingredients": ("peeling_tools", [TOOL_PEELER, TOOL_PARING_KNIFE]),
        "slice_ingredients": ("slicing_tools", [TOOL_KNIFE, TOOL_MANDOLINE]),
        "grate_ingredients": ("grating_tools", [TOOL_GRATER, TOOL_MICROPLANE]),
        "chop_ingredients": ("chopping_tools", [TOOL_CHEFS_KNIFE, TOOL_CLEAVER]),
        "stir_chai": ("stirring_tools", [TOOL_SPOON, TOOL_LADLE]),
        "strain_chai": ("straining_tools", [TOOL_STRAINER, TOOL_MUSLIN_CLOTH]),
        "aerate_chai": ("aerating_tools", [TOOL_LADLE, TOOL_FROTHER]),
    }

    # Initialize frame fields
    tooling = ChaiPrepToolingFrame()

    # Assign tool lists based on which actions are true
    for action_flag, (field_name, tools) in action_to_field_and_tools.items():
        if getattr(frame, action_flag, False):
            setattr(tooling, field_name, tools)

    return tooling

# ============================================================================
# INSTANTIATED VARIANTS
# ============================================================================

SCENE_FRAMES_VARIANTS = CookingEquipmentSceneFrameVariants()
