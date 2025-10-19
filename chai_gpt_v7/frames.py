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
TOOL_MORTAR_PESTLE = "Mortar and Pestle"
TOOL_PEELER = "Peeler"
TOOL_STRAINER = "Strainer"
TOOL_WHISK_LADLE = "Whisk or Deep Ladle"


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

class Ingredient(BaseModel):
    """
    A single ingredient item.
    """
    name: str = Field(..., description="Canonical name of the ingredient (e.g., 'milk', 'black tea').")
    quantity: Optional[float] = Field(None, description="Numeric quantity if present (e.g., 0.75).")
    unit: Optional[str] = Field(None, description="Unit string as written or normalized (e.g., 'cup', 'teaspoon').")

class ChaiPreparationIngredientsActionsFrame(BaseModel):
    """
    Frame to extract chai ingredients and infer preparation actions from the recipe text.

    Fill procedure (step-by-step):
    1) Identify chai type:
       - If the recipe names a style (e.g., “Masala Chai”), set `chai_type` to that.
       - Otherwise use a concise descriptive name (e.g., “Milk tea with spices”).

    2) Parse ingredient lines and place each into the most specific slot:
       - liquids      → water, milk (dairy/plant), etc.
       - teas         → tea leaves/bags by type.
       - sweeteners   → sugar, jaggery, honey, etc.
       - salt         → only for savory variants.
       - aromatics    → all spices (whole/ground), fresh items (ginger), herbs, florals, and citrus.
       - process_modifiers → ingredients that alter process/chemistry (e.g., baking soda, ice).
       - garnish      → toppings/finishers.

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

    # Core components
    liquids: Optional[List[Ingredient]] = Field(None, description="Base liquids (e.g., water, milk—dairy or plant).")
    teas: Optional[List[Ingredient]] = Field(None, description="Tea leaves/bags by type (e.g., 'black tea', 'green tea').")

    # Flavoring components
    sweeteners: Optional[List[Ingredient]] = Field(None, description="Sweetening agents as listed (e.g., sugar, jaggery, honey).")
    salt: Optional[List[Ingredient]] = Field(None, description="Salt only for savory variants (e.g., noon chai).")
    aromatics: Optional[List[Ingredient]] = Field(None, description="All spices (whole/ground), fresh items (ginger), herbs, florals, and citrus elements.")
    
    # Process modifiers and Garnish
    process_modifiers: Optional[List[Ingredient]] = Field(None, description="Items that modify process/chemistry/temperature (e.g., baking soda, ice).")
    garnish: Optional[List[Ingredient]] = Field(None, description="Toppings/finishers added after cooking (e.g., crushed nuts, cream).")

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

        # Helper function to add ingredients from a list
        def add_ingredients(ingredient_list):
            nonlocal ingredient_count
            if ingredient_list:
                for ing in ingredient_list:
                    if ing.quantity and ing.unit:
                        description_line = f"{ingredient_count}. {ing.quantity} {ing.unit} of {ing.name}\n"
                    elif ing.quantity:
                        description_line = f"{ingredient_count}. {ing.quantity} of {ing.name}\n"
                    else:
                        description_line = f"{ingredient_count}. {ing.name}\n"
                    description_lines.append(description_line)
                    ingredient_count += 1
        
        # Add all ingredient categories
        add_ingredients(self.liquids)
        add_ingredients(self.teas)
        add_ingredients(self.sweeteners)
        add_ingredients(self.salt)
        add_ingredients(self.aromatics)
        add_ingredients(self.process_modifiers)
        add_ingredients(self.garnish)

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
    peeling_tools: Optional[list[str]] = Field(None, description="Tools for peeling ginger, citrus: peeler, knife, spoon, etc.")
    stirring_tools: Optional[list[str]] = Field(None, description="Tools for mixing and stirring: spoon, ladle, whisk, etc.")
    straining_tools: Optional[list[str]] = Field(None, description="Tools for filtering tea: strainer, muslin cloth, tea filter, sieve, etc.")
    aerating_tools: Optional[list[str]] = Field(None, description="Tools for creating froth/aeration: whisk, deep ladle (for pulling), frother, etc.")

    def generate_tools_description(self) -> str:
        """Generate a formatted description of preparation tools with their purposes."""
        description = "Preparation tools:\n"
        tool_count = 1

        if self.crushing_tools:
            for tool in self.crushing_tools:
                description += f"{tool_count}. {tool} - for crushing spices\n"
                tool_count += 1

        if self.peeling_tools:
            for tool in self.peeling_tools:
                description += f"{tool_count}. {tool} - for peeling\n"
                tool_count += 1

        if self.stirring_tools:
            for tool in self.stirring_tools:
                description += f"{tool_count}. {tool} - for mixing and stirring\n"
                tool_count += 1

        if self.straining_tools:
            for tool in self.straining_tools:
                description += f"{tool_count}. {tool} - for filtering tea\n"
                tool_count += 1

        if self.aerating_tools:
            for tool in self.aerating_tools:
                description += f"{tool_count}. {tool} - for creating froth and aeration\n"
                tool_count += 1

        return description
   
class ChaiPreparationIngredientsActionsFrameVariants:
    """Collection of chai recipe variants using ChaiPreparationIngredientsActionsFrame."""

    def __init__(self) -> None:
        self.variants: list[ChaiPreparationIngredientsActionsFrame] = []
        self.init_recipes()

    def init_recipes(self) -> None:
        # Masala Chai
        self.variants.append(ChaiPreparationIngredientsActionsFrame(
            chai_type=CHAI_MASALA,
            liquids=[
                Ingredient(name=LIQUID_WATER, quantity=0.75, unit="cup"),
                Ingredient(name=LIQUID_WHOLE_MILK, quantity=0.5, unit="cup")
            ],
            teas=[Ingredient(name=TEA_LOOSE_BLACK, quantity=1, unit="tsp")],
            sweeteners=[Ingredient(name=SWEETENER_JAGGERY_OR_SUGAR, quantity=1, unit="tsp")],
            salt=None,
            spices_ground=[
                Ingredient(name=SPICE_GROUND_GINGER, quantity=0.5, unit="tsp"),
                Ingredient(name=SPICE_GROUND_CINNAMON, quantity=0.25, unit="tsp")
            ],
            spices_whole=[
                Ingredient(name=SPICE_WHOLE_CARDAMOM, quantity=3, unit="pods"),
                Ingredient(name=SPICE_WHOLE_CLOVES, quantity=2, unit="cloves"),
                Ingredient(name=SPICE_WHOLE_PEPPERCORNS, quantity=2, unit="peppercorns"),
                Ingredient(name=SPICE_WHOLE_FENNEL, quantity=0.25, unit="tsp")
            ],
            herbs=None,
            floral=None,
            citrus=None,
            process_modifiers=None,
            garnish=None,
            crush_spices=True,
            grind_spices=None,
            peel_ingredients=True,
            slice_ingredients=None,
            stir_chai=None,
            strain_chai=True,
            aerate_chai=None
        ))

        # Adrak Chai
        self.variants.append(ChaiPreparationIngredientsActionsFrame(
            chai_type=CHAI_ADRAK,
            liquids=[
                Ingredient(name=LIQUID_WATER, quantity=0.75, unit="cup"),
                Ingredient(name=LIQUID_WHOLE_MILK, quantity=0.5, unit="cup")
            ],
            teas=[Ingredient(name=TEA_LOOSE_BLACK, quantity=1.5, unit="tsp")],
            sweeteners=[Ingredient(name=SWEETENER_JAGGERY_OR_SUGAR, quantity=1, unit="tsp")],
            salt=None,
            spices_ground=[Ingredient(name=SPICE_GROUND_GINGER, quantity=3, unit="tsp")],
            spices_whole=None,
            herbs=None,
            floral=None,
            citrus=None,
            process_modifiers=None,
            garnish=None,
            crush_spices=True,
            grind_spices=None,
            peel_ingredients=True,
            slice_ingredients=None,
            stir_chai=None,
            strain_chai=True,
            aerate_chai=None
        ))

        # Sulaimani Chai
        self.variants.append(ChaiPreparationIngredientsActionsFrame(
            chai_type=CHAI_SULAIMANI,
            liquids=[Ingredient(name=LIQUID_WATER, quantity=1, unit="cup")],
            teas=[Ingredient(name=TEA_LOOSE_BLACK, quantity=1, unit="tsp")],
            sweeteners=[Ingredient(name=SWEETENER_HONEY_OR_JAGGERY, quantity=1, unit="tsp")],
            salt=None,
            spices_ground=None,
            spices_whole=[
                Ingredient(name=SPICE_WHOLE_CLOVES, quantity=2, unit="cloves"),
                Ingredient(name=SPICE_WHOLE_CARDAMOM, quantity=1, unit="pods"),
                Ingredient(name=SPICE_WHOLE_SAFFRON, quantity=5, unit="threads")
            ],
            herbs=[Ingredient(name=HERB_MINT, quantity=3, unit="leaves")],
            floral=None,
            citrus=[Ingredient(name=CITRUS_LEMON_JUICE, quantity=1, unit="tsp")],
            process_modifiers=None,
            garnish=None,
            crush_spices=None,
            grind_spices=None,
            peel_ingredients=None,
            slice_ingredients=None,
            stir_chai=None,
            strain_chai=True,
            aerate_chai=None
        ))

        # Kashmiri Chai
        self.variants.append(ChaiPreparationIngredientsActionsFrame(
            chai_type=CHAI_KASHMIRI,
            liquids=[
                Ingredient(name=LIQUID_WATER, quantity=1.5, unit="cup"),
                Ingredient(name=LIQUID_WHOLE_MILK, quantity=0.75, unit="cup")
            ],
            teas=[Ingredient(name=TEA_KASHMIRI_GREEN, quantity=1, unit="tsp")],
            sweeteners=None,
            salt=[Ingredient(name=SALT, quantity=0.5, unit="tsp")],
            spices_ground=None,
            spices_whole=[Ingredient(name=SPICE_WHOLE_CARDAMOM, quantity=1, unit="pods")],
            herbs=None,
            floral=None,
            citrus=None,
            process_modifiers=[
                Ingredient(name=PROCESS_BAKING_SODA, quantity=0.125, unit="tsp"),
                Ingredient(name=PROCESS_ICE, quantity=0.5, unit="cup")
            ],
            garnish=[Ingredient(name=GARNISH_CRUSHED_NUTS, quantity=0.5, unit="tbsp")],
            crush_spices=None,
            grind_spices=None,
            peel_ingredients=None,
            slice_ingredients=None,
            stir_chai=None,
            strain_chai=True,
            aerate_chai=True
        ))

        # Kahwah
        self.variants.append(ChaiPreparationIngredientsActionsFrame(
            chai_type=CHAI_KAHWAH,
            liquids=[Ingredient(name=LIQUID_WATER, quantity=1, unit="cup")],
            teas=[Ingredient(name=TEA_GREEN, quantity=0.5, unit="tsp")],
            sweeteners=[Ingredient(name=SWEETENER_HONEY_OR_SUGAR, quantity=1, unit="tsp")],
            salt=None,
            spices_ground=[Ingredient(name=SPICE_GROUND_CINNAMON, quantity=0.125, unit="tsp")],
            spices_whole=[
                Ingredient(name=SPICE_WHOLE_CARDAMOM, quantity=1, unit="pods"),
                Ingredient(name=SPICE_WHOLE_SAFFRON, quantity=5, unit="threads")
            ],
            herbs=None,
            floral=[Ingredient(name=FLORAL_ROSE_PETALS, quantity=0.5, unit="tsp")],
            citrus=None,
            process_modifiers=None,
            garnish=[Ingredient(name=GARNISH_ALMONDS, quantity=0.5, unit="tbsp")],
            crush_spices=None,
            grind_spices=None,
            peel_ingredients=None,
            slice_ingredients=None,
            stir_chai=None,
            strain_chai=True,
            aerate_chai=None
        ))

    def get_recipe(self, chai_type: str) -> ChaiPreparationIngredientsActionsFrame:
        """Get a recipe by chai type."""
        for frame in self.variants:
            if frame.chai_type == chai_type:
                return frame

        raise LookupError(f"Invalid chai type specified: {chai_type}")

class CookingEquipmentSceneFrameVariants:
    """Collection of cooking equipment scene variants."""

    def __init__(self) -> None:
        self.variants: List[CookingEquipmentInASceneFrame] = []
        self.init_home_variants()
        self.init_campsite_variants()

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

    def get_scenes(self, scene_type: str) -> List[CookingEquipmentInASceneFrame]:
        """Get all scene variants for a specific scene type."""
        result = [
            frame for frame in self.variants if frame.scene_type == scene_type
        ]

        if not len(result):
            raise LookupError(f"Invalid scene type specified: {scene_type}")

        return result

