from dataclasses import dataclass
from typing import List, Set, Optional

import re


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


@dataclass
class CookingEquipmentInASceneFrame:
    """Frame to represent the *required* equipment for cooking."""

    # Core
    scene_type: str = SCENE_HOME
    conditions: Optional[Set[str]] = None
    heat_source: Optional[str] = None
    cooking_vessel: Optional[str] = None
    cooking_vessel_handling_tools: Optional[str] = None
    cooking_platform: Optional[str] = None
    ignition_tool: Optional[str] = None
    fuel_or_power_attachment: Optional[str] = None
    fuel_connectors: Optional[str] = None

    # Environment Mitigation
    wind_mitigation_equipment: Optional[str] = None
    rain_mitigation_equipment: Optional[str] = None
    lighting_equipment: Optional[str] = None

    def __post_init__(self):
        """Set default value for conditions if None."""
        if self.conditions is None: self.conditions = {CONDITION_NORMAL}

        """Set the defaults procedurally"""
        if self.scene_type in [SCENE_HOME, SCENE_ANY]:
            if not self.heat_source: self.heat_source = HEAT_GAS_COOKTOP
            if not self.cooking_vessel: self.cooking_vessel = VESSEL_POT

        if self.scene_type == SCENE_CAMPSITE:
            if not self.heat_source: self.heat_source = HEAT_PROPANE_STOVE
            if not self.cooking_vessel: self.cooking_vessel = VESSEL_POT
            if not self.fuel_or_power_attachment: self.fuel_or_power_attachment = FUEL_PROPANE_CANISTER

    def generate_description(self):
        """Generate a formatted description of cooking equipment for the scene.

            Casual relation will NOT be explicitly stated. E.g.
            
            CookingEquipmentInASceneFrame -
                scene_type='any'
                conditions={'normal'}
                heat_source='gas cooktop'
                cooking_vessel='pot with no handle'
                cooking_vessel_handling_tools='mittens'
                ........
            
            The fact about mittens being used as the pot has no handle will NOT be explictly modeled.
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

@dataclass
class Ingredient:
    """Represents an ingredient with name, amount, and unit."""
    name: str
    amount: Optional[float] = None
    unit: Optional[str] = None

@dataclass
class ChaiPreparationFrame:
    """Frame to represent the ingredients and tools for chai preparation."""
    chai_type: str  # The name/type of chai recipe (e.g., "Masala Chai", "Adrak Chai", "Sulaimani Chai")
    # Liquids & tea
    liquids: Optional[list[Ingredient]] = None  # Base liquids like water, milk (dairy or plant-based)
    teas: Optional[list[Ingredient]] = None  # Tea leaves: black tea, green tea, kashmiri green tea, etc.

    # Sweeteners & seasoning
    sweeteners: Optional[list[Ingredient]] = None  # Sweetening agents: sugar, jaggery, honey, etc.
    salt: Optional[list[Ingredient]] = None  # Salt for savory chai variants (e.g., noon chai)

    # Spices
    spices_ground: Optional[list[Ingredient]] = None  # Pre-ground spices: ginger powder, cinnamon powder, etc.
    spices_whole: Optional[list[Ingredient]] = None  # Whole spices: cardamom pods, cloves, peppercorns, fennel seeds, saffron threads

    # Herbs / floral / citrus
    herbs: Optional[list[Ingredient]] = None  # Fresh or dried herbs: mint leaves, tulsi, lemongrass, etc.
    floral: Optional[list[Ingredient]] = None  # Floral additions: rose petals, jasmine, etc.
    citrus: Optional[list[Ingredient]] = None  # Citrus elements: lemon juice, orange zest, etc.

    # Process modifiers
    process_modifiers: Optional[list[Ingredient]] = None  # Ingredients that modify cooking process: baking soda (for pink chai), ice (for iced chai)

    # Garnish
    garnish: Optional[list[Ingredient]] = None  # Toppings and finishing touches: crushed nuts, slivered almonds, spice powder, cream, etc.

    # Actions (tool mappings)
    crushing_tools: Optional[list[str]] = None  # Tools for crushing spices: mortar and pestle, rolling pin, etc.
    peeling_tools: Optional[list[str]] = None  # Tools for peeling ginger, citrus: peeler, knife, spoon, etc.
    stirring_tools: Optional[list[str]] = None  # Tools for mixing and stirring: spoon, ladle, whisk, etc.
    straining_tools: Optional[list[str]] = None  # Tools for filtering tea: strainer, muslin cloth, tea filter, sieve, etc.
    aerating_tools: Optional[list[str]] = None  # Tools for creating froth/aeration: whisk, deep ladle (for pulling), frother, etc.

    def generate_description(self):
        """Generate a formatted description of ingredients and tools."""
        description = "Ingredients:\n"

        ingredient_count = 1

        # Helper function to add ingredients from a list
        def add_ingredients(ingredient_list):
            nonlocal ingredient_count
            if ingredient_list:
                for ing in ingredient_list:
                    if ing.amount and ing.unit:
                        description_line = f"{ingredient_count}. {ing.amount} {ing.unit} of {ing.name}\n"
                    elif ing.amount:
                        description_line = f"{ingredient_count}. {ing.amount} of {ing.name}\n"
                    else:
                        description_line = f"{ingredient_count}. {ing.name}\n"
                    description_lines.append(description_line)
                    ingredient_count += 1

        description_lines = []

        # Add all ingredient categories
        add_ingredients(self.liquids)
        add_ingredients(self.teas)
        add_ingredients(self.sweeteners)
        add_ingredients(self.salt)
        add_ingredients(self.spices_ground)
        add_ingredients(self.spices_whole)
        add_ingredients(self.herbs)
        add_ingredients(self.floral)
        add_ingredients(self.citrus)
        add_ingredients(self.process_modifiers)
        add_ingredients(self.garnish)

        description += ''.join(description_lines)

        # Add tools section
        description += "\nPreparation tools:\n"
        tool_count = 1
        all_tools = []

        if self.crushing_tools:
            all_tools.extend(self.crushing_tools)
        if self.peeling_tools:
            all_tools.extend(self.peeling_tools)
        if self.stirring_tools:
            all_tools.extend(self.stirring_tools)
        if self.straining_tools:
            all_tools.extend(self.straining_tools)
        if self.aerating_tools:
            all_tools.extend(self.aerating_tools)

        for tool in all_tools:
            description += f"{tool_count}. {tool}\n"
            tool_count += 1

        return description


class ChaiPreparationFrameVariants:
    def __init__(self) -> None:
        self.variants: List[ChaiPreparationFrame] = []
        self.init_recipes()

    def init_recipes(self) -> None:
        # Masala Chai
        self.variants.append(ChaiPreparationFrame(
            chai_type=CHAI_MASALA,
            liquids=[Ingredient(LIQUID_WATER, 0.75, "cup"), Ingredient(LIQUID_WHOLE_MILK, 0.5, "cup")],
            teas=[Ingredient(TEA_LOOSE_BLACK, 1, "tsp")],
            sweeteners=[Ingredient(SWEETENER_JAGGERY_OR_SUGAR, 1, "tsp")],
            spices_ground=[Ingredient(SPICE_GROUND_GINGER, 0.5, "tsp"), Ingredient(SPICE_GROUND_CINNAMON, 0.25, "tsp")],
            spices_whole=[Ingredient(SPICE_WHOLE_CARDAMOM, 3, "pods"), Ingredient(SPICE_WHOLE_CLOVES, 2, "cloves"),
                          Ingredient(SPICE_WHOLE_PEPPERCORNS, 2, "peppercorns"), Ingredient(SPICE_WHOLE_FENNEL, 0.25, "tsp")],
            crushing_tools=[TOOL_MORTAR_PESTLE],
            peeling_tools=[TOOL_PEELER],
            straining_tools=[TOOL_STRAINER]
        ))

        # Adrak Chai
        self.variants.append(ChaiPreparationFrame(
            chai_type=CHAI_ADRAK,
            liquids=[Ingredient(LIQUID_WATER, 0.75, "cup"), Ingredient(LIQUID_WHOLE_MILK, 0.5, "cup")],
            teas=[Ingredient(TEA_LOOSE_BLACK, 1.5, "tsp")],
            sweeteners=[Ingredient(SWEETENER_JAGGERY_OR_SUGAR, 1, "tsp")],
            spices_ground=[Ingredient(SPICE_GROUND_GINGER, 3, "tsp")],
            crushing_tools=[TOOL_MORTAR_PESTLE],
            peeling_tools=[TOOL_PEELER],
            straining_tools=[TOOL_STRAINER]
        ))

        # Sulaimani Chai
        self.variants.append(ChaiPreparationFrame(
            chai_type=CHAI_SULAIMANI,
            liquids=[Ingredient(LIQUID_WATER, 1, "cup")],
            teas=[Ingredient(TEA_LOOSE_BLACK, 1, "tsp")],
            sweeteners=[Ingredient(SWEETENER_HONEY_OR_JAGGERY, 1, "tsp")],
            spices_whole=[Ingredient(SPICE_WHOLE_CLOVES, 2, "cloves"), Ingredient(SPICE_WHOLE_CARDAMOM, 1, "pods"),
                          Ingredient(SPICE_WHOLE_SAFFRON, 5, "threads")],
            herbs=[Ingredient(HERB_MINT, 3, "leaves")],
            citrus=[Ingredient(CITRUS_LEMON_JUICE, 1, "tsp")],
            straining_tools=[TOOL_STRAINER]
        ))

        # Kashmiri Chai
        self.variants.append(ChaiPreparationFrame(
            chai_type=CHAI_KASHMIRI,
            liquids=[Ingredient(LIQUID_WATER, 1.5, "cup"), Ingredient(LIQUID_WHOLE_MILK, 0.75, "cup")],
            teas=[Ingredient(TEA_KASHMIRI_GREEN, 1, "tsp")],
            salt=[Ingredient(SALT, 0.5, "tsp")],
            spices_whole=[Ingredient(SPICE_WHOLE_CARDAMOM, 1, "pods")],
            process_modifiers=[Ingredient(PROCESS_BAKING_SODA, 0.125, "tsp"), Ingredient(PROCESS_ICE, 0.5, "cup")],
            garnish=[Ingredient(GARNISH_CRUSHED_NUTS, 0.5, "tbsp")],
            aerating_tools=[TOOL_WHISK_LADLE],
            straining_tools=[TOOL_STRAINER]
        ))

        # Kahwah
        self.variants.append(ChaiPreparationFrame(
            chai_type=CHAI_KAHWAH,
            liquids=[Ingredient(LIQUID_WATER, 1, "cup")],
            teas=[Ingredient(TEA_GREEN, 0.5, "tsp")],
            sweeteners=[Ingredient(SWEETENER_HONEY_OR_SUGAR, 1, "tsp")],
            spices_ground=[Ingredient(SPICE_GROUND_CINNAMON, 0.125, "tsp")],
            spices_whole=[Ingredient(SPICE_WHOLE_CARDAMOM, 1, "pods"), Ingredient(SPICE_WHOLE_SAFFRON, 5, "threads")],
            floral=[Ingredient(FLORAL_ROSE_PETALS, 0.5, "tsp")],
            garnish=[Ingredient(GARNISH_ALMONDS, 0.5, "tbsp")],
            straining_tools=[TOOL_STRAINER]
        ))

    def get_recipe(self, chai_type):
        for frame in self.variants:
            if frame.chai_type == chai_type:
                return frame
        
        raise LookupError("Invalid chai type specified!!")


class CookingEquipmentSceneFrameVariants:
    def __init__(self) -> None:
        self.variants: List[CookingEquipmentInASceneFrame] = []
        self.init_home_variants()
        self.init_campsite_variants()

    def init_home_variants(self) -> None:
        self.variants.append(CookingEquipmentInASceneFrame(scene_type=SCENE_HOME, heat_source=HEAT_GAS_COOKTOP, cooking_vessel=VESSEL_POT))
        self.variants.append(CookingEquipmentInASceneFrame(scene_type=SCENE_HOME, heat_source=HEAT_INDUCTION_COOKTOP, cooking_vessel=VESSEL_INDUCTION_POT))

    def init_campsite_variants(self) -> None:
       self.variants.append(CookingEquipmentInASceneFrame(scene_type=SCENE_CAMPSITE, conditions={CONDITION_WINDY}, wind_mitigation_equipment=WIND_WINDSHIELD))
       self.variants.append(CookingEquipmentInASceneFrame(scene_type=SCENE_CAMPSITE, conditions={CONDITION_RAINY}, rain_mitigation_equipment=RAIN_RAINFLY))
       self.variants.append(CookingEquipmentInASceneFrame(scene_type=SCENE_CAMPSITE, conditions={CONDITION_RAINY}, rain_mitigation_equipment=RAIN_CANOPY))
       self.variants.append(CookingEquipmentInASceneFrame(scene_type=SCENE_CAMPSITE, conditions={CONDITION_DARK}, lighting_equipment=LIGHTING_LANTERN))
       self.variants.append(CookingEquipmentInASceneFrame(scene_type=SCENE_CAMPSITE, conditions={CONDITION_DARK}, lighting_equipment=LIGHTING_HEADLAMP))

       self.variants.append(CookingEquipmentInASceneFrame(scene_type=SCENE_CAMPSITE, heat_source=HEAT_BUTANE_STOVE, fuel_or_power_attachment=FUEL_BUTANE_CANISTER))
       self.variants.append(CookingEquipmentInASceneFrame(scene_type=SCENE_CAMPSITE, heat_source=HEAT_PROPANE_STOVE, fuel_or_power_attachment=FUEL_PROPANE_TANK, fuel_connectors=CONNECTOR_PROPANE_HOSE_ADAPTER))
       self.variants.append(CookingEquipmentInASceneFrame(scene_type=SCENE_CAMPSITE, heat_source=HEAT_PROPANE_STOVE, fuel_or_power_attachment=FUEL_PROPANE_CANISTER))

       self.variants.append(CookingEquipmentInASceneFrame(scene_type=SCENE_CAMPSITE, heat_source=HEAT_INDUCTION_STOVE, cooking_vessel=VESSEL_INDUCTION_POT, fuel_or_power_attachment=FUEL_BATTERY))

       self.variants.append(CookingEquipmentInASceneFrame(scene_type=SCENE_CAMPSITE, ignition_tool=IGNITION_LIGHTER))
       self.variants.append(CookingEquipmentInASceneFrame(scene_type=SCENE_CAMPSITE, ignition_tool=IGNITION_MATCHES))

       self.variants.append(CookingEquipmentInASceneFrame(scene_type=SCENE_CAMPSITE, cooking_platform=PLATFORM_PORTABLE_TABLE))

       self.variants.append(CookingEquipmentInASceneFrame(scene_type=SCENE_ANY, cooking_vessel=VESSEL_POT_NO_HANDLE, cooking_vessel_handling_tools=HANDLING_TOOL_CLAMP))
       self.variants.append(CookingEquipmentInASceneFrame(scene_type=SCENE_ANY, cooking_vessel=VESSEL_POT_NO_HANDLE, cooking_vessel_handling_tools=HANDLING_TOOL_MITTEN))

    def get_scenes(self, scene_type):
        result = [
            frame for frame in self.variants if frame.scene_type == scene_type
        ]

        if not len(result):
            raise LookupError("Invalid scene type specified!!")
        
        return result


def think_through_scenarios_for_chai(chai_type, scene_type):
    equip_frames = CookingEquipmentSceneFrameVariants()
    prep_frames = ChaiPreparationFrameVariants()
    recipe = prep_frames.get_recipe(chai_type)
    
    scene_variants = []
    scene_variants.extend(equip_frames.get_scenes(SCENE_ANY))
    scene_variants.extend(equip_frames.get_scenes(scene_type))

    sep_string = '\n' + '_' * 30 + '\n\n'
    scene_desc_str = sep_string.join([scene.generate_description() for scene in scene_variants])

    # Build a "chain of thought about the different scenarios" here....

    combined_cot_str = f"""
        Here are the things that are needed to prepare {chai_type}.

        Let's start with the ingredients -

        {recipe.generate_description()}
        And here are the different cooking scenarios where preparation may occur.

        {scene_desc_str}
    """
    combined_cot_str= '\n'.join([re.sub(r'^[\s^\n]+', '' ,line) for line in combined_cot_str.split('\n')])
    print(combined_cot_str)

    return combined_cot_str
    


if __name__ == '__main__':
    think_through_scenarios_for_chai(CHAI_MASALA, SCENE_CAMPSITE)

