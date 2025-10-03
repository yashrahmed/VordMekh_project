from dataclasses import dataclass
from typing import Set, Optional


# Scene Type Constants
SCENE_HOME = "home"
SCENE_CAMPSITE = "campsite"
SCENE_ANY = "any"

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


@dataclass
class CookingEquipmentInASceneFrame:
    """Frame to represent the *required* equipment for cooking."""

    # Core
    sceneType: str = SCENE_HOME
    conditions: Optional[Set[str]] = None
    heatSource: Optional[str] = None
    cookingVessel: Optional[str] = None
    cookingVesselHandlingTools: Optional[str] = None
    cookingPlatform: Optional[str] = None
    ignitionTool: Optional[str] = None
    fuelOrPowerAttachment: Optional[str] = None
    fuelConnectors: Optional[str] = None

    # Environment Mitigation
    windMitigationEquipment: Optional[str] = None
    rainMitigationEquipment: Optional[str] = None
    lightingEquipment: Optional[str] = None

    def __post_init__(self):
        """Set default value for conditions if None."""
        if self.conditions is None: self.conditions = {CONDITION_NORMAL}

        """Set the defaults procedurally"""
        if self.sceneType in [SCENE_HOME, SCENE_ANY]:
            if not self.heatSource: self.heatSource = HEAT_GAS_COOKTOP
            if not self.cookingVessel: self.cookingVessel = VESSEL_POT

        if self.sceneType == SCENE_CAMPSITE:
            if not self.heatSource: self.heatSource = HEAT_PROPANE_STOVE
            if not self.cookingVessel: self.cookingVessel = VESSEL_POT
            if not self.fuelOrPowerAttachment: self.fuelOrPowerAttachment = FUEL_PROPANE_CANISTER

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


class ChaiPreparationFrameVariants:
    def __init__(self) -> None:
        self.variants = []
        self.init_recipes()

    def init_recipes(self) -> None:
        # Masala Chai
        self.variants.append(ChaiPreparationFrame(
            chai_type="Masala Chai",
            liquids=[Ingredient("Water", 0.75, "cup"), Ingredient("Whole Milk", 0.5, "cup")],
            teas=[Ingredient("Loose Black Tea", 1, "tsp")],
            sweeteners=[Ingredient("Jaggery or Sugar", 1, "tsp")],
            spices_ground=[Ingredient("Grated Fresh Ginger", 0.5, "tsp"), Ingredient("Ground Cinnamon", 0.25, "tsp")],
            spices_whole=[Ingredient("Green Cardamom Pods", 3, "pods"), Ingredient("Whole Cloves", 2, "cloves"),
                          Ingredient("Black Peppercorns", 2, "peppercorns"), Ingredient("Fennel Seeds", 0.25, "tsp")],
            crushing_tools=["Mortar and Pestle"],
            peeling_tools=["Peeler"],
            straining_tools=["Strainer"]
        ))

        # Adrak Chai
        self.variants.append(ChaiPreparationFrame(
            chai_type="Adrak Chai",
            liquids=[Ingredient("Water", 0.75, "cup"), Ingredient("Whole Milk", 0.5, "cup")],
            teas=[Ingredient("Loose Black Tea", 1.5, "tsp")],
            sweeteners=[Ingredient("Jaggery or Sugar", 1, "tsp")],
            spices_ground=[Ingredient("Grated Fresh Ginger", 3, "tsp")],
            crushing_tools=["Mortar and Pestle"],
            peeling_tools=["Peeler"],
            straining_tools=["Strainer"]
        ))

        # Sulaimani Chai
        self.variants.append(ChaiPreparationFrame(
            chai_type="Sulaimani Chai",
            liquids=[Ingredient("Water", 1, "cup")],
            teas=[Ingredient("Loose Black Tea", 1, "tsp")],
            sweeteners=[Ingredient("Honey or Jaggery", 1, "tsp")],
            spices_whole=[Ingredient("Whole Cloves", 2, "cloves"), Ingredient("Green Cardamom Pods", 1, "pods"),
                          Ingredient("Saffron Threads", 5, "threads")],
            herbs=[Ingredient("Fresh Mint Leaves", 3, "leaves")],
            citrus=[Ingredient("Lemon Juice", 1, "tsp")],
            straining_tools=["Strainer"]
        ))

        # Kashmiri Chai
        self.variants.append(ChaiPreparationFrame(
            chai_type="Kashmiri Chai",
            liquids=[Ingredient("Water", 1.5, "cup"), Ingredient("Whole Milk", 0.75, "cup")],
            teas=[Ingredient("Kashmiri Green Tea Leaves", 1, "tsp")],
            salt=[Ingredient("Salt", 0.5, "tsp")],
            spices_whole=[Ingredient("Green Cardamom Pods", 1, "pods")],
            process_modifiers=[Ingredient("Baking Soda", 0.125, "tsp"), Ingredient("Ice", 0.5, "cup")],
            garnish=[Ingredient("Crushed Nuts", 0.5, "tbsp")],
            aerating_tools=["Whisk or Deep Ladle"],
            straining_tools=["Strainer"]
        ))

        # Kahwah
        self.variants.append(ChaiPreparationFrame(
            chai_type="Kahwah",
            liquids=[Ingredient("Water", 1, "cup")],
            teas=[Ingredient("Green Tea Leaves", 0.5, "tsp")],
            sweeteners=[Ingredient("Honey or Sugar", 1, "tsp")],
            spices_ground=[Ingredient("Ground Cinnamon", 0.125, "tsp")],
            spices_whole=[Ingredient("Green Cardamom Pods", 1, "pods"), Ingredient("Saffron Threads", 5, "threads")],
            floral=[Ingredient("Dried Rose Petals", 0.5, "tsp")],
            garnish=[Ingredient("Slivered Almonds", 0.5, "tbsp")],
            straining_tools=["Strainer"]
        ))


class CookingEquipmentSceneFrameVariants:
    def __init__(self) -> None:
        self.variants = []
        self.init_home_variants()
        self.init_campsite_variants()

    def init_home_variants(self) -> None:
        self.variants.append(CookingEquipmentInASceneFrame(sceneType=SCENE_HOME, heatSource=HEAT_GAS_COOKTOP, cookingVessel=VESSEL_POT))
        self.variants.append(CookingEquipmentInASceneFrame(sceneType=SCENE_HOME, heatSource=HEAT_INDUCTION_COOKTOP, cookingVessel=VESSEL_INDUCTION_POT))

    def init_campsite_variants(self) -> None:
       self.variants.append(CookingEquipmentInASceneFrame(sceneType=SCENE_CAMPSITE, conditions={CONDITION_WINDY}, windMitigationEquipment=WIND_WINDSHIELD))
       self.variants.append(CookingEquipmentInASceneFrame(sceneType=SCENE_CAMPSITE, conditions={CONDITION_RAINY}, rainMitigationEquipment=RAIN_RAINFLY))
       self.variants.append(CookingEquipmentInASceneFrame(sceneType=SCENE_CAMPSITE, conditions={CONDITION_RAINY}, rainMitigationEquipment=RAIN_CANOPY))
       self.variants.append(CookingEquipmentInASceneFrame(sceneType=SCENE_CAMPSITE, conditions={CONDITION_DARK}, lightingEquipment=LIGHTING_LANTERN))
       self.variants.append(CookingEquipmentInASceneFrame(sceneType=SCENE_CAMPSITE, conditions={CONDITION_DARK}, lightingEquipment=LIGHTING_HEADLAMP))

       self.variants.append(CookingEquipmentInASceneFrame(sceneType=SCENE_CAMPSITE, heatSource=HEAT_BUTANE_STOVE, fuelOrPowerAttachment=FUEL_BUTANE_CANISTER))
       self.variants.append(CookingEquipmentInASceneFrame(sceneType=SCENE_CAMPSITE, heatSource=HEAT_PROPANE_STOVE, fuelOrPowerAttachment=FUEL_PROPANE_TANK, fuelConnectors=CONNECTOR_PROPANE_HOSE_ADAPTER))
       self.variants.append(CookingEquipmentInASceneFrame(sceneType=SCENE_CAMPSITE, heatSource=HEAT_PROPANE_STOVE, fuelOrPowerAttachment=FUEL_PROPANE_CANISTER))

       self.variants.append(CookingEquipmentInASceneFrame(sceneType=SCENE_CAMPSITE, heatSource=HEAT_INDUCTION_STOVE, cookingVessel=VESSEL_INDUCTION_POT, fuelOrPowerAttachment=FUEL_BATTERY))

       self.variants.append(CookingEquipmentInASceneFrame(sceneType=SCENE_CAMPSITE, ignitionTool=IGNITION_LIGHTER))
       self.variants.append(CookingEquipmentInASceneFrame(sceneType=SCENE_CAMPSITE, ignitionTool=IGNITION_MATCHES))

       self.variants.append(CookingEquipmentInASceneFrame(sceneType=SCENE_CAMPSITE, cookingPlatform=PLATFORM_PORTABLE_TABLE))

       self.variants.append(CookingEquipmentInASceneFrame(sceneType=SCENE_ANY, cookingVessel=VESSEL_POT_NO_HANDLE, cookingVesselHandlingTools=HANDLING_TOOL_CLAMP))
       self.variants.append(CookingEquipmentInASceneFrame(sceneType=SCENE_ANY, cookingVessel=VESSEL_POT_NO_HANDLE, cookingVesselHandlingTools=HANDLING_TOOL_MITTEN))


if __name__ == '__main__':
    variants = CookingEquipmentSceneFrameVariants()

    print(variants.variants[-1])  

