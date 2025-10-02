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

