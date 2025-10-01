Parent Frame: CookingTooling

Purpose: Frame to represent the available tools in a cooking area.

Core
	•	scene (home | campsite)
	•	conditions (Set: wind | rain | dark | cold | highAltitude | none)
	•	heatSource (gasStove | inductionHob | butaneStove)
	•	cookingVessel (pot | saucepan | kettleBody)
	•	surface (countertop | table | portableSurface)
	•	ignition (piezo | lighter | matches | none)
	•	fuelOrPower (butaneCanister | LPG | electricity | none)

Environment Mitigation
	•	windMitigation (windshield | none)
	•	rainMitigation (rainfly | canopy | none)
	•	lighting (lantern | headlamp | none)

⸻

Parent Frame: ChaiTooling

Purpose: Common slots for chai prep across variants.

CookingAreaASetup [CookingTooling]

Preparation
	•	aromaticsPrep (Set: peeler | grater | mortarAndPestle)
	•	measurement (Set: teaspoon | tablespoon | measuringCup | scale)
	•	stirring (spoon | ladle | whisk)

Aeration
	•	aerationTools (whisk | ladle | none)

Straining & Serving
	•	strainingTool (strainer | muslin | decant)
	•	servingVessels (Set: cup | mug)
	•	handling (mitts | handleOnly | trivet)

Consumables & Waste
	•	waterSource (tap | bottle | jug)
	•	sweetenerContainer (jaggeryJar | sugarJar | saltContainer | none)
	•	wasteHandling (trashcan | trashbag | packOut)

⸻

Child Frame: KashmiriChaiTooling (inherits ChaiTooling)

Adds the rare “shock” step.

Preparation
	•	shockingTools (ice | coldWater | none) [Additional slot]

