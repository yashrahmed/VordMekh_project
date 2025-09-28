Top-Level Task

Compound Task: prepare-masala-chai
Description: Prepare Masala Chai, from prep to cleanup.

Method: general-preparation
Subtasks (ordered):
	1.	prepare-cooking-area
	2.	prepare-aromatics
	3.	build-base-in-pot
	4.	add-tea-and-steep
	5.	add-milk-and-heat
	6.	strain-chai
	7.	sweeten-chai
	8.	cleanup

⸻

Compound Task: prepare-cooking-area

Description: Prepare the cooking area so chai can be brewed safely and effectively under varying outdoor or indoor conditions.

Method: home-default
Subtasks:
	1.	no-action-required (primitive)

Method: campsite-setup
Subtasks (ordered, conditional):
	1.	prep-cooking-surface (compound)
	2.	prepare-stove (compound)
	3.	prep-rain-cover (if raining/expected) (primitive)
	•	Deploy rainfly/canopy to protect stove and work area.
	4.	prep-wind-cover (if windy) (primitive)
	•	Position windshield or barrier; orient stove downwind.
	5.	prep-lighting (if dark/low light) (primitive)
	•	Place lantern/headlamp; ensure light reaches pot/work surface.

⸻

Compound Task: prep-cooking-surface

Description: Prepare or select a cooking surface suitable for placing the stove and pot.

Method: home-default
Subtasks:
	1.	no-action-required (primitive)

Method: campsite-setup
Subtasks (ordered):
	1.	locate-or-carry-surface (primitive)
	2.	stabilize-surface (primitive)

⸻

Compound Task: prepare-stove

Description: Ready the stove for chai preparation.

Method: home-default
Subtasks:
	1.	no-action-required (primitive)

Method: campsite-butane
Subtasks (ordered):
	1.	attach-fuel-canister (primitive)
	2.	check-for-leaks (primitive)
	3.	stabilize-stove-base (primitive)
	4.	verify-burner-control (primitive)

⸻

Compound Task: prepare-aromatics

Description: Ready ginger and spices.

Method: default-prep
Subtasks:
	1.	peel-ginger (primitive)
	2.	grate-ginger (primitive)
	3.	measure-ginger (primitive)
	4.	crush-spices (primitive)
	5.	measure-cinnamon (primitive)

⸻

Compound Task: build-base-in-pot

Description: Create spiced water base.

Method: default-build
Subtasks:
	1.	add-water (primitive)
	2.	add-spices (primitive)
	3.	stir-mixture (primitive)
	4.	boil-and-simmer (primitive)

⸻

Compound Task: add-tea-and-steep

Description: Infuse tea.

Method: default-steep
Subtasks:
	1.	measure-tea (primitive)
	2.	add-tea (primitive)
	3.	stir-mixture (primitive)

⸻

Compound Task: add-milk-and-heat

Description: Incorporate milk.

Method: default-heat
Subtasks:
	1.	measure-milk (primitive)
	2.	add-milk (primitive)

⸻

Compound Task: strain-chai

Description: Strain chai into cups.

Method: default-strain
Subtasks:
	1.	turn-off-stove (primitive)
	2.	find-straining-tool (compound)
	3.	strain-to-cups (primitive)

⸻

Compound Task: find-straining-tool

Description: Select tool for straining.

Method: strainer-available
Subtasks:
	1.	use-strainer (primitive)

Method: muslin-available
Subtasks:
	1.	use-muslin-cloth (primitive)

⸻

Compound Task: sweeten-chai

Description: Add sweetener to chai.

Method: default-sweetening
Subtasks:
	1.	measure-sweetener (primitive)
	2.	stir-sweetener (primitive)

⸻

Compound Task: cleanup

Description: Dispose waste and tidy.

Method: home-cleanup
Subtasks:
	1.	discard-into-trashcan (primitive)

Method: campsite-cleanup
Subtasks:
	1.	discard-into-trashbag (primitive)


