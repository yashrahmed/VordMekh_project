from pydantic import BaseModel, Field
from typing import Optional


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


class Ingredient(BaseModel):
    """Represents an ingredient with name, amount, and unit."""
    name: str = Field(..., description="Name of the ingredient")
    amount: Optional[float] = Field(None, description="Quantity of the ingredient")
    unit: Optional[str] = Field(None, description="Unit of measurement (e.g., cup, tsp, tbsp)")


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


class ChaiPreparationIngredientsActionsFrame(BaseModel):
    """Frame to represent the ingredients and tools for chai preparation."""
    chai_type: str = Field(..., description="The name/type of chai recipe (e.g., 'Masala Chai', 'Adrak Chai', 'Sulaimani Chai')")

    # Liquids & tea
    liquids: Optional[list[Ingredient]] = Field(None, description="Base liquids like water, milk (dairy or plant-based)")
    teas: Optional[list[Ingredient]] = Field(None, description="Tea leaves: black tea, green tea, kashmiri green tea, etc.")

    # Sweeteners & seasoning
    sweeteners: Optional[list[Ingredient]] = Field(None, description="Sweetening agents: sugar, jaggery, honey, etc.")
    salt: Optional[list[Ingredient]] = Field(None, description="Salt for savory chai variants (e.g., noon chai)")

    # Spices
    spices_ground: Optional[list[Ingredient]] = Field(None, description="Pre-ground spices: ginger powder, cinnamon powder, etc.")
    spices_whole: Optional[list[Ingredient]] = Field(None, description="Whole spices: cardamom pods, cloves, peppercorns, fennel seeds, saffron threads")

    # Herbs / floral / citrus
    herbs: Optional[list[Ingredient]] = Field(None, description="Fresh or dried herbs: mint leaves, tulsi, lemongrass, etc.")
    floral: Optional[list[Ingredient]] = Field(None, description="Floral additions: rose petals, jasmine, etc.")
    citrus: Optional[list[Ingredient]] = Field(None, description="Citrus elements: lemon juice, orange zest, etc.")

    # Process modifiers
    process_modifiers: Optional[list[Ingredient]] = Field(None, description="Ingredients that modify cooking process: baking soda (for pink chai), ice (for iced chai)")

    # Garnish
    garnish: Optional[list[Ingredient]] = Field(None, description="Toppings and finishing touches: crushed nuts, slivered almonds, spice powder, cream, etc.")

    # Actions (verb-noun boolean properties)
    crush_spices: Optional[bool] = Field(None, description="Whether spices need to be crushed during preparation")
    peel_ingredients: Optional[bool] = Field(None, description="Whether ingredients (ginger, citrus) need to be peeled")
    stir_chai: Optional[bool] = Field(None, description="Whether chai needs to be stirred/mixed during preparation")
    strain_chai: Optional[bool] = Field(None, description="Whether chai needs to be strained/filtered")
    aerate_chai: Optional[bool] = Field(None, description="Whether chai needs to be aerated/frothed (e.g., by pulling)")

    def generate_description(self) -> str:
        """Generate a formatted description of ingredients and preparation actions."""
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
                Ingredient(name=LIQUID_WATER, amount=0.75, unit="cup"),
                Ingredient(name=LIQUID_WHOLE_MILK, amount=0.5, unit="cup")
            ],
            teas=[Ingredient(name=TEA_LOOSE_BLACK, amount=1, unit="tsp")],
            sweeteners=[Ingredient(name=SWEETENER_JAGGERY_OR_SUGAR, amount=1, unit="tsp")],
            salt=None,
            spices_ground=[
                Ingredient(name=SPICE_GROUND_GINGER, amount=0.5, unit="tsp"),
                Ingredient(name=SPICE_GROUND_CINNAMON, amount=0.25, unit="tsp")
            ],
            spices_whole=[
                Ingredient(name=SPICE_WHOLE_CARDAMOM, amount=3, unit="pods"),
                Ingredient(name=SPICE_WHOLE_CLOVES, amount=2, unit="cloves"),
                Ingredient(name=SPICE_WHOLE_PEPPERCORNS, amount=2, unit="peppercorns"),
                Ingredient(name=SPICE_WHOLE_FENNEL, amount=0.25, unit="tsp")
            ],
            herbs=None,
            floral=None,
            citrus=None,
            process_modifiers=None,
            garnish=None,
            crush_spices=True,
            peel_ingredients=True,
            stir_chai=None,
            strain_chai=True,
            aerate_chai=None
        ))

        # Adrak Chai
        self.variants.append(ChaiPreparationIngredientsActionsFrame(
            chai_type=CHAI_ADRAK,
            liquids=[
                Ingredient(name=LIQUID_WATER, amount=0.75, unit="cup"),
                Ingredient(name=LIQUID_WHOLE_MILK, amount=0.5, unit="cup")
            ],
            teas=[Ingredient(name=TEA_LOOSE_BLACK, amount=1.5, unit="tsp")],
            sweeteners=[Ingredient(name=SWEETENER_JAGGERY_OR_SUGAR, amount=1, unit="tsp")],
            salt=None,
            spices_ground=[Ingredient(name=SPICE_GROUND_GINGER, amount=3, unit="tsp")],
            spices_whole=None,
            herbs=None,
            floral=None,
            citrus=None,
            process_modifiers=None,
            garnish=None,
            crush_spices=True,
            peel_ingredients=True,
            stir_chai=None,
            strain_chai=True,
            aerate_chai=None
        ))

        # Sulaimani Chai
        self.variants.append(ChaiPreparationIngredientsActionsFrame(
            chai_type=CHAI_SULAIMANI,
            liquids=[Ingredient(name=LIQUID_WATER, amount=1, unit="cup")],
            teas=[Ingredient(name=TEA_LOOSE_BLACK, amount=1, unit="tsp")],
            sweeteners=[Ingredient(name=SWEETENER_HONEY_OR_JAGGERY, amount=1, unit="tsp")],
            salt=None,
            spices_ground=None,
            spices_whole=[
                Ingredient(name=SPICE_WHOLE_CLOVES, amount=2, unit="cloves"),
                Ingredient(name=SPICE_WHOLE_CARDAMOM, amount=1, unit="pods"),
                Ingredient(name=SPICE_WHOLE_SAFFRON, amount=5, unit="threads")
            ],
            herbs=[Ingredient(name=HERB_MINT, amount=3, unit="leaves")],
            floral=None,
            citrus=[Ingredient(name=CITRUS_LEMON_JUICE, amount=1, unit="tsp")],
            process_modifiers=None,
            garnish=None,
            crush_spices=None,
            peel_ingredients=None,
            stir_chai=None,
            strain_chai=True,
            aerate_chai=None
        ))

        # Kashmiri Chai
        self.variants.append(ChaiPreparationIngredientsActionsFrame(
            chai_type=CHAI_KASHMIRI,
            liquids=[
                Ingredient(name=LIQUID_WATER, amount=1.5, unit="cup"),
                Ingredient(name=LIQUID_WHOLE_MILK, amount=0.75, unit="cup")
            ],
            teas=[Ingredient(name=TEA_KASHMIRI_GREEN, amount=1, unit="tsp")],
            sweeteners=None,
            salt=[Ingredient(name=SALT, amount=0.5, unit="tsp")],
            spices_ground=None,
            spices_whole=[Ingredient(name=SPICE_WHOLE_CARDAMOM, amount=1, unit="pods")],
            herbs=None,
            floral=None,
            citrus=None,
            process_modifiers=[
                Ingredient(name=PROCESS_BAKING_SODA, amount=0.125, unit="tsp"),
                Ingredient(name=PROCESS_ICE, amount=0.5, unit="cup")
            ],
            garnish=[Ingredient(name=GARNISH_CRUSHED_NUTS, amount=0.5, unit="tbsp")],
            crush_spices=None,
            peel_ingredients=None,
            stir_chai=None,
            strain_chai=True,
            aerate_chai=True
        ))

        # Kahwah
        self.variants.append(ChaiPreparationIngredientsActionsFrame(
            chai_type=CHAI_KAHWAH,
            liquids=[Ingredient(name=LIQUID_WATER, amount=1, unit="cup")],
            teas=[Ingredient(name=TEA_GREEN, amount=0.5, unit="tsp")],
            sweeteners=[Ingredient(name=SWEETENER_HONEY_OR_SUGAR, amount=1, unit="tsp")],
            salt=None,
            spices_ground=[Ingredient(name=SPICE_GROUND_CINNAMON, amount=0.125, unit="tsp")],
            spices_whole=[
                Ingredient(name=SPICE_WHOLE_CARDAMOM, amount=1, unit="pods"),
                Ingredient(name=SPICE_WHOLE_SAFFRON, amount=5, unit="threads")
            ],
            herbs=None,
            floral=[Ingredient(name=FLORAL_ROSE_PETALS, amount=0.5, unit="tsp")],
            citrus=None,
            process_modifiers=None,
            garnish=[Ingredient(name=GARNISH_ALMONDS, amount=0.5, unit="tbsp")],
            crush_spices=None,
            peel_ingredients=None,
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

