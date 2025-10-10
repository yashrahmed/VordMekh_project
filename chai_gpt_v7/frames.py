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


class ChaiPreparationFrame(BaseModel):
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

    # Actions (tool mappings)
    crushing_tools: Optional[list[str]] = Field(None, description="Tools for crushing spices: mortar and pestle, rolling pin, etc.")
    peeling_tools: Optional[list[str]] = Field(None, description="Tools for peeling ginger, citrus: peeler, knife, spoon, etc.")
    stirring_tools: Optional[list[str]] = Field(None, description="Tools for mixing and stirring: spoon, ladle, whisk, etc.")
    straining_tools: Optional[list[str]] = Field(None, description="Tools for filtering tea: strainer, muslin cloth, tea filter, sieve, etc.")
    aerating_tools: Optional[list[str]] = Field(None, description="Tools for creating froth/aeration: whisk, deep ladle (for pulling), frother, etc.")

    def generate_description(self) -> str:
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
