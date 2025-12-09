from typing import List, Dict

from pydantic import BaseModel, Field

class Item(BaseModel):
    """Inventory item describing a tool, vessel, or appliance plus its supported actions."""

    key: str = Field(..., description="Unique machine-friendly identifier that matches items.yml.")
    name: str = Field(..., description="Human-readable label for the item.")
    description: str = Field(..., description="Text describing construction, form factor, or usage context.")
    actions: List[str] = Field(..., description="List of CookingActions keys this item can perform.")

    def pretty_print(self) -> str:
        """Readable representation that lists the metadata and supported actions."""
        actions_text = ", ".join(self.actions) if self.actions else "None"
        return (
            f"{self.name} ({self.key})\n"
            f"Description: {self.description}\n"
            f"Supported actions: {actions_text}"
        )

class CookingActions(BaseModel):
    actions: List[str] = Field(default=[], description="List of detected cooking actions")
