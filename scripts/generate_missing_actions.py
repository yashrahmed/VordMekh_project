import re
from pathlib import Path

SRC = Path('chai_gpt_v7/recipe-exploration/recipe-responses.txt')
OUT = Path('missing-actions.txt')

# Actions covered by ChaiPreparationIngredientsActionsFrame
COVERED = {
    'crush', 'grind', 'peel', 'slice', 'grate', 'chop', 'stir', 'strain', 'aerate',
}

# Patterns for additional actions that involve tools or ingredient manipulation
ACTION_PATTERNS = [
    ('bring_to_boil', re.compile(r'bring\b[^\n]*\bboil|\bboil\b', re.I)),
    ('simmer', re.compile(r'\bsimmer\b', re.I)),
    ('pour_milk', re.compile(r'\b(pour|slowly\s+pour)\b[^\n]*\bmilk\b', re.I)),
    ('add_sweetener', re.compile(r'\badd\b[^\n]*\b(sugar|jaggery|honey|stevia)\b', re.I)),
    ('steep_or_rest', re.compile(r'\bsteep\b|\b(rest|settle)\b', re.I)),
    ('ladle_transfer', re.compile(r'\bladle\b', re.I)),
    ('squeeze_citrus', re.compile(r'\b(squeeze|squeezing)\b[^\n]*\b(lemon|lime)\b', re.I)),
    ('pierce_dried_lime', re.compile(r'\bpierce(d)?\b[^\n]*\b(lime|loomi)\b', re.I)),
    ('warm_milk_separately', re.compile(r'\bwarm\b[^\n]*\bmilk\b|\bmilk\b[^\n]*\bwarm\b', re.I)),
    ('zest_citrus', re.compile(r'\bzest\b', re.I)),
    ('bruise_muddle_herbs', re.compile(r'(bruise(d)?|muddle(d)?|lightly\s+crush(ed)?)\b[^\n]*\b(mint|tulsi|lemongrass)\b', re.I)),
    ('heat_clay_cup', re.compile(r'(heat|heated)\b[^\n]*\b(clay\s+cup|kulhad)|\b(clay\s+cup|kulhad)\b[^\n]*\bheat', re.I)),
    ('smoke_infusion', re.compile(r'\bsmok(e|y|ed|ing)\b', re.I)),
    ('pour_into_kulhad_and_back', re.compile(r'pour\b[^\n]*\b(kulhad|clay\s+cup)\b|pour\s+back', re.I)),
    ('add_baking_soda', re.compile(r'\badd\b[^\n]*\bbaking\s+soda\b', re.I)),
    ('shock_with_ice_or_cold_water', re.compile(r'\b(ice|cold\s+water)\b', re.I)),
]

# Canonical labels for output
LABELS = {
    'bring_to_boil': 'bring to a gentle boil',
    'simmer': 'simmer to extract/infuse',
    'pour_milk': 'slowly pour in milk',
    'add_sweetener': 'add sweetener (sugar/jaggery/honey/stevia)',
    'steep_or_rest': 'steep/rest to infuse flavors',
    'ladle_transfer': 'ladle/transfer in batches',
    'squeeze_citrus': 'squeeze citrus (lemon/lime) juice',
    'pierce_dried_lime': 'pierce dried lime (loomi)',
    'warm_milk_separately': 'warm milk separately',
    'zest_citrus': 'zest citrus',
    'bruise_muddle_herbs': 'bruise/muddle herbs (mint/tulsi/lemongrass)',
    'heat_clay_cup': 'heat clay cup (kulhad) with tongs',
    'smoke_infusion': 'smoke infusion using hot vessel/masala',
    'pour_into_kulhad_and_back': 'pour into hot kulhad and back',
    'add_baking_soda': 'add baking soda',
    'shock_with_ice_or_cold_water': 'shock with ice/add cold water',
}

def iter_recipes(text: str):
    """Yield (title, body) for each recipe block."""
    blocks = re.split(r'\n\+{30,}\n', text)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        # Title is the first non-empty line starting with 'Recipe for'
        lines = [l for l in block.splitlines() if l.strip()]
        title = None
        for l in lines:
            if l.lower().startswith('recipe for'):
                title = l.strip()
                break
        if title is None:
            # Skip non-recipe text
            continue
        yield title, block

def detect_missing_actions(block: str):
    text = block
    found = set()
    for key, pat in ACTION_PATTERNS:
        if pat.search(text):
            found.add(key)
    # Remove actions that are already covered by explicit flags if the text only indicates those.
    # (Heuristic: if the only verbs present are covered ones, result would be empty; but we keep all detected above)
    return [LABELS[k] for k in sorted(found)]

def main():
    text = SRC.read_text(encoding='utf-8')
    out_lines = []
    counts = {}
    for title, block in iter_recipes(text):
        counts[title] = counts.get(title, 0) + 1
        idx = counts[title]
        header = f"{title} — instance {idx}"
        out_lines.append(header)
        missing = detect_missing_actions(block)
        if not missing:
            out_lines.append("- No additional actions detected")
        else:
            for m in missing:
                out_lines.append(f"- {m}")
        out_lines.append("")

    OUT.write_text("\n".join(out_lines), encoding='utf-8')

if __name__ == '__main__':
    main()
