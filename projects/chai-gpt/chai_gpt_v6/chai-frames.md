ChaiPreparationFrame:
  # Liquids & tea
  liquids:
    - { name: "water", amount: ?, unit: "cup" }
    - { name: "milk", amount: ?, unit: "cup" }
  teas:
    - { name: "loose black tea", amount: ?, unit: "tsp" }
    - { name: "kashmiri green tea leaves", amount: ?, unit: "tsp" }
    - { name: "green tea leaves", amount: ?, unit: "tsp" }

  # Sweeteners & seasoning
  sweeteners:
    - { name: "jaggery", amount: ?, unit: "tsp" }
    - { name: "sugar", amount: ?, unit: "tsp" }
    - { name: "honey", amount: ?, unit: "tsp" }
  salt:
    - { name: "salt", amount: ?, unit: "tsp" }

  # Spices (generic, list form)
  spices_ground:
    - { name: "ginger", amount: ?, unit: "tsp" }
    - { name: "cinnamon", amount: ?, unit: "tsp" }
  spices_whole:
    - { name: "cardamom pods", amount: ?, unit: "pods" }
    - { name: "cloves", amount: ?, unit: "cloves" }
    - { name: "peppercorns", amount: ?, unit: "peppercorns" }
    - { name: "fennel seeds", amount: ?, unit: "tsp" }
    - { name: "saffron threads", amount: ?, unit: "threads" }

  # Herbs / floral / citrus
  herbs:
    - { name: "mint leaves", amount: ?, unit: "leaves" }
  floral:
    - { name: "rose petals", amount: ?, unit: "tsp" }
  citrus:
    - { name: "lemon juice", amount: ?, unit: "tsp" }

  # Process modifiers
  process_modifiers:
    - { name: "baking soda", amount: ?, unit: "tsp" }
    - { name: "ice", amount: ?, unit: "cup" }

  # Garnish
  garnish:
    - { name: "crushed nuts", amount: ?, unit: "tbsp" }
    - { name: "slivered almonds", amount: ?, unit: "tbsp" }

  # Process/tool actions
  actions:
    crushing:   [ "mortar and pestle" ]
    peeling:    [ "peeler" ]
    stirring:   [ "spoon", "ladle" ]
    straining:  [ "strainer", "muslin" ]
    aerating:   [ "whisk", "deep ladle" ]