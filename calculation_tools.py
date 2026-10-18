import json

# Make price dictionary for Python
PRICE_MENU = {
    "cheeseburger": 6.99,
    "bacon double": 8.99,
    "veggie wrap": 7.49,
    "chicken tenders": 7.99,
    "fries": 3.49,
}

# Define the function that tells Llama 3 to calculate the order total based on the price dictionary (above) and


def calculate_order_total(item_names: list[str], tax_rate: float = 0.08) -> str:
    """
    Calculates subtotal, tax, and total for a customer order.
    Returns a JSON string so the LLM can read the structured data.
    """
    # Start with baseline subtotal and make empty sets for the found(Python found them on menu) and missing(Python did not find them on menu) items
    subtotal = 0.0
    found_items = []
    missing_items = []

    # Preprocess input
    for item in item_names:
        clean_name = item.lower().strip()
        # Store the cleaned names of found and missing items in their respective lists
        if clean_name in PRICE_MENU:
            subtotal += PRICE_MENU[clean_name]
            found_items.append(clean_name)
        else:
            missing_items.append(clean_name)
    # Calcuate the total, including tax
    tax = subtotal * tax_rate
    total = subtotal + tax
    # Output for Llama 3 to know what information needs to be relayed
    result = {
        "items_processed": found_items,
        "items_not_found": missing_items,
        "subtotal": round(subtotal, 2),
        "tax": round(tax, 2),
        "total": round(total, 2),
    }
    # Convert the Python dictionary into JSON formatting so that Llama 3 can process it
    return json.dumps(result)

# Define the function that should be called when the customer wants to know the menu


def get_menu() -> str:
    """
    Returns the current menu and prices.
    """

    return PRICE_MENU


# Give Llama 3 a description of what it has access to and functions of each tool
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate_order_total",
            "description": (
                "Calculate subtotal, tax, and grand total "
                "for menu items ordered by the customer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of menu items ordered."
                    },
                    "tax_rate": {
                        "type": "number",
                        "description": "Sales tax rate. Default is 0.08."
                    }
                },
                "required": ["item_names"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_menu",
            "description": (
                "Return the available menu items and pricing."
            ),
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]

# Make a dictionary of the available functions so Python can locate the function that corresponds to the JSON string that Llama 3 outputs
available_functions = {
    "calculate_order_total": calculate_order_total,
    "get_menu": get_menu,
}
