import json

PRICE_MENU = {
    "cheeseburger": 6.99,
    "bacon double": 8.99,
    "veggie wrap": 7.49,
    "chicken tenders": 7.99,
    "fries": 3.49,
}


def calculate_order_total(item_names: list[str], tax_rate: float = 0.08) -> str:
    """
    Calculates subtotal, tax, and total for a customer order.
    Returns a JSON string so the LLM can read the structured data.
    """

    subtotal = 0.0
    found_items = []
    missing_items = []

    for item in item_names:
        clean_name = item.lower().strip()

        if clean_name in PRICE_MENU:
            subtotal += PRICE_MENU[clean_name]
            found_items.append(clean_name)
        else:
            missing_items.append(clean_name)

    tax = subtotal * tax_rate
    total = subtotal + tax

    result = {
        "items_processed": found_items,
        "items_not_found": missing_items,
        "subtotal": round(subtotal, 2),
        "tax": round(tax, 2),
        "total": round(total, 2),
    }

    return json.dumps(result)


def get_menu() -> str:
    """
    Returns the current menu and prices.
    """

    return json.dumps(PRICE_MENU)


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

available_functions = {
    "calculate_order_total": calculate_order_total,
    "get_menu": get_menu,
}
