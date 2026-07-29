import os
import json
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

PRICE_MENU = {
    "cheeseburger": 6.99,
    "bacon double": 8.99,
    "veggie wrap": 7.49,
    "chicken tenders": 7.99,
    "fries": 3.49
}


def calculate_order_total(item_names: list[str], tax_rate: float = 0.08) -> str:
    """Calculates subtotal, tax, and grand total for a list of menu items."""
    subtotal = 0.0
    found_items = []

    for item in item_names:
        clean_name = item.lower().strip()
        if clean_name in PRICE_MENU:
            subtotal += PRICE_MENU[clean_name]
            found_items.append(clean_name)

    tax = subtotal * tax_rate
    total = subtotal + tax

    result = {
        "items_processed": found_items,
        "subtotal": round(subtotal, 2),
        "tax": round(tax, 2),
        "total": round(total, 2)
    }
    return json.dumps(result)


tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate_order_total",
            "description": "Calculates the total cost for order items including tax. Call this whenever a customer asks for a price total or bill.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of menu item names ordered by the customer (e.g. ['cheeseburger', 'fries'])"
                    },
                    "tax_rate": {
                        "type": "number",
                        "description": "Sales tax rate decimal (default is 0.08)"
                    }
                },
                "required": ["item_names"]
            }
        }
    }
]


available_functions = {
    "calculate_order_total": calculate_order_total
}


def transcribe_audio_file(audio_path: str) -> str:
    """Uses Groq's free Whisper API to transcribe customer audio into text."""
    with open(audio_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=(Path(audio_path).name, file.read()),
            model="whisper-large-v3-turbo",
            response_format="json",
            language="en"
        )
    return transcription.text


audio_file_path = "customer_order.wav"
print(f"Processing Audio File: {audio_file_path}")

user_prompt = transcribe_audio_file(audio_file_path)
print(f"Transcribed Customer Prompt\n'{user_prompt}'\n")

messages = [
    {
        "role": "system",
        "content": "You are an AI Drive-Thru assistant. Use the provided calculate_order_total tool to calculate exact prices. Never guess or hallucinate math."
    },
    {
        "role": "user",
        "content": user_prompt
    }
]

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

response_message = response.choices[0].message
tool_calls = response_message.tool_calls

if tool_calls:
    print("Tool Called")

    messages.append(response_message)

    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_to_call = available_functions[function_name]
        function_args = json.loads(tool_call.function.arguments)

        print(f"Executing Function: {function_name}()")
        print(f"Extracted Arguments: {function_args}")

        function_response = function_to_call(**function_args)
        print(f"Function Output: {function_response}\n")

        messages.append({
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": function_name,
            "content": function_response,
        })

    second_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )

    print("Final Response")
    print(second_response.choices[0].message.content)
else:
    print("No tool call was triggered. Response:")
    print(response_message.content)
