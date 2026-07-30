import os
import json

from dotenv import load_dotenv
from groq import Groq

from recording_to_playback import (
    record_audio,
    transcribe_audio,
    speak
)

from calculation_tools import (
    tools,
    available_functions
)

from menu import retrieve_menu_context

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def classify_intent(text):

    text = text.lower()

    if "menu" in text:
        return "menu"

    if "price" in text or "total" in text:
        return "pricing"

    if any(
        item in text
        for item in [
            "burger",
            "cheeseburger",
            "fries",
            "wrap",
            "tenders",
            "order",
            "bacon",
            "meal",
            "food",
            "eat",
            "drink",
            "thirsty",
            "hungry",
            "lunch",
            "dinner",
        ]
    ):
        return "ordering"

    return "unknown"


def validate_request(text):

    blocked_topics = [
        "politics",
        "stocks",
        "bitcoin",
        "weather",
        "election"
    ]

    text = text.lower()

    return not any(
        topic in text
        for topic in blocked_topics
    )


def run_drive_thru_turn():

    print("\nPLEASE PLACE YOUR ORDER")

    record_audio(
        output_file="customer_input.wav",
        duration=6
    )

    user_prompt = transcribe_audio(
        "customer_input.wav"
    )

    print(f'\nCustomer Said: "{user_prompt}"')

    if not user_prompt.strip():

        final_reply = (
            "I'm sorry, I couldn't hear you clearly. "
            "Please repeat your order."
        )

        print("\nResponse:")
        print(final_reply)

        speak(final_reply)
        return

    if not validate_request(user_prompt):

        final_reply = (
            "I can only help with restaurant orders "
            "and menu questions."
        )

        print("\nResponse:")
        print(final_reply)

        speak(final_reply)
        return

    intent = classify_intent(user_prompt)

    print(f"\nIntent: {intent}")

    if intent == "unknown":

        final_reply = (
            "I'm sorry, I didn't catch a food order. "
            "Could you please repeat your order?"
        )

        print("\nResponse:")
        print(final_reply)

        speak(final_reply)
        return

    context = retrieve_menu_context(
        user_prompt
    )

    print("\nMenu Context:")
    print(context)

    messages = [
        {
            "role": "system",
            "content": f"""
You are a friendly AI Drive-Thru assistant.

Detected Intent:
{intent}

Retrieved Menu Knowledge:
{context}

Rules:
- Use retrieved menu information.
- Use tools whenever pricing is needed.
- Keep responses short.
- Speak naturally.
- Ask clarifying questions when necessary.
"""
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

    response_message = (
        response.choices[0].message
    )

    tool_calls = (
        response_message.tool_calls
    )

    if tool_calls:

        messages.append(response_message)

        for tool_call in tool_calls:

            function_name = (
                tool_call.function.name
            )

            function_to_call = (
                available_functions[
                    function_name
                ]
            )

            raw_args = (
                tool_call.function.arguments
            )

            function_args = (
                json.loads(raw_args)
                if raw_args
                else {}
            )

            print(
                f"\n[Tool Call] {function_name}"
            )

            function_response = (
                function_to_call(
                    **function_args
                )
            )

            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )

        second_response = (
            client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages
            )
        )

        final_reply = (
            second_response
            .choices[0]
            .message
            .content
        )

    else:

        final_reply = (
            response_message.content
        )

    print("\nResponse:")
    print(final_reply)

    speak(final_reply)


if __name__ == "__main__":
    run_drive_thru_turn()
