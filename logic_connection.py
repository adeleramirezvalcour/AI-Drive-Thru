import os
import json
from dotenv import load_dotenv
from groq import Groq

from recording_to_playback import record_audio, transcribe_audio, speak
from calculation_tools import tools, available_functions

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def run_drive_thru_turn():
    """
    Executes one full drive-thru interaction turn:
    1. Records live microphone input
    2. Transcribes with Groq Whisper
    3. Processes order with Llama 3.3 + Calculation Tools
    4. Speaks the final response back to the customer
    """
    print("\nPLEASE PLACE YOUR ORDER")
    record_audio(output_file="customer_input.wav", duration=6)

    user_prompt = transcribe_audio("customer_input.wav")
    print(f"Customer Said: \"{user_prompt}\"")

    if not user_prompt.strip():
        speak("I'm sorry, I couldn't hear you clearly. Could you please repeat that?")
        return

    messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly AI Drive-Thru assistant. Use the calculate_order_total tool "
                "to calculate exact totals when customers ask for prices or order items. "
                "Keep responses concise and clear for spoken audio. "
                "If the transcript sounds nonsensical or unrelated to fast food, politely ask "
                "the customer to repeat their order."
            )
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
        messages.append(response_message)

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]

            raw_args = tool_call.function.arguments
            function_args = json.loads(raw_args) if raw_args else {}

            if function_args is None:
                function_args = {}

            print(f"[Tool Call] Executing: {function_name}({function_args})")
            function_response = function_to_call(**function_args)

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
        final_reply = second_response.choices[0].message.content
    else:
        final_reply = response_message.content

    print(f"\nAI Response: \"{final_reply}\"")

    print("\nORDER SUMMARY:")
    speak(final_reply)


if __name__ == "__main__":
    run_drive_thru_turn()
