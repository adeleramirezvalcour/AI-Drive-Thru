import os
from playsound import playsound
from gtts import gTTS
from dotenv import load_dotenv
from groq import Groq
import soundfile as sf
import sounddevice as sd
from pathlib import Path

# Connect with API credentials
load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# Define the variable that records the audio input, defining paramaters optimal for ordering


def record_audio(
    output_file: str = "customer_input.wav",
    duration: int = 5,
    sample_rate: int = 16000
):

    print(f"\nRecording for {duration} seconds...")

    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16"
    )

    sd.wait()

    sf.write(
        output_file,
        recording,
        sample_rate
    )

    print(f"Audio saved: {output_file}")

# Define the function that sends the audio to Groq Whisper and returns the transcribed text


def transcribe_audio(audio_path: str) -> str:

    with open(audio_path, "rb") as audio_file:

        transcription = client.audio.transcriptions.create(
            file=(
                Path(audio_path).name,
                audio_file.read()
            ),
            model="whisper-large-v3-turbo",
            language="en",
            response_format="json"
        )

    return transcription.text

# Defines the function that converts the transcribed text into spoken audio and saves this output as an MP3 file


def text_to_speech(
    text: str,
    output_file: str = "response.mp3"
):

    tts = gTTS(
        text=text,
        lang="en"
    )

    tts.save(output_file)

    print(f"Audio response saved: {output_file}")

# Defines the function that plays the outputted MP3 file


def play_audio(audio_file: str = "response.mp3"):

    print("Playing response...")

    playsound(audio_file)

# Define a function that brings together a few other previously defined function (for convienience)


def speak(text: str):
    text_to_speech(text)

    play_audio("response.mp3")


# Pipeline verification
if __name__ == "__main__":

    record_audio(duration=5)

    transcript = transcribe_audio(
        "customer_input.wav"
    )

    print("\nCustomer Said:")
    print(transcript)

    response = (
        f"You said: {transcript}"
    )

    speak(response)
