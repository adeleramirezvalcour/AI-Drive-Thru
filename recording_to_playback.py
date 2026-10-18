import os
from playsound import playsound
from gtts import gTTS
from dotenv import load_dotenv
from groq import Groq
import soundfile as sf
import sounddevice as sd
from pathlib import Path

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def record_audio(
    output_file: str = "customer_input.wav",
    duration: int = 5,
    sample_rate: int = 16000
):
    """
    Records audio from the microphone.

    Args:
        output_file:
            Where the recording will be saved.

        duration:
            Recording length in seconds.

        sample_rate:
            Audio quality. 16000 is perfect for speech.
    """

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


def transcribe_audio(audio_path: str) -> str:
    """
    Sends audio to Groq Whisper and
    returns the transcription text.
    """

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


def text_to_speech(
    text: str,
    output_file: str = "response.mp3"
):
    """
    Converts text into spoken audio.
    Saves the result as an MP3 file.
    """

    tts = gTTS(
        text=text,
        lang="en"
    )

    tts.save(output_file)

    print(f"Audio response saved: {output_file}")


def play_audio(audio_file: str = "response.mp3"):
    """
    Plays the generated MP3 response.
    """

    print("Playing response...")

    playsound(audio_file)


def speak(text: str):
    """
    Convenience function.

    Generates speech and
    immediately plays it.
    """

    text_to_speech(text)

    play_audio("response.mp3")


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
