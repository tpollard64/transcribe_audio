"""
File Name: transcribe_audio.py
Author: Tyler Pollard
Description:
    This script converts an MP3 audio file to a WAV format and transcribes the audio content into text 
    using the OpenAI Whisper model. The transcription process is efficient and runs entirely offline.
Usage:
    1. Replace "path-to-file.mp3" with the path to your MP3 file.
    2. Run the script using Python 3.x:
        python transcribe_audio.py
    3. The transcription will be printed to the console.

Dependencies:
    - openai-whisper >= 1.0
    - pydub >= 0.25
    - ffmpeg (for audio conversion, ensure it's installed and in your PATH)
License:
    MIT License
"""
import whisper
from pydub import AudioSegment
def convert_mp3_to_wav(input_mp3, output_wav):
    audio = AudioSegment.from_mp3(input_mp3)
    audio.export(output_wav, format="wav")
def transcribe_audio(file_path):
    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    return result["text"]

# Replace with your MP3 file path
mp3_file = "path-to-file.mp3"  
wav_file = "converted_audio.wav"

convert_mp3_to_wav(mp3_file, wav_file)

transcription = transcribe_audio(wav_file)

print(transcription)
