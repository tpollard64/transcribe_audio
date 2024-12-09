"""
File Name: transcribe_audio.py
Author: Tyler Pollard (Enhanced by Claude)
Description:
    This script converts an MP3 audio file to a WAV format and transcribes the audio content into text
    using the OpenAI Whisper model. The transcription is saved as a formatted markdown file.
    The script includes error handling and optimization for efficiency.

Usage:
    1. Replace "path-to-file.mp3" with the path to your MP3 file
    2. Run the script using Python 3.x:
       python transcribe_audio.py
    3. The transcription will be saved as a markdown file in the same directory

Dependencies:
    - openai-whisper >= 1.0
    - pydub >= 0.25
    - ffmpeg (for audio conversion)

License:
    MIT License
"""
import whisper
from pydub import AudioSegment
from pathlib import Path
from datetime import datetime
import logging
import os

def setup_logging():
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('transcription.log')
        ]
    )

def transcribe_audio(file_path: str) -> dict:
    """
    Transcribe audio using Whisper model.
    
    Args:
        file_path: Path to WAV file
    
    Returns:
        dict: Transcription result containing text and segments
    """
    try:
        logging.info("Loading Whisper model...")
        model = whisper.load_model("base")
        
        logging.info("Transcribing audio...")
        result = model.transcribe(
            file_path,
            verbose=False,  # Reduce console output
            fp16=False  # Use CPU if GPU unavailable
        )
        return result
    
    except Exception as e:
        logging.error(f"Error during transcription: {str(e)}")
        return None

"""
[Previous imports and helper functions remain the same until create_markdown()]
"""

def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to MM:SS format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        str: Formatted time string
    """
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def create_markdown(transcription: dict, input_file: str) -> str:
    """
    Create formatted markdown from transcription results.
    
    Args:
        transcription: Whisper transcription result
        input_file: Original input file path
    
    Returns:
        str: Formatted markdown content
    """
    file_name = Path(input_file).name
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    markdown = f"""# Audio Transcription

## File Information
- **Source File:** {file_name}
- **Transcription Date:** {current_time}

## Segments
"""
    
    # Add timestamped segments with improved formatting
    if 'segments' in transcription:
        for segment in transcription['segments']:
            time = format_timestamp(segment['start'])
            text = segment['text'].strip()
            markdown += f"[{time}] {text}\n\n"
    
    return markdown

def main():
    """Main script execution."""
    setup_logging()
    
    # Input and output files
    audio_file = "TPS/transcribe/lat_dir_grumpy.mp3"
    markdown_file = Path(audio_file).stem + "_transcription.md"
    
    # Validate input file
    if not os.path.exists(audio_file):
        logging.error(f"Input file not found: {audio_file}")
        return

    # Transcribe
    transcription = transcribe_audio(audio_file)
    if not transcription:
        return
    
    # Create and save markdown
    try:
        markdown_content = create_markdown(transcription, audio_file)
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        logging.info(f"Transcription saved to {markdown_file}")
        
    except Exception as e:
        logging.error(f"Error saving markdown: {str(e)}")

if __name__ == "__main__":
    main()
