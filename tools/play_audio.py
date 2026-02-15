"""
Quick utility to decode and save base64 audio from API response.

Usage:
    # Save from a JSON response file:
    python -m tools.play_audio response.json

    # Or pipe from curl:
    curl -s -X POST "http://localhost:8000/api/v1/translate" \
        -F "file=@my_voice.wav" | python -m tools.play_audio -
"""
import base64
import json
import sys
from pathlib import Path


def decode_and_save(response_data: dict, output_path: str = "meow_output.wav") -> Path:
    """Decode audio_base64 from API response and save as WAV."""
    audio_b64 = response_data.get("audio_base64")
    if not audio_b64:
        print("Error: No audio_base64 field found in response.")
        print("  synthesis_ok:", response_data.get("synthesis_ok"))
        sys.exit(1)

    out = Path(output_path)
    out.write_bytes(base64.b64decode(audio_b64))

    # Print summary
    sr = response_data.get("synthesis_metadata", {}).get("sample_rate", "?")
    dur = response_data.get("synthesis_metadata", {}).get("duration_seconds", "?")
    desc = response_data.get("preview_description", {})

    print(f"Audio saved to: {out.resolve()}")
    print(f"  Duration: {dur}s  |  Sample rate: {sr} Hz")
    print(f"  Emotion:  {response_data.get('emotion_category')}")
    print(f"  Intent:   {desc.get('intent_label', '?')}")
    print(f"  Summary:  {desc.get('summary', '?')}")
    print(f"\nPlay with:  aplay {out}  (Linux)  |  afplay {out}  (macOS)")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m tools.play_audio <response.json | ->")
        sys.exit(1)

    source = sys.argv[1]
    if source == "-":
        data = json.load(sys.stdin)
    else:
        data = json.loads(Path(source).read_text())

    output = sys.argv[2] if len(sys.argv) > 2 else "meow_output.wav"
    decode_and_save(data, output)
