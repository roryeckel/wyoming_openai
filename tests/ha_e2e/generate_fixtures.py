"""Generate the committed Home Assistant E2E fixture WAVs with Speaches Kokoro.

Run once with Speaches up, then commit the WAVs and manifest:

    docker compose -f docker-compose.speaches-cpu.yml -f docker-compose.e2e.yml up -d speaches
    python tests/ha_e2e/generate_fixtures.py

The proxy under test is deliberately not involved: the fixtures are produced
by Speaches' Kokoro and resampled to the 16 kHz mono format Home Assistant's
Assist pipeline expects.  A trailing silence gives Home Assistant's VAD a
clean end of speech.
"""

from __future__ import annotations

import array
import hashlib
import io
import json
import logging
import os
import sys
import wave
from datetime import UTC, datetime
from pathlib import Path

import httpx

LOGGER = logging.getLogger("ha_e2e.generate_fixtures")

OUTPUT_DIR = Path(__file__).resolve().parent / "fixtures" / "audio"
SPEACHES_URL = os.environ.get("HA_E2E_SPEACHES_URL", "http://127.0.0.1:8000/v1").rstrip("/")
TTS_MODEL = os.environ.get("HA_E2E_FIXTURE_TTS_MODEL", "speaches-ai/Kokoro-82M-v1.0-ONNX")
VOICE = os.environ.get("HA_E2E_FIXTURE_VOICE", "af_heart")
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
CHANNELS = 1
TRAILING_SILENCE_SECONDS = 1.5

PHRASES = {
    "kitchen_lights": "Turn on the kitchen lights.",
    "weather_berlin": "What's the weather like tomorrow in Berlin?",
    "timer_five_minutes": "Set a timer for five minutes.",
}


def _decode_wav(data: bytes) -> tuple[int, int, int, array.array[int]]:
    with wave.open(io.BytesIO(data), "rb") as wav:
        rate = wav.getframerate()
        width = wav.getsampwidth()
        channels = wav.getnchannels()
        frames = wav.readframes(wav.getnframes())
    if width != SAMPLE_WIDTH:
        raise RuntimeError(f"expected 16-bit Kokoro audio, got {width * 8}-bit")
    samples = array.array("h")
    samples.frombytes(frames)
    if sys.byteorder != "little":
        samples.byteswap()
    return rate, width, channels, samples


def _to_mono(samples: array.array[int], channels: int) -> array.array[int]:
    if channels == 1:
        return samples
    mono = array.array("h")
    for frame_start in range(0, len(samples), channels):
        frame = samples[frame_start : frame_start + channels]
        if len(frame) < channels:
            break
        mono.append(round(sum(frame) / channels))
    return mono


def _resample(samples: array.array[int], source_rate: int) -> array.array[int]:
    if source_rate == SAMPLE_RATE:
        return samples
    output_frames = round(len(samples) * SAMPLE_RATE / source_rate)
    output = array.array("h")
    for index in range(output_frames):
        position = index * source_rate / SAMPLE_RATE
        left = int(position)
        fraction = position - left
        right = min(left + 1, len(samples) - 1)
        output.append(round(samples[left] * (1.0 - fraction) + samples[right] * fraction))
    return output


def _wav_bytes(samples: array.array[int]) -> bytes:
    if sys.byteorder != "little":
        samples = array.array("h", samples)
        samples.byteswap()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(samples.tobytes())
    return buffer.getvalue()


def _synthesize(client: httpx.Client, phrase: str) -> bytes:
    response = client.post(
        f"{SPEACHES_URL}/audio/speech",
        json={
            "model": TTS_MODEL,
            "voice": VOICE,
            "input": phrase,
            "response_format": "wav",
            "speed": 1.0,
        },
        timeout=120.0,
    )
    if not response.is_success:
        raise RuntimeError(f"Speaches TTS returned HTTP {response.status_code}: {response.text[:2000]}")
    if not response.content.startswith(b"RIFF"):
        raise RuntimeError("Speaches TTS did not return a WAV file")
    return response.content


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "speaches_url": SPEACHES_URL,
        "model": TTS_MODEL,
        "voice": VOICE,
        "sample_rate": SAMPLE_RATE,
        "width": SAMPLE_WIDTH,
        "channels": CHANNELS,
        "trailing_silence_seconds": TRAILING_SILENCE_SECONDS,
        "fixtures": [],
    }
    fixtures: list[dict[str, object]] = []

    with httpx.Client() as client:
        for fixture_id, phrase in PHRASES.items():
            LOGGER.info("Generating %s: %s", fixture_id, phrase)
            source_rate, _, source_channels, samples = _decode_wav(_synthesize(client, phrase))
            mono = _to_mono(samples, source_channels)
            resampled = _resample(mono, source_rate)
            silence = array.array("h", [0]) * round(SAMPLE_RATE * TRAILING_SILENCE_SECONDS)
            wav_bytes = _wav_bytes(resampled + silence)
            output_path = OUTPUT_DIR / f"{fixture_id}.wav"
            output_path.write_bytes(wav_bytes)
            duration = (len(resampled) + len(silence)) / SAMPLE_RATE
            fixtures.append(
                {
                    "id": fixture_id,
                    "file": output_path.name,
                    "text": phrase,
                    "sha256": hashlib.sha256(wav_bytes).hexdigest(),
                    "duration_seconds": round(duration, 3),
                }
            )
            LOGGER.info("Wrote %s (%.2fs, %d bytes)", output_path, duration, len(wav_bytes))

    manifest["fixtures"] = fixtures
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Wrote %s", manifest_path)


if __name__ == "__main__":
    main()
