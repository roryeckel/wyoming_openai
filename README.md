# wyoming_openai
OpenAI-Compatible Proxy Client for the Wyoming Protocol

Maintained by Rory Eckel

## Overview
This project provides an OpenAI-compatible proxy server for the [Wyoming](https://github.com/rhasspy/wyoming) framework. It enables transcription (ASR - Automatic Speech Recognition) and text-to-speech synthesis (TTS) using OpenAI compatible APIs, making it easy to integrate these powerful services into your Wyoming-based applications. This is especially useful for those hosting their own ASR and TTS models locally, as it is usually infeasible to run servers for both protocols simultaneously due to resource constraints.


## Objectives
1. **Proxy Server for OpenAI APIs**: Act as an intermediary between Wyoming protocol and OpenAI's ASR and TTS services.
2. **Support Multiple Models/Voices**: Allow configuration of specific ASR models and TTS voices via command line arguments.
3. **Streamline Audio Handling**: Efficiently manage audio streams for both transcription and synthesis.

## Installation Instructions

### Prerequisites
- Tested with Python 3.12 or later
- OpenAI API key(s) if using proprietary models

### Steps to Install

1. **Clone the Repository**
   ```bash
   git clone https://github.com/roryeckel/wyoming-openai.git
   cd wyoming-openai
   ```

2. **Create a Virtual Environment** (optional but recommended)
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure OpenAI API Keys** (ensure you have your OpenAI API keys ready)

## Command Line Arguments

The proxy server can be configured using several command line arguments to tailor its behavior to your specific needs.

### Example Usage
```bash
python -m wyoming_openai \
    --uri tcp://0.0.0.0:10300 \
    --log-level INFO \
    --stt-openai-key YOUR_STT_API_KEY_HERE \
    --stt-openai-url https://api.openai.com/v1 \
    --stt-patterns '["^whisper-", "transcribe", "stt", "speech.*text"]' \
    --tts-openai-key YOUR_TTS_API_KEY_HERE \
    --tts-openai-url https://api.openai.com/v1 \
    --tts-patterns '["^tts-", "speech.*synthesis", "text.*speech"]' \
    --tts-voices alloy echo fable onyx nova shimmer
```

## Environment Variables

In addition to using command-line arguments, you can configure the Wyoming OpenAI proxy server via environment variables. This is especially useful for containerized deployments.

### Table of Environment Variables for Command Line Arguments

| **Command Line Argument**               | **Environment Variable**                   | **Description**                                             |
|-----------------------------------------|--------------------------------------------|-------------------------------------------------------------|
| `--uri`                                 | `WYOMING_URI`                              | The URI for the Wyoming server to bind to.                  |
| `--log-level`                           | `WYOMING_LOG_LEVEL`                        | Sets the logging level (e.g., INFO, DEBUG).                 |
| `--stt-openai-key`                      | `STT_OPENAI_KEY`                           | The API key for accessing OpenAI's speech-to-text services. |
| `--stt-openai-url`                      | `STT_OPENAI_URL`                           | The URL for OpenAI's STT endpoint.                          |
| `--stt-patterns`                        | `STT_PATTERNS`                             | Patterns to match for using the STT service.                |
| `--tts-openai-key`                      | `TTS_OPENAI_KEY`                           | The API key for accessing OpenAI's text-to-speech services. |
| `--tts-openai-url`                      | `TTS_OPENAI_URL`                           | The URL for OpenAI's TTS endpoint.                          |
| `--tts-patterns`                        | `TTS_PATTERNS`                             | Patterns to match for using the TTS service.                |
| `--tts-voices`                          | `TTS_VOICES`                               | Space-separated list of voices for TTS.                     |

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests. For major changes, please first discuss the proposed changes in an issue.

---
