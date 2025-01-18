# Wyoming OpenAI

OpenAI-Compatible Proxy Client for the Wyoming Protocol

**Author:** Rory Eckel

## Overview

This project introduces an OpenAI-compatible proxy server that integrates seamlessly with the [Wyoming](https://github.com/rhasspy/wyoming) framework. It provides transcription (Automatic Speech Recognition - ASR) and text-to-speech synthesis (TTS) capabilities using OpenAI-compatible APIs. By acting as a bridge between the Wyoming protocol and OpenAI's services, this proxy server enables efficient utilization of local ASR and TTS models. This is particularly advantageous for homelab users who aim to consolidate multiple protocols into a single server, thereby addressing resource constraints.

## Objectives

1. **OpenAI API Proxy Server**: Function as an intermediary between the Wyoming protocol and OpenAI's ASR and TTS services.
2. **Service Consolidation**: Allow users operating different protocols to run them on a single server without needing separate instances for each service.
3. **Asynchronous Processing**: Enable efficient handling of multiple requests by supporting asynchronous processing of audio streams.

## Terminology

- **TTS (Text-to-Speech)**: The process of converting text into audible speech output.
- **ASR (Automatic Speech Recognition) / STT (Speech-to-Text)**: Technologies that convert spoken language into written text. ASR and STT are often used interchangeably to describe this function.

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
    --languages en \
    --stt-openai-key YOUR_STT_API_KEY_HERE \
    --stt-openai-url https://api.openai.com/v1 \
    --stt-models whisper-1 \
    --tts-openai-key YOUR_TTS_API_KEY_HERE \
    --tts-openai-url https://api.openai.com/v1 \
    --tts-models tts-1 \
    --tts-voices alloy echo fable onyx nova shimmer
```

## Environment Variables

In addition to using command-line arguments, you can configure the Wyoming OpenAI proxy server via environment variables. This is especially useful for containerized deployments.

### Table of Environment Variables for Command Line Arguments

| **Command Line Argument**               | **Environment Variable**                   | **Description**                                             |
|-----------------------------------------|--------------------------------------------|-------------------------------------------------------------|
| `--uri`                                 | `WYOMING_URI`                              | The URI for the Wyoming server to bind to.                  |
| `--log-level`                           | `WYOMING_LOG_LEVEL`                        | Sets the logging level (e.g., INFO, DEBUG).                 |
| `--languages`                           | `WYOMING_LANGUAGES`                        | Space-separated list of supported languages to avertise.    |
| `--stt-openai-key`                      | `STT_OPENAI_KEY`                           | The API key for accessing OpenAI's speech-to-text services. |
| `--stt-openai-url`                      | `STT_OPENAI_URL`                           | The URL for OpenAI's STT endpoint.                          |
| `--stt-models`                          | `STT_MODELS`                               | Space-separated list of models to use for the STT service.  |
| `--tts-openai-key`                      | `TTS_OPENAI_KEY`                           | The API key for accessing OpenAI's text-to-speech services. |
| `--tts-openai-url`                      | `TTS_OPENAI_URL`                           | The URL for OpenAI's TTS endpoint.                          |
| `--tts-models`                          | `TTS_MODELS`                               | Space-separated list of models to use for the TTS service.  |
| `--tts-voices`                          | `TTS_VOICES`                               | Space-separated list of voices for TTS.                     |

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests. For major changes, please first discuss the proposed changes in an issue.

### Future Plans

- OpenAI Realtime API
- Improved streaming support directly to OpenAI APIs