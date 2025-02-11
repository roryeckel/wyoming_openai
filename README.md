# Wyoming OpenAI

OpenAI-Compatible Proxy Client for the Wyoming Protocol

**Author:** Rory Eckel

Note: This project is not affiliated with OpenAI or the Wyoming project.

## Overview

This project introduces an OpenAI-compatible proxy server that integrates seamlessly with the [Wyoming](https://github.com/rhasspy/wyoming) framework. It provides transcription (Automatic Speech Recognition - ASR) and text-to-speech synthesis (TTS) capabilities using OpenAI-compatible APIs. By acting as a bridge between the Wyoming protocol and OpenAI's services, this proxy server enables efficient utilization of local ASR and TTS models. This is particularly advantageous for homelab users who aim to consolidate multiple protocols into a single server, thereby addressing resource constraints.

## Objectives

1. **Wyoming Server, OpenAI-compatible Client**: Function as an intermediary between the Wyoming protocol and OpenAI's ASR and TTS services.
2. **Service Consolidation**: Allow users operating different protocols to run them on a single server without needing separate instances for each service.
Example: Sharing TTS/STT services between Open WebUI and Home Assistant.
3. **Asynchronous Processing**: Enable efficient handling of multiple requests by supporting asynchronous processing of audio streams.

## Terminology

- **TTS (Text-to-Speech)**: The process of converting text into audible speech output.
- **ASR (Automatic Speech Recognition) / STT (Speech-to-Text)**: Technologies that convert spoken language into written text. ASR and STT are often used interchangeably to describe this function.

## Installation (Local Development)

### Prerequisites

- Tested with Python 3.12 or later
- OpenAI API key(s) if using proprietary models

### Instructions

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

## Docker (Recommended)

### Prerequisites

- Ensure you have [Docker](https://www.docker.com/products/docker-desktop) and [Docker Compose](https://docs.docker.com/compose/install/) installed on your system.

### Deployment Options

You can deploy the Wyoming OpenAI proxy server in different environments depending on whether you are using official OpenAI services or a local alternative like Speaches. Below are example scenarios:

#### 1. Deploying with Official OpenAI Services

To set up the Wyoming OpenAI proxy to work with official OpenAI APIs, follow these steps:

- **Environment Variables**: Create a `.env` file in your project directory that includes necessary environment variables such as `STT_OPENAI_KEY`, `TTS_OPENAI_KEY`.

- **Docker Compose Configuration**: Use the provided `docker-compose.yml` template. This setup binds a Wyoming server to port 10300 and uses environment variables for OpenAI URLs, model configurations, and voices as specified in the compose file.

- **Command**:
  
  ```bash
  docker-compose -f docker-compose.yml up -d
  ```

#### 2. Deploying with Speaches Local Service

If you prefer using a local service like Speaches instead of official OpenAI services, follow these instructions:

- **Docker Compose Configuration**: Use the `docker-compose.speaches.yml` template which includes configuration for both the Wyoming OpenAI proxy and the Speaches service.

- **Speaches Setup**:
  - The Speaches container is configured with specific model settings (`Systran/faster-distil-whisper-large-v3` for STT and `hexgrad/Kokoro-82M` for TTS).
  - It uses a local port (8000) to expose the Speaches service.
  - NVIDIA GPU support is enabled, so ensure your system has an appropriate setup if you plan to utilize GPU resources.

- **Command**:
  
  ```bash
  docker-compose -f docker-compose.speaches.yml up -d
  ```

#### 3. Development with Docker

If you are developing the Wyoming OpenAI proxy server and want to build it from source, use the `docker-compose.dev.yml` file along with the base configuration.

- **Command**:
  
  ```bash
  docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build
  ```

#### 4. Development with Speaches Local Service

For a development setup using the Speaches local service, combine `docker-compose.speaches.yml` and `docker-compose.dev.yml`.

- **Command**:
  
  ```bash
  docker-compose -f docker-compose.speaches.yml -f docker-compose.dev.yml up -d --build
  ```

### General Deployment Steps

1. **Start Services**: Run the appropriate Docker Compose command based on your deployment option.
2. **Verify Deployment**: Ensure that all services are running by checking the logs with `docker-compose logs -f` or accessing the Wyoming OpenAI proxy through its exposed port (e.g., 10300) to ensure it responds as expected.
3. **Configuration Changes**: You can modify environment variables in the `.env` file or directly within your Docker Compose configuration files to adjust settings such as languages, models, and voices without rebuilding containers.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests. For major changes, please first discuss the proposed changes in an issue.

### Future Plans (Descending Priority)

- Improved streaming support directly to OpenAI APIs
- Reverse direction support (Server for OpenAI compatible endpoints - possibly FastAPI)
- OpenAI Realtime API
